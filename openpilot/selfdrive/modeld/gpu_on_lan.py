"""
GPU on LAN: run the big driving model on a machine on the local network.

Speaks the KServe v2 / Open Inference Protocol over HTTP with the binary tensor data
extension, and the Triton sequence extension (sequence_id / sequence_start) for the
model's temporal state. The inputs are the driving model's non-state inputs (new_img, desire,
traffic_convention, action_t); the server keeps the state_* tensors, like Triton's implicit
sequence state.

modeld submits right after the camera warp and collects after the local model finishes, so
the remote model runs in parallel with the local one. The local model always runs; a slow or
missing server just means its output is used for that frame.

Frames go to the server in order. When it falls MAX_BACKLOG frames behind, frames are dropped
until it catches up, like modeld drops camera frames when a slow (chestnut) model can't keep
up: the history just has a hole, and outputs aren't used again until HEALTHY_FRAMES_TO_ENGAGE
on-time frames in a row.

CONNECTIONS requests are in flight at once, so the next frames upload while the server
computes; the server steps a sequence in id order whatever order they arrive in.
"""
import collections
import http.client
import json
import random
import socket
import threading
import time

import numpy as np

from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import usb_network

DEFAULT_PORT = 8000  # KServe / Triton HTTP default
USB_TARGET = "usb"  # server is whatever is on the other end of the USB Ethernet link (see usb_network)
MODEL_NAME = "driving_policy"
BUDGET_S = 0.040  # submit -> collect; warp + this + publishing must fit the 50ms frame
RECONNECT_INTERVAL_S = 2.0
IO_TIMEOUT_S = 1.0  # a stalled server, not a slow frame; slow frames are handled by BUDGET_S
HEALTHY_FRAMES_TO_ENGAGE = 20  # consecutive on-time warm responses before outputs are used
MAX_BACKLOG = 3  # frames waiting to be sent; beyond this new frames are dropped
CONNECTIONS = 3  # requests in flight at once
ENGAGE_GRACE_S = 10.  # connecting + warming the server's history before it counts as failing


def parse_host(target: str) -> tuple[str, int]:
  target = target.strip()
  if target.startswith('['):  # [v6]:port
    host, _, port = target[1:].partition(']')
    return host, int(port.lstrip(':') or DEFAULT_PORT)
  if target.count(':') == 1:
    host, port = target.split(':')
    return host, int(port)
  return target, DEFAULT_PORT


def encode_infer_request(seq: int, sequence_id: int, start: bool, tensors: dict[str, np.ndarray]) -> tuple[bytes, int]:
  datatypes = {np.dtype(np.uint8): "UINT8", np.dtype(np.float32): "FP32"}
  header = {
    "id": str(seq),
    "parameters": {"sequence_id": sequence_id, "sequence_start": start, "sequence_end": False},
    "inputs": [{"name": k, "shape": list(v.shape), "datatype": datatypes[v.dtype],
                "parameters": {"binary_data_size": v.nbytes}} for k, v in tensors.items()],
    "outputs": [{"name": "outputs", "parameters": {"binary_data": True}}],
  }
  header_bytes = json.dumps(header, separators=(',', ':')).encode()
  return header_bytes + b''.join(np.ascontiguousarray(v).tobytes() for v in tensors.values()), len(header_bytes)


def decode_infer_response(body: bytes, header_len: int) -> tuple[dict, np.ndarray]:
  header = json.loads(body[:header_len])
  out = next(o for o in header["outputs"] if o["name"] == "outputs")
  size = out["parameters"]["binary_data_size"]
  return header, np.frombuffer(body, dtype=np.float32, count=size // 4, offset=header_len)


class GpuOnLan:
  def __init__(self, target: str):
    host, self.port = parse_host(target.strip() or USB_TARGET)
    self.usb = host.lower() == USB_TARGET
    self.host: str | None = None if self.usb else host
    self.output_slices: dict[str, slice] | None = None
    self.model_name: str | None = None  # what the server says it runs, e.g. "Cinque Terre Model v2"
    self._input_shapes: dict[str, list[int]] = {}  # the server model's, from its metadata
    self._published: tuple[bool | None, str | None] = (None, None)  # GpuOnLanActive, GpuOnLanModel
    self._cv = threading.Condition()
    self._queue: collections.deque[tuple[int, bytes, int]] = collections.deque()  # (seq, body, header_len) to send
    self._results: dict[int, tuple[np.ndarray | None, bool]] = {}  # seq -> (outputs, warm)
    self._submitted: int | None = None
    self._submit_time = 0.
    self._seq = 0  # request id, never reused, so a late result can't pass for a newer one
    self._sequence_id = 0
    self._start = True  # the next frame starts a new sequence (server state reset)
    self.dropped = 0  # frames not sent because the server was behind
    self.reason = "connecting"  # why the big model is or isn't driving this frame, logged on change
    self._logged_reason = ""
    self._logged_time = 0.
    self._connections = 0  # workers with a live connection
    self._healthy = 0
    self._engaged = False  # the last collect() returned outputs
    self._engaged_once = False
    self._grace_start = time.monotonic()
    for i in range(CONNECTIONS):
      threading.Thread(target=self._worker, daemon=True, name=f"gpu_on_lan{i}").start()

  @property
  def _connected(self) -> bool:
    return self._connections > 0

  @property
  def active(self) -> bool:
    return self._connected and self._healthy >= HEALTHY_FRAMES_TO_ENGAGE

  def state(self) -> bool | None:
    """True: the big model on the server drives. False: it doesn't, because the server is
    unreachable or too slow, or dropped out after driving. None: still getting there."""
    if self._engaged:
      return True
    if self._engaged_once or time.monotonic() - self._grace_start > ENGAGE_GRACE_S:
      return False
    return None

  def submit(self, new_img: np.ndarray, desire: np.ndarray, traffic_convention: np.ndarray, action_t: np.ndarray) -> None:
    """Called right after the warp; never blocks on the network."""
    self._submitted = None
    with self._cv:
      if not self._connected:
        self.reason = "not connected"
        return
      if len(self._queue) >= MAX_BACKLOG:
        self.dropped += 1
        self._healthy = 0
        self.reason = f"server {MAX_BACKLOG} frames behind, dropping frames"
        return
      tensors = {
        "new_img": new_img,
        "desire": desire.astype(np.float32),
        "traffic_convention": traffic_convention.astype(np.float32),
        "action_t": action_t.astype(np.float32),
      }
      # a model bundle may lay out the same inputs differently, e.g. desire (1, 8) vs (8,)
      for name, shape in self._input_shapes.items():
        if name in tensors:
          if tensors[name].size != np.prod(shape):
            cloudlog.error(f"gpu on lan: {name} {tensors[name].shape} doesn't fit the server model's {shape}")
            return
          tensors[name] = tensors[name].reshape(shape)
      body, header_len = encode_infer_request(self._seq, self._sequence_id, self._start, tensors)
      self._start = False
      self._queue.append((self._seq, body, header_len))
      self._submitted = self._seq
      self._submit_time = time.monotonic()
      self._seq += 1
      self._cv.notify_all()

  def publish(self, params) -> None:
    """GpuOnLanActive (unset while connecting) and GpuOnLanModel for the UI, like ChestnutActive."""
    state = (self.state(), self.model_name)
    if state == self._published:
      return
    if state[0] != self._published[0]:
      if state[0] is None:
        params.remove("GpuOnLanActive")
      else:
        params.put_bool("GpuOnLanActive", state[0])
    if state[1] != self._published[1]:
      if state[1]:
        params.put("GpuOnLanModel", state[1])
      else:
        params.remove("GpuOnLanModel")
    self._published = state

  def _restart_sequence(self) -> None:
    """A new session on the server, for a fresh connection. Under self._cv."""
    self._queue.clear()
    self._sequence_id, self._start, self._healthy = random.getrandbits(63), True, 0

  def collect(self) -> np.ndarray | None:
    """Raw big model outputs for the last submit if they arrive within budget and are trustworthy."""
    outputs = self._collect()
    self._engaged = outputs is not None
    self._engaged_once |= self._engaged
    if self.reason != self._logged_reason and time.monotonic() - self._logged_time > 1.:  # at most once a second
      cloudlog.warning(f"gpu on lan: big model {'driving' if self._engaged else 'not used'}: {self.reason}")
      self._logged_reason, self._logged_time = self.reason, time.monotonic()
    return outputs

  def _collect(self) -> np.ndarray | None:
    seq = self._submitted
    if seq is None:
      self._healthy = 0
      return None
    deadline = self._submit_time + BUDGET_S
    with self._cv:
      while seq not in self._results:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._connected:
          self._healthy = 0
          self.reason = f"late, no result within {BUDGET_S * 1e3:.0f} ms" if self._connected else "connection lost"
          return None
        self._cv.wait(remaining)
      outputs, warm = self._results[seq]
    if outputs is None or not warm or not np.all(np.isfinite(outputs)):
      self._healthy = 0
      self.reason = "request failed" if outputs is None else "server model warming up" if not warm else "non-finite output"
      return None
    self._healthy += 1
    if self._healthy < HEALTHY_FRAMES_TO_ENGAGE:
      self.reason = f"on time, waiting for {HEALTHY_FRAMES_TO_ENGAGE} in a row"
      return None
    self.reason = "on time"
    return outputs

  def _resolve(self) -> str:
    if self.host is not None and not self.usb:
      return self.host
    # the link itself is brought up from settings (UsbNetworkEnabled), not from modeld
    peer = usb_network.peer_address()
    if peer is None:
      raise ConnectionError("no laptop on the USB network")
    self.host = peer
    return peer

  def _connect(self) -> http.client.HTTPConnection:
    conn = http.client.HTTPConnection(self._resolve(), self.port, timeout=IO_TIMEOUT_S)
    conn.request("GET", "/v2/health/ready")
    resp = conn.getresponse()
    resp.read()
    if resp.status != 200:
      raise ConnectionError(f"server not ready: {resp.status}")
    conn.request("GET", f"/v2/models/{MODEL_NAME}")
    meta = json.loads(conn.getresponse().read())
    slices = json.loads(meta["parameters"]["output_slices"])
    self.output_slices = {k: slice(*v) for k, v in slices.items()}
    self.model_name = meta["parameters"].get("model_name")
    self._input_shapes = {i["name"]: i["shape"] for i in meta.get("inputs", [])}
    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    cloudlog.warning(f"gpu on lan: connected to {self.host}:{self.port}, model {meta['parameters'].get('model_checkpoint')}")
    return conn

  def _worker(self) -> None:
    conn = None
    while True:
      if conn is None:
        try:
          conn = self._connect()
          with self._cv:
            if not self._connected:
              self._restart_sequence()
              if not self._engaged_once:
                self._grace_start = time.monotonic()  # the server may come up mid-drive
            self._connections += 1
        except Exception as e:
          cloudlog.debug(f"gpu on lan: connect to {'usb' if self.usb else self.host}:{self.port} failed: {e}")
          time.sleep(RECONNECT_INTERVAL_S)
          continue

      with self._cv:
        while not self._queue:
          self._cv.wait()
        seq, body, header_len = self._queue.popleft()

      try:
        conn.request("POST", f"/v2/models/{MODEL_NAME}/infer", body=body, headers={
          "Content-Type": "application/octet-stream",
          "Inference-Header-Content-Length": str(header_len),
        })
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
          raise ConnectionError(f"infer failed: {resp.status} {data[:200]!r}")
        header, outputs = decode_infer_response(data, int(resp.getheader("Inference-Header-Content-Length")))
        result = (outputs, bool(header.get("parameters", {}).get("sequence_warm", False)))
      except Exception as e:
        cloudlog.warning(f"gpu on lan: dropping connection ({type(e).__name__}: {e})")
        conn.close()
        conn, result = None, (None, False)

      with self._cv:
        self._results[seq] = result
        for old in [k for k in self._results if k < seq - 2 * CONNECTIONS - MAX_BACKLOG]:
          del self._results[old]
        if conn is None:
          self._connections -= 1  # this frame never made it: the server's history has a hole
        self._cv.notify_all()
