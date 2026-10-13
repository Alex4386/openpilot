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

Every frame goes to the server, in order, even when its result will be too late to use: the
model's temporal state is built at 20Hz, so a skipped frame corrupts its history. When the
server falls behind by more than MAX_BACKLOG frames, the sequence restarts (and re-warms).
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
USB_TARGET = "usb"  # server is whatever is on the other end of a USB-C cable (see usb_network)
MODEL_NAME = "driving_policy"
BUDGET_S = 0.040  # submit -> collect; warp + this + publishing must fit the 50ms frame
RECONNECT_INTERVAL_S = 2.0
IO_TIMEOUT_S = 1.0  # a stalled server, not a slow frame; slow frames are handled by BUDGET_S
HEALTHY_FRAMES_TO_ENGAGE = 20  # consecutive on-time warm responses before outputs are used
MAX_BACKLOG = 3  # frames waiting to be sent; beyond this the server can't keep up at 20Hz
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
    self._cv = threading.Condition()
    self._queue: collections.deque[tuple[int, bytes, int]] = collections.deque()  # (seq, body, header_len) to send
    self._result: tuple[int, np.ndarray | None, bool] | None = None  # (seq, outputs, warm)
    self._submitted: int | None = None
    self._submit_time = 0.
    self._seq = 0  # request id, never reused, so a late result can't pass for a newer one
    self._sequence_id = 0
    self._start = True  # the next frame starts a new sequence (server state reset)
    self.restarts = 0
    self._connected = False
    self._healthy = 0
    self._engaged = False  # the last collect() returned outputs
    self._engaged_once = False
    self._grace_start = time.monotonic()
    threading.Thread(target=self._worker, daemon=True, name="gpu_on_lan").start()

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
        return
      if len(self._queue) >= MAX_BACKLOG:
        # the server can't keep up; skipping frames would corrupt its history, so start over
        cloudlog.warning(f"gpu on lan: server {len(self._queue)} frames behind, restarting the sequence")
        self._restart_sequence()
      tensors = {
        "new_img": new_img,
        "desire": desire.astype(np.float32),
        "traffic_convention": traffic_convention.astype(np.float32),
        "action_t": action_t.astype(np.float32),
      }
      body, header_len = encode_infer_request(self._seq, self._sequence_id, self._start, tensors)
      self._start = False
      self._queue.append((self._seq, body, header_len))
      self._submitted = self._seq
      self._submit_time = time.monotonic()
      self._seq += 1
      self._cv.notify_all()

  def _restart_sequence(self) -> None:
    """Under self._cv."""
    self._queue.clear()
    self._sequence_id, self._start, self._healthy = random.getrandbits(63), True, 0
    self.restarts += 1

  def collect(self) -> np.ndarray | None:
    """Raw big model outputs for the last submit if they arrive within budget and are trustworthy."""
    outputs = self._collect()
    self._engaged = outputs is not None
    self._engaged_once |= self._engaged
    return outputs

  def _collect(self) -> np.ndarray | None:
    seq = self._submitted
    if seq is None:
      self._healthy = 0
      return None
    deadline = self._submit_time + BUDGET_S
    with self._cv:
      while self._result is None or self._result[0] != seq:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._connected:
          self._healthy = 0
          return None
        self._cv.wait(remaining)
      _, outputs, warm = self._result
    if outputs is None or not warm or not np.all(np.isfinite(outputs)):
      self._healthy = 0
      return None
    self._healthy += 1
    return outputs if self._healthy >= HEALTHY_FRAMES_TO_ENGAGE else None

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
            self._restart_sequence()
            self._connected = True
            if not self._engaged_once:
              self._grace_start = time.monotonic()  # the server may come up mid-drive
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
        result = (seq, outputs, bool(header.get("parameters", {}).get("sequence_warm", False)))
      except Exception as e:
        cloudlog.warning(f"gpu on lan: dropping connection ({type(e).__name__}: {e})")
        conn.close()
        conn, result = None, (seq, None, False)

      with self._cv:
        self._result = result
        self._connected = conn is not None
        if conn is None:
          self._queue.clear()  # a new sequence starts on reconnect
        self._cv.notify_all()
