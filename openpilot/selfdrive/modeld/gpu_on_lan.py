"""
GPU on LAN: the chestnut's big driving model, on a computer on the local network.

Speaks the KServe v2 / Open Inference Protocol over HTTP with the binary tensor data
extension, and the Triton sequence extension (sequence_id / sequence_start) for the
model's temporal state. The inputs are the driving model's non-state inputs (new_img, desire,
traffic_convention, action_t); the server keeps the state_* tensors, like Triton's implicit
sequence state.

modeld uses it the way it uses the chestnut: it connects while modeld starts (Big Model
Loading), then the big model drives every frame, and modeld waits for it. A slow server means
dropped camera frames (Driving Model Lagging); an error or a stall makes modeld fall back to
the on-device model for the rest of the drive (Big Model Failed). The request goes out right
after the warp, so the on-device model runs meanwhile and is warm if it has to take over.
"""
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
TIMEOUT_S = 3.0  # like the chestnut's HCQDEV_WAIT_TIMEOUT_MS
CONNECT_RETRY_S = 1.0


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
    self._conn: http.client.HTTPConnection | None = None
    self._seq = 0
    self._sequence_id = random.getrandbits(63)
    self._request: threading.Thread | None = None
    self._result: np.ndarray | Exception | None = None

  def connect(self, timeout: float) -> bool:
    """Blocks until the server answers with its model and has run it once, or timeout."""
    deadline = time.monotonic() + timeout
    while True:
      try:
        self._conn = self._connect()
        self._warmup()
        return True
      except Exception as e:
        if time.monotonic() > deadline:
          cloudlog.exception(f"gpu on lan: no server at {'usb' if self.usb else self.host}:{self.port}: {e}")
          return False
        time.sleep(CONNECT_RETRY_S)

  def _warmup(self) -> None:
    """One throwaway frame through the whole path, like the chestnut model's warmup() while loading;
    the real frames then start a fresh sequence."""
    zeros = {name: np.zeros(shape, dtype=np.uint8 if name == "new_img" else np.float32) for name, shape in self._input_shapes.items()}
    self._sequence_id = random.getrandbits(63)
    self._infer(0, zeros)
    if isinstance(self._result, Exception):
      raise self._result
    self._sequence_id = random.getrandbits(63)

  def submit(self, new_img: np.ndarray, desire: np.ndarray, traffic_convention: np.ndarray, action_t: np.ndarray) -> None:
    """Right after the warp: sends this frame, the on-device model runs while it's in flight."""
    tensors = {
      "new_img": new_img,
      "desire": desire.astype(np.float32),
      "traffic_convention": traffic_convention.astype(np.float32),
      "action_t": action_t.astype(np.float32),
    }
    self._result = None
    self._request = threading.Thread(target=self._infer, args=(self._seq, tensors), daemon=True)
    self._request.start()
    self._seq += 1

  def result(self) -> np.ndarray:
    """The big model's raw outputs for the submitted frame. Raises if the server fails or stalls."""
    assert self._request is not None, "nothing submitted"
    self._request.join(TIMEOUT_S)
    if self._request.is_alive():
      raise TimeoutError(f"gpu on lan: no result in {TIMEOUT_S:.0f}s")
    if isinstance(self._result, Exception):
      raise self._result
    assert self._result is not None
    if not np.all(np.isfinite(self._result)):
      raise RuntimeError("gpu on lan: model output not finite")
    return self._result

  def _infer(self, seq: int, tensors: dict[str, np.ndarray]) -> None:
    """Any failure ends up in result(), never in modeld's warp."""
    try:
      # a model bundle may lay out the same inputs differently, e.g. desire (1, 8) vs (8,)
      for name, shape in self._input_shapes.items():
        if name in tensors:
          tensors[name] = tensors[name].reshape(shape)
      body, header_len = encode_infer_request(seq, self._sequence_id, seq == 0, tensors)
      assert self._conn is not None
      self._conn.request("POST", f"/v2/models/{MODEL_NAME}/infer", body=body, headers={
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": str(header_len),
      })
      resp = self._conn.getresponse()
      data = resp.read()
      if resp.status != 200:
        raise ConnectionError(f"infer failed: {resp.status} {data[:200]!r}")
      self._result = decode_infer_response(data, int(resp.getheader("Inference-Header-Content-Length")))[1]
    except Exception as e:
      self._result = e

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
    conn = http.client.HTTPConnection(self._resolve(), self.port, timeout=TIMEOUT_S)
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
