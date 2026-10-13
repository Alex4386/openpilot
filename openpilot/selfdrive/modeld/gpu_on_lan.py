"""
GPU on LAN: the chestnut's big driving model, on a computer on the local network.

Speaks the KServe v2 / Open Inference Protocol over HTTP with the binary tensor data
extension, and the Triton sequence extension (sequence_id / sequence_start) for the
model's temporal state. The inputs are the driving model's non-state inputs (new_img, desire,
traffic_convention, action_t); the server keeps the state_* tensors, like Triton's implicit
sequence state.

modeld uses it the way it uses the chestnut: it connects while modeld starts (Big Model
Loading), then the big model drives every frame instead of the on-device one: modeld warps the
frame, sends it and waits. A slow server means dropped camera frames (Driving Model Lagging); an
error or a stall makes modeld fall back to the on-device model for the rest of the drive (Big
Model Failed).
"""
import http.client
import json
import random
import socket
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
  # a copy: the parser works in place, and frombuffer over bytes is read-only
  return header, np.frombuffer(body, dtype=np.float32, count=size // 4, offset=header_len).copy()


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

  def connect(self, timeout: float) -> bool:
    """Blocks until the server answers with its model (it loads and warms it up before it listens), or timeout."""
    deadline = time.monotonic() + timeout
    while True:
      try:
        self._conn = self._connect()
        return True
      except Exception as e:
        if time.monotonic() > deadline:
          cloudlog.exception(f"gpu on lan: no server at {'usb' if self.usb else self.host}:{self.port}: {e}")
          return False
        time.sleep(CONNECT_RETRY_S)

  def infer(self, new_img: np.ndarray, desire: np.ndarray, traffic_convention: np.ndarray, action_t: np.ndarray) -> np.ndarray:
    """The big model's raw outputs for this frame (slice with output_slices). Raises if the server
    fails or stalls for TIMEOUT_S."""
    assert self._conn is not None
    tensors = {
      "new_img": new_img,
      "desire": desire.astype(np.float32),
      "traffic_convention": traffic_convention.astype(np.float32),
      "action_t": action_t.astype(np.float32),
    }
    # a model bundle may lay out the same inputs differently, e.g. desire (1, 8) vs (8,)
    for name, shape in self._input_shapes.items():
      if name in tensors:
        tensors[name] = tensors[name].reshape(shape)
    body, header_len = encode_infer_request(self._seq, self._sequence_id, self._seq == 0, tensors)
    self._seq += 1
    self._conn.request("POST", f"/v2/models/{MODEL_NAME}/infer", body=body, headers={
      "Content-Type": "application/octet-stream",
      "Inference-Header-Content-Length": str(header_len),
    })
    resp = self._conn.getresponse()
    data = resp.read()
    if resp.status != 200:
      raise ConnectionError(f"gpu on lan: infer failed: {resp.status} {data[:200]!r}")
    outputs = decode_infer_response(data, int(resp.getheader("Inference-Header-Content-Length")))[1]
    if not np.all(np.isfinite(outputs)):
      raise RuntimeError("gpu on lan: model output not finite")
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
