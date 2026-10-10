import json

import numpy as np

from openpilot.selfdrive.modeld.gpu_on_lan import parse_host, encode_infer_request, decode_infer_response, DEFAULT_PORT


def test_parse_host():
  assert parse_host("192.168.43.100") == ("192.168.43.100", DEFAULT_PORT)
  assert parse_host(" mac.local:9000 ") == ("mac.local", 9000)
  assert parse_host("[fe80::1]:8001") == ("fe80::1", 8001)
  assert parse_host("[fe80::1]") == ("fe80::1", DEFAULT_PORT)


def test_request_layout():
  tensors = {"img": np.arange(12, dtype=np.uint8).reshape(1, 3, 4), "action_t": np.array([[0.25, 0.5]], dtype=np.float32)}
  body, header_len = encode_infer_request(5, 42, True, tensors)
  header = json.loads(body[:header_len])
  assert header["id"] == "5"
  assert header["parameters"] == {"sequence_id": 42, "sequence_start": True, "sequence_end": False}
  assert [(i["name"], i["datatype"], i["shape"], i["parameters"]["binary_data_size"]) for i in header["inputs"]] == \
    [("img", "UINT8", [1, 3, 4], 12), ("action_t", "FP32", [1, 2], 8)]
  assert body[header_len:header_len + 12] == bytes(range(12))
  assert np.frombuffer(body[header_len + 12:], dtype=np.float32).tolist() == [0.25, 0.5]


def test_decode_response():
  outputs = np.array([1., 2., 3.], dtype=np.float32)
  header = json.dumps({"model_name": "driving_policy", "parameters": {"sequence_warm": True},
                       "outputs": [{"name": "outputs", "shape": [1, 3], "datatype": "FP32",
                                    "parameters": {"binary_data_size": 12}}]}).encode()
  parsed_header, parsed = decode_infer_response(header + outputs.tobytes(), len(header))
  assert parsed_header["parameters"]["sequence_warm"]
  np.testing.assert_array_equal(parsed, outputs)



def test_usb_target(monkeypatch):
  from openpilot.selfdrive.modeld import gpu_on_lan
  monkeypatch.setattr(gpu_on_lan.GpuOnLan, "_worker", lambda self: None)  # no network, no sudo
  for target, usb, host, port in (("", True, None, DEFAULT_PORT), ("usb", True, None, DEFAULT_PORT),
                                  ("USB:9000", True, None, 9000), ("10.0.0.2:8001", False, "10.0.0.2", 8001)):
    c = gpu_on_lan.GpuOnLan(target)
    assert (c.usb, c.host, c.port) == (usb, host, port), target
