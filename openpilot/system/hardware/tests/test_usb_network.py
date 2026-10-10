import os

import pytest

from openpilot.system.hardware import usb_network


def make_gadget(root, functions):
  gadget = root / "g1"
  config = gadget / "configs" / "b.1"
  config.mkdir(parents=True)
  (gadget / "UDC").write_text("a600000.dwc3\n")
  for name, ifname in functions.items():
    (gadget / "functions" / name).mkdir(parents=True)
    if ifname:
      (gadget / "functions" / name / "ifname").write_text(ifname + "\n")
    os.symlink(gadget / "functions" / name, config / name)
  return gadget


@pytest.fixture
def configfs(tmp_path, monkeypatch):
  monkeypatch.setattr(usb_network, "GADGET_ROOT", str(tmp_path))
  monkeypatch.setattr(usb_network, "sudo_read", lambda p: open(p).read().strip() if os.path.exists(p) else "")
  return tmp_path


def test_parse_address():
  assert str(usb_network.parse_address(" 10.47.0.1/24 ")) == "10.47.0.1/24"
  assert str(usb_network.parse_address("192.168.7.1/30")) == "192.168.7.1/30"
  for bad in ("10.47.0.0/24", "10.47.0.255/24", "8.8.8.8/24", "10.0.0.1/31", "10.0.0.1", "nope"):
    assert usb_network.parse_address(bad) is None, bad


def test_uses_existing_ncm_without_touching_gadget(configfs, monkeypatch):
  make_gadget(configfs, {"ffs.adb": None, "ncm.usb0": "usb0"})
  monkeypatch.setattr(usb_network, "_rebind", lambda *a: pytest.fail("must not rebind when NCM exists"))
  assert usb_network._ensure_ncm() == "usb0"


def test_prefers_agnos_ncm_over_ours(configfs):
  make_gadget(configfs, {usb_network.OUR_FUNCTION: "usb1", "ncm.usb0": "usb0"})
  assert usb_network.interface() == "usb0"


def test_remove_leaves_agnos_ncm(configfs, monkeypatch):
  make_gadget(configfs, {"ncm.usb0": "usb0"})
  monkeypatch.setattr(usb_network, "_rebind", lambda *a: pytest.fail("nothing of ours to remove"))
  usb_network._remove_ncm()
