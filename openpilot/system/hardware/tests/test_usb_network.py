import ipaddress

from openpilot.system.hardware import usb_network


def test_parse_address():
  assert str(usb_network.parse_address(" 10.47.0.1/24 ")) == "10.47.0.1/24"
  assert str(usb_network.parse_address("192.168.7.1/30")) == "192.168.7.1/30"
  for bad in ("10.47.0.0/24", "10.47.0.255/24", "8.8.8.8/24", "10.0.0.1/31", "10.0.0.1", "nope"):
    assert usb_network.parse_address(bad) is None, bad


def test_peer_from_leases(tmp_path, monkeypatch):
  leases = tmp_path / "dnsmasq-eth0.leases"
  leases.write_text("100 aa:bb 10.47.0.20 mac *\n200 cc:dd 10.47.0.30 mac *\n300 ee:ff 192.168.43.5 phone *\n")
  monkeypatch.setattr(usb_network, "configured_address", lambda: ipaddress.IPv4Interface("10.47.0.1/24"))
  monkeypatch.setattr(usb_network.glob, "glob", lambda _: [str(leases)])
  assert usb_network.peer_address() == "10.47.0.30"  # newest on our subnet, not the hotspot's


def test_dnsmasq_hands_out_no_gateway():
  args = usb_network.dnsmasq_args(ipaddress.IPv4Interface("10.47.0.1/24"))
  assert "--dhcp-range=10.47.0.2,10.47.0.254,255.255.255.0,12h" in args
  assert "--dhcp-option=3" in args and "--dhcp-option=6" in args  # no router, no DNS server
  assert "--port=0" in args
  assert "--dhcp-range=10.47.0.1,10.47.0.253,255.255.255.0,12h" in usb_network.dnsmasq_args(ipaddress.IPv4Interface("10.47.0.254/24"))
