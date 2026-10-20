#!/usr/bin/env python3
"""
USB network: a wired link to a laptop through a USB Ethernet adapter on the device (RTL8153,
AX88179, or any CDC-Ether/NCM one; AGNOS has the drivers built in) and an Ethernet cable.

The device takes the address from UsbNetworkAddress (default 10.47.0.1/24, CIDR form) through a
NetworkManager connection, configured over D-Bus so polkit authorizes it, the same path the
settings UI uses for tethering. The connection isn't tied to an interface name, so an adapter
plugged in at any time gets it; the device has no other wired Ethernet.

The laptop gets its address over DHCP from our own dnsmasq. By default it hands out no router
and no DNS server, so the laptop keeps its own internet connection. With UsbNetworkGateway the
device becomes the laptop's router and DNS (dnsmasq forwards to the device's resolvers) and NATs
its traffic out of whichever uplink the device uses (Wi-Fi, or LTE on ppp0/rmnet, whose address
changes and whose MTU is below the cable's 1500, so forwarded TCP gets its MSS clamped).

Turned on from Settings -> lan gpu -> usb network (UsbNetworkEnabled), re-applied at boot by
hardwared.

  python -m openpilot.system.hardware.usb_network [up|down]
"""
import glob
import ipaddress
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

from openpilot.common.hardware.usb import PRIMARY_USB_CONTROLLER, controller
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

CONNECTION_ID = "USB network"
DEFAULT_ADDRESS = "10.47.0.1/24"
DNSMASQ_PID = "/run/usb-network-dnsmasq.pid"
DNSMASQ_LEASES = "/run/usb-network-dnsmasq.leases"
RULE_TAG = "usb-network"  # iptables comment on every rule we add, so teardown removes exactly ours
NET_PATH = Path("/sys/class/net")


def adapter_present() -> bool:
  """A USB Ethernet adapter on the USB-C port. The port is the chestnut's, so its cable isn't one then."""
  try:
    return any((ctrl := controller(net / "device")) is not None and ctrl.name == PRIMARY_USB_CONTROLLER
               for net in NET_PATH.iterdir() if (net / "device").exists())
  except OSError:
    return False


def parse_address(text: str) -> ipaddress.IPv4Interface | None:
  """Device address on the USB link in CIDR form, e.g. 10.47.0.1/24. Private IPv4, /8 to /30."""
  try:
    iface = ipaddress.IPv4Interface(text.strip())
  except ValueError:
    return None
  if not iface.ip.is_private or not 8 <= iface.network.prefixlen <= 30:
    return None
  if iface.ip in (iface.network.network_address, iface.network.broadcast_address):
    return None
  return iface


def configured_address() -> ipaddress.IPv4Interface:
  return parse_address(Params().get("UsbNetworkAddress") or "") or ipaddress.IPv4Interface(DEFAULT_ADDRESS)


def dnsmasq_args(address: ipaddress.IPv4Interface, gateway: bool = False) -> list[str]:
  """DHCP on the USB Ethernet adapter. Without gateway: no DNS server, and no router or DNS handed out.
  With gateway: dnsmasq hands itself out as router and DNS, forwarding to the device's resolvers."""
  net = address.network
  if address.ip + 1 < net.broadcast_address:
    first, last = address.ip + 1, net.broadcast_address - 1
  else:
    first, last = net.network_address + 1, address.ip - 1
  args = ["dnsmasq", "--conf-file=/dev/null", "--bind-dynamic",
          "--interface=eth*", "--interface=enx*", "--except-interface=lo",  # USB Ethernet adapters only
          f"--dhcp-range={first},{last},{net.netmask},12h",
          f"--dhcp-leasefile={DNSMASQ_LEASES}", f"--pid-file={DNSMASQ_PID}"]
  if gateway:
    return args
  return args + ["--port=0", "--dhcp-option=3", "--dhcp-option=6"]  # no data: send no router and no DNS server


def nat_rules(address: ipaddress.IPv4Interface) -> list[tuple[str, list[str]]]:
  """(table, rule) for sharing the device's internet with the laptop, each tagged with RULE_TAG."""
  net = str(address.network)
  tag = ["-m", "comment", "--comment", RULE_TAG]
  return [
    # out of any uplink (wlan0, ppp0, rmnet...); MASQUERADE follows LTE's changing address
    ("nat", ["POSTROUTING", "-s", net, "!", "-d", net, *tag, "-j", "MASQUERADE"]),
    ("filter", ["FORWARD", "-s", net, *tag, "-j", "ACCEPT"]),
    ("filter", ["FORWARD", "-d", net, "-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED", *tag, "-j", "ACCEPT"]),
    # the uplink's MTU (ppp0/LTE) is below the cable's 1500: clamp forwarded TCP so connections don't stall
    ("mangle", ["FORWARD", "-p", "tcp", "--tcp-flags", "SYN,RST", "SYN", "-s", net, *tag, "-j", "TCPMSS", "--clamp-mss-to-pmtu"]),
    ("mangle", ["FORWARD", "-p", "tcp", "--tcp-flags", "SYN,RST", "SYN", "-d", net, *tag, "-j", "TCPMSS", "--clamp-mss-to-pmtu"]),
  ]


def _remove_nat() -> None:
  for table in ("nat", "filter", "mangle"):
    rules = subprocess.run(["sudo", "iptables", "-t", table, "-S"], capture_output=True, text=True).stdout
    for line in rules.splitlines():
      if line.startswith("-A ") and f"--comment {RULE_TAG}" in line:
        subprocess.run(["sudo", "iptables", "-t", table, "-D", *shlex.split(line)[1:]], capture_output=True)


def _add_nat(address: ipaddress.IPv4Interface) -> None:
  subprocess.run(["sudo", "sysctl", "-q", "-w", "net.ipv4.ip_forward=1"], check=True, capture_output=True)
  for table, rule in nat_rules(address):
    # inserted first in the chain, ahead of any DROP rule
    subprocess.run(["sudo", "iptables", "-t", table, "-I", *rule], check=True, capture_output=True)


def _stop_dnsmasq() -> None:
  subprocess.run(["sudo", "pkill", "-F", DNSMASQ_PID], capture_output=True)
  subprocess.run(["sudo", "rm", "-f", DNSMASQ_LEASES], capture_output=True)  # so peer_address doesn't find a stale laptop


# NetworkManager over D-Bus (polkit-authorized, like the hotspot)

def _nm_connection(conn, address: ipaddress.IPv4Interface | None) -> str | None:
  """Finds our connection; with an address, (re)creates it so it matches."""
  from jeepney import DBusAddress, new_method_call
  from openpilot.system.ui.lib.networkmanager import NM, NM_SETTINGS_PATH, NM_SETTINGS_IFACE, NM_CONNECTION_IFACE

  settings = DBusAddress(NM_SETTINGS_PATH, bus_name=NM, interface=NM_SETTINGS_IFACE)
  method = 'manual'  # DHCP and NAT are ours (dnsmasq_args, nat_rules), not NetworkManager's shared mode
  for path in conn.send_and_get_reply(new_method_call(settings, 'ListConnections')).body[0]:
    addr = DBusAddress(path, bus_name=NM, interface=NM_CONNECTION_IFACE)
    current = conn.send_and_get_reply(new_method_call(addr, 'GetSettings')).body[0]
    if current['connection']['id'][1] != CONNECTION_ID:
      continue
    if address is None:
      return path
    ipv4 = current.get('ipv4', {})
    data = ipv4.get('address-data', ('', []))[1]
    if 'interface-name' not in current['connection'] and data and ipv4.get('method', ('', ''))[1] == method and \
       data[0]['address'][1] == str(address.ip) and data[0]['prefix'][1] == address.network.prefixlen:
      return path
    conn.send_and_get_reply(new_method_call(addr, 'Delete'))  # subnet changed in settings, or an old shared/gadget one
  if address is None:
    return None
  connection = {
    'connection': {
      'type': ('s', '802-3-ethernet'),
      'uuid': ('s', str(uuid.uuid4())),
      'id': ('s', CONNECTION_ID),
      'autoconnect': ('b', True),
    },
    'ipv4': {
      'method': ('s', method),
      'address-data': ('aa{sv}', [[('address', ('s', str(address.ip))), ('prefix', ('u', address.network.prefixlen))]]),
      'never-default': ('b', True),
    },
    'ipv6': {'method': ('s', 'ignore')},
  }
  return conn.send_and_get_reply(new_method_call(settings, 'AddConnection', 'a{sa{sv}}', (connection,))).body[0]


def up() -> bool:
  """Idempotent. NetworkManager brings the link up whenever an adapter is plugged in."""
  from jeepney.io.blocking import open_dbus_connection
  address, gateway = configured_address(), Params().get_bool("UsbNetworkGateway")
  try:
    _stop_dnsmasq()
    _remove_nat()
    with open_dbus_connection(bus="SYSTEM") as conn:
      if _nm_connection(conn, address) is None:
        return False
    subprocess.run(["sudo", *dnsmasq_args(address, gateway)], check=True, capture_output=True)  # forks into the background
    if gateway:
      _add_nat(address)
    return True
  except Exception:
    cloudlog.exception("usb network: bring-up failed")
    return False


def down() -> None:
  from jeepney import DBusAddress, new_method_call
  from jeepney.io.blocking import open_dbus_connection
  from openpilot.system.ui.lib.networkmanager import NM, NM_CONNECTION_IFACE
  try:
    _stop_dnsmasq()
    _remove_nat()
    with open_dbus_connection(bus="SYSTEM") as conn:
      if (path := _nm_connection(conn, None)) is not None:
        conn.send_and_get_reply(new_method_call(DBusAddress(path, bus_name=NM, interface=NM_CONNECTION_IFACE), 'Delete'))
  except Exception:
    cloudlog.exception("usb network: teardown failed")


def peer_address() -> str | None:
  """The laptop on the other end: newest DHCP lease on our subnet, else a live neighbor on it."""
  address = configured_address()

  def on_link(ip: str) -> bool:
    try:
      return ipaddress.IPv4Address(ip) in address.network and ipaddress.IPv4Address(ip) != address.ip
    except ValueError:
      return False

  rows = []
  for leases in [DNSMASQ_LEASES] + glob.glob("/var/lib/NetworkManager/dnsmasq-*.leases") + ["/var/lib/misc/dnsmasq.leases"]:
    try:
      with open(leases) as f:
        rows += [r for r in (line.split() for line in f) if len(r) >= 3 and on_link(r[2])]  # expiry mac ip ...
    except OSError:
      pass
  if rows:
    return max(rows, key=lambda r: int(r[0]))[2]

  neigh = subprocess.run(["ip", "-4", "neigh", "show"], capture_output=True, text=True).stdout
  for parts in (line.split() for line in neigh.splitlines()):
    if parts and parts[-1] in ("REACHABLE", "STALE", "DELAY", "PROBE") and on_link(parts[0]):
      return parts[0]
  return None


if __name__ == "__main__":
  if sys.argv[1:] == ["down"]:
    down()
    print("usb network down")
  else:
    print(f"connection {'set' if up() else 'failed'}, device {configured_address()}, peer {peer_address()}")
