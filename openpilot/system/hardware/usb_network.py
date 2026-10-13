#!/usr/bin/env python3
"""
USB network: a wired link to a laptop through a USB Ethernet adapter on the device (RTL8153,
AX88179, or any CDC-Ether/NCM one; AGNOS has the drivers built in) and an Ethernet cable.

NetworkManager shares the link like it does the Wi-Fi hotspot (DHCP via dnsmasq) on the subnet
from UsbNetworkAddress (default 10.47.0.1/24, device address in CIDR form), configured over
D-Bus so polkit authorizes it, the same path the settings UI uses for tethering. The connection
isn't tied to an interface name, so an adapter plugged in at any time gets it; the device has
no other wired Ethernet.

Turned on from Settings -> lan gpu -> usb network (UsbNetworkEnabled), re-applied at boot by
hardwared.

  python -m openpilot.system.hardware.usb_network [up|down]
"""
import glob
import ipaddress
import subprocess
import sys
import uuid

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

CONNECTION_ID = "USB network"
DEFAULT_ADDRESS = "10.47.0.1/24"


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


# NetworkManager over D-Bus (polkit-authorized, like the hotspot)

def _nm_connection(conn, address: ipaddress.IPv4Interface | None) -> str | None:
  """Finds our connection; with an address, (re)creates it so it matches."""
  from jeepney import DBusAddress, new_method_call
  from openpilot.system.ui.lib.networkmanager import NM, NM_SETTINGS_PATH, NM_SETTINGS_IFACE, NM_CONNECTION_IFACE

  settings = DBusAddress(NM_SETTINGS_PATH, bus_name=NM, interface=NM_SETTINGS_IFACE)
  for path in conn.send_and_get_reply(new_method_call(settings, 'ListConnections')).body[0]:
    addr = DBusAddress(path, bus_name=NM, interface=NM_CONNECTION_IFACE)
    current = conn.send_and_get_reply(new_method_call(addr, 'GetSettings')).body[0]
    if current['connection']['id'][1] != CONNECTION_ID:
      continue
    if address is None:
      return path
    data = current.get('ipv4', {}).get('address-data', ('', []))[1]
    if 'interface-name' not in current['connection'] and data and \
       data[0]['address'][1] == str(address.ip) and data[0]['prefix'][1] == address.network.prefixlen:
      return path
    conn.send_and_get_reply(new_method_call(addr, 'Delete'))  # subnet changed in settings, or an old gadget one
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
      'method': ('s', 'shared'),  # dnsmasq hands the laptop an address
      'address-data': ('aa{sv}', [[('address', ('s', str(address.ip))), ('prefix', ('u', address.network.prefixlen))]]),
      'never-default': ('b', True),
    },
    'ipv6': {'method': ('s', 'ignore')},
  }
  return conn.send_and_get_reply(new_method_call(settings, 'AddConnection', 'a{sa{sv}}', (connection,))).body[0]


def up() -> bool:
  """Idempotent. NetworkManager brings the link up whenever an adapter is plugged in."""
  from jeepney.io.blocking import open_dbus_connection
  try:
    with open_dbus_connection(bus="SYSTEM") as conn:
      return _nm_connection(conn, configured_address()) is not None
  except Exception:
    cloudlog.exception("usb network: bring-up failed")
    return False


def down() -> None:
  from jeepney import DBusAddress, new_method_call
  from jeepney.io.blocking import open_dbus_connection
  from openpilot.system.ui.lib.networkmanager import NM, NM_CONNECTION_IFACE
  try:
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
  for leases in glob.glob("/var/lib/NetworkManager/dnsmasq-*.leases") + ["/var/lib/misc/dnsmasq.leases"]:
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
