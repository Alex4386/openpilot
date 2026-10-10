#!/usr/bin/env python3
"""
USB network: an Ethernet link over a USB-C to USB-C cable to a laptop, with the device as the
USB gadget (CDC-NCM). macOS and Linux drive NCM natively, nothing to install on the laptop.

AGNOS's gadget already carries an NCM function (usb0, down by default); then we only configure
it. On builds without one we add our own (ncm.usbnet) next to ADB, which briefly drops ADB
over USB, and only that one is ever removed again.

NetworkManager shares the link like it does the Wi-Fi hotspot (DHCP via dnsmasq) on the subnet
from UsbNetworkAddress (default 10.47.0.1/24, device address in CIDR form), configured over
D-Bus so polkit authorizes it, the same path the settings UI uses for tethering. Only the
configfs gadget edit needs root, and goes through the usual sudo helpers.

Turned on from Settings -> Device -> USB network (UsbNetworkEnabled), re-applied at boot by
hardwared.

  python -m openpilot.system.hardware.usb_network [up|down]
"""
import glob
import ipaddress
import os
import subprocess
import sys
import uuid

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import sudo_read, sudo_write

GADGET_ROOT = "/sys/kernel/config/usb_gadget"
OUR_FUNCTION = "ncm.usbnet"  # only added when the gadget has no NCM of its own
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


def _gadget() -> tuple[str, str] | None:
  """(gadget dir, config dir) of the configfs gadget AGNOS set up for ADB."""
  for gadget in sorted(glob.glob(f"{GADGET_ROOT}/*")):
    configs = sorted(glob.glob(f"{gadget}/configs/*"))
    if os.path.exists(f"{gadget}/UDC") and configs:
      return gadget, configs[0]
  return None


def _udc(gadget: str) -> str:
  return sudo_read(f"{gadget}/UDC") or next(iter(sorted(os.listdir("/sys/class/udc"))), "")


def _rebind(gadget: str, edit) -> None:
  """Functions can only be added/removed on an unbound gadget; ADB over USB blips meanwhile."""
  udc = _udc(gadget)
  sudo_write("\n", f"{gadget}/UDC")  # an empty write never reaches configfs
  try:
    edit()
  finally:
    sudo_write(udc, f"{gadget}/UDC")


def _ncm_functions(config: str) -> list[str]:
  """NCM function instances linked into the active configuration, AGNOS's first."""
  linked = [os.path.basename(p) for p in glob.glob(f"{config}/ncm.*")]
  return sorted(linked, key=lambda name: name == OUR_FUNCTION)


def interface() -> str | None:
  """Network interface of the gadget's NCM function (usually usb0), if there is one."""
  found = _gadget()
  if found is None:
    # not a configfs gadget; a statically configured one still shows up as usb0
    return "usb0" if os.path.exists("/sys/class/net/usb0") else None
  gadget, config = found
  for function in _ncm_functions(config):
    if ifname := sudo_read(f"{gadget}/functions/{function}/ifname"):
      return ifname
  return None


def _ensure_ncm() -> str | None:
  if (ifname := interface()) is not None:
    return ifname  # the gadget already has NCM: nothing to change, ADB stays up

  found = _gadget()
  if found is None:
    cloudlog.warning("usb network: no USB gadget in configfs")
    return None
  gadget, config = found
  function = f"{gadget}/functions/{OUR_FUNCTION}"

  def add():
    if not os.path.exists(function):
      subprocess.run(["sudo", "mkdir", function], check=True)  # needs CONFIG_USB_CONFIGFS_NCM
    # composite device with interface association, so hosts bind NCM next to ADB
    sudo_write("0xEF", f"{gadget}/bDeviceClass")
    sudo_write("0x02", f"{gadget}/bDeviceSubClass")
    sudo_write("0x01", f"{gadget}/bDeviceProtocol")
    subprocess.run(["sudo", "ln", "-s", function, f"{config}/{OUR_FUNCTION}"], check=True)
  _rebind(gadget, add)
  cloudlog.warning(f"usb network: added {OUR_FUNCTION} to {gadget}")
  return interface()


def _remove_ncm() -> None:
  """Removes only the NCM function we added; AGNOS's own stays."""
  found = _gadget()
  if found is None or not os.path.exists(f"{found[1]}/{OUR_FUNCTION}"):
    return
  _rebind(found[0], lambda: subprocess.run(["sudo", "rm", f"{found[1]}/{OUR_FUNCTION}"], check=True))


# NetworkManager over D-Bus (polkit-authorized, like the hotspot)

def _nm_connection(conn, ifname: str | None, address: ipaddress.IPv4Interface | None) -> str | None:
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
    if data and data[0]['address'][1] == str(address.ip) and data[0]['prefix'][1] == address.network.prefixlen:
      return path
    conn.send_and_get_reply(new_method_call(addr, 'Delete'))  # subnet changed in settings
  if address is None or ifname is None:
    return None
  connection = {
    'connection': {
      'type': ('s', '802-3-ethernet'),
      'uuid': ('s', str(uuid.uuid4())),
      'id': ('s', CONNECTION_ID),
      'interface-name': ('s', ifname),
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


def _nm_up(ifname: str) -> None:
  from jeepney import DBusAddress, new_method_call
  from jeepney.io.blocking import open_dbus_connection
  from openpilot.system.ui.lib.networkmanager import NM, NM_PATH, NM_IFACE

  with open_dbus_connection(bus="SYSTEM") as conn:
    path = _nm_connection(conn, ifname, configured_address())
    nm = DBusAddress(NM_PATH, bus_name=NM, interface=NM_IFACE)
    conn.send_and_get_reply(new_method_call(nm, 'ActivateConnection', 'ooo', (path, '/', '/')))


def _nm_delete() -> None:
  from jeepney import DBusAddress, new_method_call
  from jeepney.io.blocking import open_dbus_connection
  from openpilot.system.ui.lib.networkmanager import NM, NM_CONNECTION_IFACE

  with open_dbus_connection(bus="SYSTEM") as conn:
    if (path := _nm_connection(conn, None, None)) is not None:
      conn.send_and_get_reply(new_method_call(DBusAddress(path, bus_name=NM, interface=NM_CONNECTION_IFACE), 'Delete'))


def up() -> str | None:
  """Idempotent. Returns the interface name, or None if this device can't do it."""
  try:
    ifname = _ensure_ncm()
    if ifname is not None:
      _nm_up(ifname)
    return ifname
  except Exception:
    cloudlog.exception("usb network: bring-up failed")
    return None


def down() -> None:
  try:
    _nm_delete()
    _remove_ncm()
  except Exception:
    cloudlog.exception("usb network: teardown failed")


def peer_address(ifname: str | None = None) -> str | None:
  """The laptop on the other end: newest DHCP lease, else any live neighbor on the link."""
  ifname = ifname or interface()
  if ifname is None:
    return None
  address = configured_address()

  def on_link(ip: str) -> bool:
    try:
      return ipaddress.IPv4Address(ip) in address.network and ipaddress.IPv4Address(ip) != address.ip
    except ValueError:
      return False

  for leases in (f"/var/lib/NetworkManager/dnsmasq-{ifname}.leases", "/var/lib/misc/dnsmasq.leases"):
    try:
      with open(leases) as f:
        rows = [r for r in (line.split() for line in f) if len(r) >= 3 and on_link(r[2])]  # expiry mac ip ...
      if rows:
        return max(rows, key=lambda r: int(r[0]))[2]
    except (OSError, ValueError):
      pass

  neigh = subprocess.run(["ip", "-4", "neigh", "show", "dev", ifname], capture_output=True, text=True).stdout
  for parts in (line.split() for line in neigh.splitlines()):
    if parts and parts[-1] in ("REACHABLE", "STALE", "DELAY", "PROBE") and on_link(parts[0]):
      return parts[0]
  return None


if __name__ == "__main__":
  if sys.argv[1:] == ["down"]:
    down()
    print("usb network down")
  else:
    ifname = up()
    print(f"interface {ifname}, device {configured_address()}, peer {peer_address(ifname) if ifname else None}")
