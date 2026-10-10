"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import threading
import time

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.hardware import usb_network
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.input_dialog import InputDialogSP
from openpilot.system.ui.sunnypilot.widgets.list_view import button_item_sp, toggle_item_sp
from openpilot.system.ui.widgets import DialogResult, Widget
from openpilot.system.ui.widgets.confirm_dialog import alert_dialog
from openpilot.system.ui.widgets.scroller_tici import Scroller


class LanGpuLayout(Widget):
  """Run the big driving model on a computer on the network (e.g. a Mac running Bamyanggang)."""

  def __init__(self):
    super().__init__()
    self._usb_peer = ""
    self._usb_peer_checked = 0.
    self._scroller = Scroller(self._initialize_items(), line_separator=True, spacing=0)

  def _initialize_items(self):
    self._enabled = toggle_item_sp(
      title=lambda: tr("lan gpu"),
      description=lambda: tr("Run the big driving model on a computer on the network, e.g. a Mac running Bamyanggang. " +
                             "The on-device model keeps running and is used whenever the server is slow or unreachable. " +
                             "Takes effect on the next drive."),
      param="GpuOnLanEnabled",
    )

    self._server = button_item_sp(
      title=lambda: tr("Server"),
      button_text=lambda: tr("EDIT"),
      description=lambda: tr("IP address or hostname of the server, optionally with :port (default 8000). " +
                             "Leave empty to use the computer on the USB network."),
      callback=self._edit_server,
    )
    self._server.action_item.set_value(self._server_label)

    self._usb_network = toggle_item_sp(
      title=lambda: tr("USB Network"),
      description=lambda: tr("Connect the computer with a USB-C to USB-C cable to get a local network between it and the device."),
      param="UsbNetworkEnabled",
      callback=lambda on: threading.Thread(target=usb_network.up if on else usb_network.down, daemon=True).start(),
    )

    self._usb_address = button_item_sp(
      title=lambda: tr("USB Network Address"),
      button_text=lambda: tr("EDIT"),
      description=lambda: tr("This device's address on the USB network, with the subnet size (CIDR). " +
                             "The computer gets an address from the same subnet."),
      callback=self._edit_usb_address,
    )
    self._usb_address.action_item.set_value(lambda: str(usb_network.configured_address()))

    return [self._enabled, self._server, self._usb_network, self._usb_address]

  def _server_label(self) -> str:
    if host := ui_state.params.get("GpuOnLanHost"):
      return host
    return f"USB · {self._usb_peer}" if self._usb_peer else tr("USB cable")

  def _update_state(self):
    super()._update_state()
    now = time.monotonic()
    if now - self._usb_peer_checked > 2.:
      self._usb_peer_checked = now
      self._usb_peer = (usb_network.peer_address() or "") if ui_state.params.get_bool("UsbNetworkEnabled") else ""
    self._usb_address.action_item.set_enabled(ui_state.params.get_bool("UsbNetworkEnabled"))

  @staticmethod
  def _edit_server():
    InputDialogSP(tr("Server"), tr("e.g. 192.168.43.100:8000, or empty for the USB network"),
                  current_text=ui_state.params.get("GpuOnLanHost") or "", param="GpuOnLanHost").show()

  @staticmethod
  def _edit_usb_address():
    def on_result(result: DialogResult, text: str):
      if result != DialogResult.CONFIRM:
        return
      if (address := usb_network.parse_address(text)) is None:
        gui_app.push_widget(alert_dialog(tr("Enter a private IPv4 address with a /8 to /30 subnet, e.g. 10.47.0.1/24")))
        return
      ui_state.params.put("UsbNetworkAddress", str(address))
      if ui_state.params.get_bool("UsbNetworkEnabled"):
        threading.Thread(target=usb_network.up, daemon=True).start()  # re-apply on the new subnet

    InputDialogSP(tr("USB Network Address"), tr("e.g. 10.47.0.1/24"), current_text=str(usb_network.configured_address()),
                  callback=on_result).show()

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()
