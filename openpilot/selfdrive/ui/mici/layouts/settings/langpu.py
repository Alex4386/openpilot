import threading
import time

from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigParamControl
from openpilot.selfdrive.ui.mici.widgets.dialog import BigInputDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.hardware import usb_network
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import NavScroller


class LanGpuLayoutMici(NavScroller):
  """Run the big driving model on a computer on the network (e.g. a Mac running Bamyanggang)."""

  def __init__(self):
    super().__init__()
    self._usb_peer_checked = 0.

    enabled_toggle = BigParamControl("lan gpu", "GpuOnLanEnabled",
                                     description="Run the big driving model on a computer on the network, e.g. a Mac running Bamyanggang. " +
                                                 "The on-device model is used whenever the server is slow or unreachable. Takes effect on the next drive.")

    def server_callback():
      def on_confirm(text: str):
        text = text.strip()
        if text:
          ui_state.params.put("GpuOnLanHost", text)
        else:
          ui_state.params.remove("GpuOnLanHost")
        self._usb_peer_checked = 0.
      gui_app.push_widget(BigInputDialog("server IP[:port], empty for USB...", ui_state.params.get("GpuOnLanHost") or "",
                                         minimum_length=0, confirm_callback=on_confirm))

    self._server_btn = BigButton("server", ui_state.params.get("GpuOnLanHost") or "USB cable",
                                 description="IP address or hostname of the server, optionally with :port (default 8000). " +
                                             "Leave empty to use the computer on the USB network.")
    self._server_btn.set_click_callback(server_callback)

    def usb_network_callback(checked: bool):
      threading.Thread(target=usb_network.up if checked else usb_network.down, daemon=True).start()

    usb_toggle = BigParamControl("usb network", "UsbNetworkEnabled", toggle_callback=usb_network_callback,
                                 description="Connect the computer with a USB-C to USB-C cable to get a local network between it and the device.")

    def usb_address_callback():
      def on_confirm(text: str):
        address = usb_network.parse_address(text)
        ui_state.params.put("UsbNetworkAddress", str(address))
        self._usb_address_btn.set_value(str(address))
        if ui_state.params.get_bool("UsbNetworkEnabled"):
          threading.Thread(target=usb_network.up, daemon=True).start()  # re-apply on the new subnet
      gui_app.push_widget(BigInputDialog("10.47.0.1/24...", str(usb_network.configured_address()),
                                         confirm_callback=on_confirm, text_validator=lambda t: usb_network.parse_address(t) is not None))

    self._usb_address_btn = BigButton("usb network\naddress", str(usb_network.configured_address()),
                                      description="This device's address on the USB network, with the subnet size (CIDR). " +
                                                  "The computer gets an address from the same subnet.")
    self._usb_address_btn.set_click_callback(usb_address_callback)
    self._usb_address_btn.set_enabled(lambda: ui_state.params.get_bool("UsbNetworkEnabled"))

    self._scroller.add_widgets([
      enabled_toggle,
      self._server_btn,
      usb_toggle,
      self._usb_address_btn,
    ])

  def _update_state(self):
    super()._update_state()
    # show which computer lan gpu will use when it rides the USB cable
    now = time.monotonic()
    if now - self._usb_peer_checked > 2.:
      self._usb_peer_checked = now
      if host := ui_state.params.get("GpuOnLanHost"):
        self._server_btn.set_value(host)
      else:
        peer = usb_network.peer_address() if ui_state.params.get_bool("UsbNetworkEnabled") else None
        self._server_btn.set_value(f"USB · {peer}" if peer else "USB cable")
