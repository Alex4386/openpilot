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
                                                 "Works like a chestnut: if the server fails, the on-device model drives until the next drive. " +
                                                 "Takes effect on the next drive.")

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

    self._server_btn = BigButton("server", ui_state.params.get("GpuOnLanHost") or "USB Ethernet",
                                 description="IP address or hostname of the server, optionally with :port (default 8000). " +
                                             "Leave empty to use the computer on the USB network.")
    self._server_btn.set_click_callback(server_callback)

    def usb_network_callback(checked: bool):
      threading.Thread(target=usb_network.up if checked else usb_network.down, daemon=True).start()

    usb_toggle = BigParamControl("usb network", "UsbNetworkEnabled", toggle_callback=usb_network_callback,
                                 description="Plug a USB Ethernet adapter into the device and connect it to the computer with an Ethernet cable " +
                                             "to get a wired network between them.")

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

    def usb_gateway_callback(checked: bool):
      if ui_state.params.get_bool("UsbNetworkEnabled"):
        threading.Thread(target=usb_network.up, daemon=True).start()

    usb_gateway_toggle = BigParamControl("share internet", "UsbNetworkGateway", toggle_callback=usb_gateway_callback,
                                         description="Share this device's internet (Wi-Fi or LTE) with the computer on the USB network: " +
                                                     "it becomes the computer's router and DNS, with NAT. Uses LTE data when on LTE. " +
                                                     "Off: the computer keeps using its own internet connection.")
    usb_gateway_toggle.set_enabled(lambda: ui_state.params.get_bool("UsbNetworkEnabled"))

    self._scroller.add_widgets([
      enabled_toggle,
      self._server_btn,
      usb_toggle,
      self._usb_address_btn,
      usb_gateway_toggle,
    ])

  def _update_state(self):
    super()._update_state()
    # show which computer lan gpu will use over the USB Ethernet adapter
    now = time.monotonic()
    if now - self._usb_peer_checked > 2.:
      self._usb_peer_checked = now
      if host := ui_state.params.get("GpuOnLanHost"):
        self._server_btn.set_value(host)
      else:
        peer = usb_network.peer_address() if ui_state.params.get_bool("UsbNetworkEnabled") else None
        self._server_btn.set_value(f"USB · {peer}" if peer else "USB Ethernet")
