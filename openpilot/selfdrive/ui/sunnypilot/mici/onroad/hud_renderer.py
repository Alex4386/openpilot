"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import pyray as rl

from openpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer, SET_SPEED_PERSISTENCE
from openpilot.selfdrive.ui.sunnypilot.onroad.blind_spot_indicators import BlindSpotIndicators
from openpilot.selfdrive.ui.sunnypilot.ui_state import LanGpuState
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app


class HudRendererSP(HudRenderer):
  def __init__(self):
    super().__init__()
    self.blind_spot_indicators = BlindSpotIndicators()
    self._txt_lan_gpu: rl.Texture = gui_app.texture('icons_mici/lan_gpu.png', 60, 44)
    self._txt_lan_gpu_green: rl.Texture = gui_app.texture('icons_mici/lan_gpu_green.png', 60, 44)
    self._txt_lan_gpu_orange: rl.Texture = gui_app.texture('icons_mici/lan_gpu_orange.png', 75, 44)
    self._sp_engaged: bool = False

  def _update_state(self) -> None:
    super()._update_state()
    self.blind_spot_indicators.update()

    sp_engaged = ui_state.engaged
    if sp_engaged != self._sp_engaged:
      self._chestnut_fade_time = rl.get_time() if sp_engaged else 0
    self._sp_engaged = sp_engaged

  def _render(self, rect: rl.Rectangle) -> None:
    super()._render(rect)
    self.blind_spot_indicators.render(rect)

  def _draw_model_source(self, rect: rl.Rectangle) -> None:
    """The chestnut's GPU icon, with LAN on it, when the big model runs on a computer on the network."""
    state = ui_state.lan_gpu_state
    if state == LanGpuState.DISABLED:
      super()._draw_model_source(rect)
      return
    if ui_state.sm.recv_frame['selfdriveState'] < ui_state.started_frame:
      return

    loading = state == LanGpuState.CONNECTING
    if loading:
      icon = self._txt_lan_gpu
      opacity = 0.35 + 0.65 * (0.5 - 0.5 * math.cos(rl.get_time() * 6.0))
    elif state == LanGpuState.FALLBACK:
      icon, opacity = self._txt_lan_gpu_orange, 1.0
    elif state == LanGpuState.ACTIVE:
      icon, opacity = self._txt_lan_gpu_green, 1.0
    else:
      return

    # same placement and fade as the chestnut icon
    if icon is not self._chestnut_icon:
      self._chestnut_fade_time = rl.get_time()
      self._chestnut_icon = icon
    visible = loading or rl.get_time() - self._chestnut_fade_time < SET_SPEED_PERSISTENCE
    alpha = self._chestnut_alpha_filter.update(visible)
    if alpha < 1e-2:
      return

    pos = rl.Vector2(rect.x + rect.width - 10 - icon.width,
                     rect.y + rect.height - 14 - (self._txt_wheel.height + icon.height) / 2)
    rl.draw_texture_ex(icon, pos, 0.0, 1.0, rl.Color(255, 255, 255, int(255 * opacity * alpha)))

  def _has_blind_spot_detected(self) -> bool:

    return self.blind_spot_indicators.detected
