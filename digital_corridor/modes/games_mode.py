# =============================================================================
# modes/games_mode.py — Встроенные мини-игры (OpenCV, без браузера)
# =============================================================================
"""
GamesMode v3 — Три игры работают ВНУТРИ окна приложения:
  - Chrome Dino (бесконечный раннер)
  - Battle City (танки)
  - Super Mario (платформер)

Управление позицией ладони (зоны):
  Верхняя 35%  → W / прыжок
  Нижняя  35%  → S / присесть
  Левая   35%  → A / влево
  Правая  35%  → D / вправо
  Открытая ладонь → огонь / прыжок
  
Две открытые ладони → возврат в меню
"""

import cv2
import numpy as np
import time
from typing import Optional

from modes.base_mode import BaseMode
from modules.hand_tracker import HandData
from modules.gesture_detector import Gesture
from modules.ui import UIRenderer, HoverButton, draw_rounded_rect, put_text_centered
from config import (
    MODE_MENU, COLOR_WHITE, COLOR_ACCENT, COLOR_GREEN,
    COLOR_ORANGE, COLOR_GRAY, BUTTON_WIDTH, BUTTON_HEIGHT,
    HUD_HEIGHT, FONT_BOLD, FONT_MAIN, FONT_SMALL
)
from games.base_game import VIRT_W, VIRT_H
from games.dino_game   import DinoGame
from games.battle_game import BattleGame
from games.mario_game  import MarioGame


# Зоны управления (доли кадра)
ZONE_TOP    = 0.35
ZONE_BOTTOM = 0.65
ZONE_LEFT   = 0.35
ZONE_RIGHT  = 0.65

ZONE_COLORS = {
    "up":     (0,   220, 80 ),
    "down":   (0,   60,  220),
    "left":   (255, 140, 0  ),
    "right":  (180, 0,   255),
    "center": (60,  60,  60 ),
}

# =============================================================================
GAMES_CONFIG = [
    {
        "label":   "Chrome Dino",
        "icon":    "[D]",
        "hint":    "Ruka VYSHE = pryzhok  |  NIZHE = sit'",
        "factory": DinoGame,
        "accent":  COLOR_ACCENT,
    },
    {
        "label":   "Battle City",
        "icon":    "[B]",
        "hint":    "Zony = WASD  |  Otkr. ladon' = ogon'",
        "factory": BattleGame,
        "accent":  COLOR_ORANGE,
    },
    {
        "label":   "Super Mario",
        "icon":    "[M]",
        "hint":    "Vlevo/Vpravo = dvizheniye  |  VYSHE/Ladon' = pryzhok",
        "factory": MarioGame,
        "accent":  COLOR_GREEN,
    },
]


class GamesMode(BaseMode):
    STATE_SELECT  = "select"
    STATE_PLAYING = "playing"

    def __init__(self, ui: UIRenderer):
        super().__init__(ui)
        self._state       = self.STATE_SELECT
        self._game_idx    = 0
        self._game        = None      # Текущий экземпляр игры
        self._select_btns: list[HoverButton] = []
        self._current_zone = "center"
        self._last_dt_time = 0.0
        self._virtual_canvas = None   # Холст виртуального разрешения

    # ------------------------------------------------------------------
    def on_enter(self):
        self._state    = self.STATE_SELECT
        self._select_btns = []
        self._game = None
        self._current_zone = "center"

    def on_exit(self):
        self._game = None

    # ------------------------------------------------------------------
    def update(self, frame, hands, gesture, primary_hand) -> Optional[str]:
        back = self._check_back_gesture(gesture)
        if back:
            self._game = None
            return back

        if self._state == self.STATE_SELECT:
            return self._update_select(frame, hands, gesture, primary_hand)
        else:
            return self._update_playing(frame, hands, gesture, primary_hand)

    # ------------------------------------------------------------------
    # Select screen
    # ------------------------------------------------------------------

    def _update_select(self, frame, hands, gesture, primary_hand):
        fh, fw = frame.shape[:2]
        if not self._select_btns:
            self._create_buttons(fw, fh)

        # Тёмный overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, HUD_HEIGHT), (fw, fh), (10,10,18), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        put_text_centered(frame, "VYBERI IGRU",
                          fw//2, HUD_HEIGHT + 52, FONT_BOLD, 1.2, COLOR_ACCENT, 2)
        put_text_centered(frame, "Derzhite palec nad knopkoy 2 sekundy",
                          fw//2, HUD_HEIGHT + 82, FONT_MAIN, 0.55, COLOR_GRAY, 1)

        finger_pos = None
        if primary_hand and gesture == Gesture.POINTING:
            ix, iy = primary_hand.index_tip_px
            finger_pos = (ix, iy)
            self._ui.draw_finger_cursor(frame, ix, iy)

        for i, btn in enumerate(self._select_btns):
            btn.draw(frame, finger_pos)
            if finger_pos and gesture == Gesture.POINTING:
                if btn.update(finger_pos[0], finger_pos[1]):
                    self._launch_game(i)

        self._ui.draw_hud(frame, "Igry — vybor", 0, "", len(hands))
        self._ui.draw_back_hint(frame)
        return None

    def _create_buttons(self, fw, fh):
        n = len(GAMES_CONFIG); gap = 28
        sy = (fh - n*BUTTON_HEIGHT - (n-1)*gap)//2 + HUD_HEIGHT//2
        bx = (fw - BUTTON_WIDTH)//2
        for i, g in enumerate(GAMES_CONFIG):
            y = sy + i*(BUTTON_HEIGHT+gap)
            self._select_btns.append(HoverButton(
                x=bx, y=y, label=g["label"], icon=g["icon"],
                color_accent=g["accent"]
            ))

    def _launch_game(self, idx: int):
        self._game_idx  = idx
        self._game      = GAMES_CONFIG[idx]["factory"]()
        self._state     = self.STATE_PLAYING
        self._last_dt_time = time.time()
        print(f"[GamesMode] Запуск: {GAMES_CONFIG[idx]['label']}")

    # ------------------------------------------------------------------
    # Playing screen
    # ------------------------------------------------------------------

    def _update_playing(self, frame, hands, gesture, primary_hand):
        fh, fw = frame.shape[:2]
        cfg = GAMES_CONFIG[self._game_idx]

        # --- Delta time ---
        now = time.time()
        dt  = min(now - self._last_dt_time, 0.1)   # Клипаем чтоб не взорваться при паузе
        self._last_dt_time = now

        # --- Зона руки ---
        zone = "center"
        is_open_palm = False
        if primary_hand:
            px, py  = primary_hand.palm_center_px()
            nx, ny  = px/fw, py/fh
            zone    = self._get_zone(nx, ny)
            is_open_palm = (gesture == Gesture.OPEN_PALM)

        self._current_zone = zone

        # --- Обновляем игру ---
        self._game.update(dt, zone, is_open_palm)

        # --- Рендер игры → виртуальный холст → масштаб в кадр ---
        game_area_y1 = HUD_HEIGHT
        game_area_y2 = fh - 70          # Снизу — полоска подсказки
        game_area_h  = game_area_y2 - game_area_y1
        game_area_w  = fw

        # Виртуальный холст
        if (self._virtual_canvas is None or
                self._virtual_canvas.shape[:2] != (VIRT_H, VIRT_W)):
            self._virtual_canvas = np.zeros((VIRT_H, VIRT_W, 3), dtype=np.uint8)

        self._game.render(self._virtual_canvas)

        # Масштабируем с сохранением пропорций
        scale = min(game_area_w/VIRT_W, game_area_h/VIRT_H)
        rw    = int(VIRT_W * scale)
        rh    = int(VIRT_H * scale)
        rendered = cv2.resize(self._virtual_canvas, (rw, rh), interpolation=cv2.INTER_LINEAR)

        # Центрируем в игровой области
        x_off = (game_area_w - rw)//2
        y_off = game_area_y1 + (game_area_h - rh)//2
        frame[y_off:y_off+rh, x_off:x_off+rw] = rendered

        # --- Мини-зоны (полупрозрачные, края) ---
        self._draw_zone_hints(frame, fw, fh)

        # --- Подсветка активной зоны ---
        if primary_hand:
            px, py = primary_hand.palm_center_px()
            color  = ZONE_COLORS.get(zone, (255,255,255))
            self._ui.draw_finger_cursor(frame, px, py, color, radius=20)

        # --- Нижняя полоска с подсказкой ---
        bottom_strip = frame.copy()
        cv2.rectangle(bottom_strip, (0, fh-70), (fw, fh), (8,8,14), -1)
        cv2.addWeighted(bottom_strip, 0.75, frame, 0.25, 0, frame)
        put_text_centered(frame, cfg["hint"], fw//2, fh-42,
                          FONT_SMALL, 1.05, COLOR_GRAY, 1)
        put_text_centered(frame, "Dve ladoni = MENU", fw//2, fh-18,
                          FONT_SMALL, 0.9, (50,50,50), 1)

        # HUD
        self._ui.draw_hud(frame, f"Igra: {cfg['label']}", 0,
                          zone.upper(), len(hands))
        return None

    # ------------------------------------------------------------------
    # Зоны
    # ------------------------------------------------------------------

    def _get_zone(self, nx, ny) -> str:
        if ny < ZONE_TOP:   return "up"
        if ny > ZONE_BOTTOM: return "down"
        if nx < ZONE_LEFT:   return "left"
        if nx > ZONE_RIGHT:  return "right"
        return "center"

    def _draw_zone_hints(self, frame: np.ndarray, fw: int, fh: int):
        """Рисует узкие цветные полоски на краях для подсказки зон."""
        s  = 8   # Толщина полоски
        top = HUD_HEIGHT
        a   = 0.45

        def bar(x1,y1,x2,y2,color):
            ov = frame.copy()
            cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
            cv2.addWeighted(ov, a, frame, 1-a, 0, frame)

        bar(0,    top,  fw,  top+s, ZONE_COLORS["up"])     # Верх
        bar(0,    fh-80, fw, fh-70, ZONE_COLORS["down"])   # Низ
        bar(0,    top,  s,   fh-70, ZONE_COLORS["left"])   # Лево
        bar(fw-s, top,  fw,  fh-70, ZONE_COLORS["right"])  # Право
