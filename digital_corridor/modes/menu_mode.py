# =============================================================================
# modes/menu_mode.py — Главное меню «Цифровой коридор»
# =============================================================================
"""
MenuMode отображает три большие кнопки поверх живого видео с камеры.
Выбор режима — удержание указательного пальца над кнопкой 2 секунды.
Кнопки расположены вертикально по центру экрана с красивым overlay.
"""

import cv2
import numpy as np
import math
import time
from typing import Optional

from modes.base_mode import BaseMode
from modules.hand_tracker import HandData
from modules.gesture_detector import Gesture
from modules.ui import UIRenderer, HoverButton, draw_rounded_rect, put_text_centered
from config import (
    MODE_DRAWING, MODE_SLIDES, MODE_GAMES,
    COLOR_WHITE, COLOR_ACCENT, COLOR_ACCENT2, COLOR_DARK,
    BUTTON_WIDTH, BUTTON_HEIGHT, HUD_HEIGHT, FONT_MAIN, FONT_BOLD
)


class MenuMode(BaseMode):
    """Главное меню с тремя виртуальными кнопками."""

    def __init__(self, ui: UIRenderer):
        super().__init__(ui)
        self._buttons: list[HoverButton] = []
        self._button_modes: list[str] = []
        self._anim_time = 0.0           # Для анимации появления
        self._title_pulse = 0.0         # Пульсация заголовка

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self):
        """Инициализируем кнопки при входе в меню."""
        self._anim_time  = time.time()
        self._title_pulse = 0.0
        self._buttons    = []
        self._button_modes = []
        # Кнопки создаются в _build_buttons() — там нужны размеры кадра

    def on_exit(self):
        """Сбрасываем все кнопки."""
        for btn in self._buttons:
            btn.reset()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        hands: list[HandData],
        gesture: str,
        primary_hand: Optional[HandData],
    ) -> Optional[str]:
        """Рисует меню и проверяет нажатия кнопок."""
        fh, fw = frame.shape[:2]

        # Создаём кнопки при первом кадре (когда знаем размеры)
        if not self._buttons:
            self._create_buttons(fw, fh)

        # --- Фоновый overlay меню ---
        self._draw_background(frame, fw, fh)

        # --- Заголовок ---
        self._draw_title(frame, fw, fh)

        # --- Кнопки ---
        finger_pos = None
        if primary_hand and gesture == Gesture.POINTING:
            ix, iy = primary_hand.index_tip_px
            finger_pos = (ix, iy)
            self._ui.draw_finger_cursor(frame, ix, iy)

        for btn, mode in zip(self._buttons, self._button_modes):
            btn.draw(frame, finger_pos)

            # Проверяем нажатие (только при жесте POINTING)
            if finger_pos and gesture == Gesture.POINTING:
                if btn.update(finger_pos[0], finger_pos[1]):
                    return mode

        # --- HUD ---
        n_hands = len(hands)
        self._ui.draw_hud(frame, "Цифровой коридор", 0, "", n_hands)

        # --- Подсказка внизу ---
        self._draw_hint(frame, fw, fh)

        return None

    # ------------------------------------------------------------------
    # Построение кнопок
    # ------------------------------------------------------------------

    def _create_buttons(self, fw: int, fh: int):
        """Создаёт кнопки, центрированные по вертикали."""
        items = [
            (MODE_DRAWING, "Рисование",  "[ ]"),
            (MODE_SLIDES,  "Слайды",     "[>]"),
            (MODE_GAMES,   "Игры",       "[G]"),
        ]

        n = len(items)
        gap = 30        # Отступ между кнопками
        total_h = n * BUTTON_HEIGHT + (n - 1) * gap
        start_y = (fh - total_h) // 2 + HUD_HEIGHT // 2
        btn_x   = (fw - BUTTON_WIDTH) // 2

        accent_colors = [
            (0, 200, 255),   # голубой
            (180, 0, 255),   # фиолетовый
            (0, 220, 80),    # зелёный
        ]

        for i, (mode, label, icon) in enumerate(items):
            y = start_y + i * (BUTTON_HEIGHT + gap)
            btn = HoverButton(
                x=btn_x, y=y,
                width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                label=label, icon=icon,
                color_accent=accent_colors[i],
            )
            self._buttons.append(btn)
            self._button_modes.append(mode)

    # ------------------------------------------------------------------
    # Внешний вид
    # ------------------------------------------------------------------

    def _draw_background(self, frame: np.ndarray, fw: int, fh: int):
        """Плавный тёмный overlay поверх камеры."""
        overlay = frame.copy()
        # Центральный прямоугольник меню
        margin_x = max(0, (fw - BUTTON_WIDTH) // 2 - 60)
        cv2.rectangle(overlay, (margin_x, HUD_HEIGHT),
                      (fw - margin_x, fh - 40), (12, 12, 22), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Декоративные линии
        cv2.line(frame, (margin_x, HUD_HEIGHT), (margin_x, fh - 40),
                 COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.line(frame, (fw - margin_x, HUD_HEIGHT), (fw - margin_x, fh - 40),
                 COLOR_ACCENT, 1, cv2.LINE_AA)

    def _draw_title(self, frame: np.ndarray, fw: int, fh: int):
        """Заголовок с пульсирующим акцентом."""
        self._title_pulse = (self._title_pulse + 0.03) % (2 * math.pi)
        pulse = 0.85 + 0.15 * math.sin(self._title_pulse)

        title   = "CIFROVOY KORIDOR"
        sub     = "Выберите режим указательным пальцем"

        # Тень заголовка
        put_text_centered(frame, title, fw // 2 + 2, HUD_HEIGHT + 50,
                          FONT_BOLD, 1.1 * pulse, (0, 0, 0), 3)
        put_text_centered(frame, title, fw // 2, HUD_HEIGHT + 50,
                          FONT_BOLD, 1.1 * pulse, COLOR_ACCENT, 2)

        put_text_centered(frame, sub, fw // 2, HUD_HEIGHT + 82,
                          FONT_MAIN, 0.55, (160, 160, 160), 1)

    def _draw_hint(self, frame: np.ndarray, fw: int, fh: int):
        """Подсказка внизу: держите палец над кнопкой."""
        hint = "Удержите указательный палец над кнопкой 2 секунды"
        put_text_centered(frame, hint, fw // 2, fh - 18,
                          FONT_MAIN, 0.5, (100, 100, 100), 1)
