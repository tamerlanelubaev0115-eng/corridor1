# =============================================================================
# modes/drawing_mode.py — Рисование (мульти-рука: до 2 пользователей)
# =============================================================================
"""
DrawingMode v2 — Мультипользовательское рисование:
  - Каждая рука получает свой набор цветов и собственный prev_point
  - Рука 0 → цвета из основной палитры (выбирается наведением)
  - Рука 1 → автоматически второй цвет из другой части палитры
  - Жест «кулак» любой рукой → очистка холста
  - Оба пользователя видят индикатор своей руки (разные цвета курсора)
  - Жест «две ладони» → возврат в меню
"""

import cv2
import numpy as np
import math
import time
from typing import Optional

from modes.base_mode import BaseMode
from modules.hand_tracker import HandData
from modules.gesture_detector import Gesture
from modules.ui import UIRenderer, draw_rounded_rect, put_text_centered
from config import (
    MODE_MENU, COLOR_WHITE, COLOR_ACCENT,
    BRUSH_COLORS, HUD_HEIGHT, FONT_BOLD, FONT_SMALL,
    FIST_FRAMES_THRESHOLD
)

PALETTE_RADIUS  = 26
PALETTE_MARGIN  = 22
PALETTE_GAP     = 10
BRUSH_MIN       = 3
BRUSH_MAX       = 22
BRUSH_NORM_SIZE = 120

# Курсорные цвета для каждой руки (чтобы пользователи различали себя)
HAND_CURSOR_COLORS = [
    COLOR_ACCENT,      # Рука 0 — голубой
    (180, 0, 255),     # Рука 1 — фиолетовый
]

# Стартовый индекс цвета для второй руки
HAND_DEFAULT_COLOR = [0, 4]   # Рука 0 → красный, Рука 1 → синий


class _HandDrawState:
    """Состояние рисования для одной руки."""
    def __init__(self, default_color_idx: int = 0):
        self.prev_point: Optional[tuple] = None
        self.color_idx: int = default_color_idx
        self.ok_frames: int = 0   # Счётчик кадров жеста OK (для очистки)

OK_CLEAR_FRAMES = 10   # ~0.33 сек при 30fps — задержка перед очисткой


class DrawingMode(BaseMode):
    """Режим мультипользовательского рисования (до 2 рук)."""

    def __init__(self, ui: UIRenderer):
        super().__init__(ui)
        self._canvas: Optional[np.ndarray] = None
        self._hand_states: list[_HandDrawState] = []
        self._cleared = False
        self._clear_anim_end = 0.0   # Время окончания анимации очистки

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self):
        self._canvas = None
        self._cleared = False
        self._clear_anim_end = 0.0
        self._hand_states = [
            _HandDrawState(HAND_DEFAULT_COLOR[0]),
            _HandDrawState(HAND_DEFAULT_COLOR[1]),
        ]

    def on_exit(self):
        for s in self._hand_states:
            s.prev_point = None

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
        fh, fw = frame.shape[:2]

        if self._canvas is None or self._canvas.shape[:2] != (fh, fw):
            self._canvas = np.zeros((fh, fw, 3), dtype=np.uint8)

        # Возврат в меню
        back = self._check_back_gesture(gesture)
        if back:
            return back

        # Обрабатываем каждую руку независимо
        any_ok = False
        for i, hand in enumerate(hands[:2]):
            state = self._hand_states[i]

            # --- OK-жест → счётчик очистки ---
            if hand.is_ok_sign():
                state.ok_frames += 1
                state.prev_point = None   # Не рисуем во время OK
                if state.ok_frames >= 1:
                    any_ok = True
            else:
                state.ok_frames = 0

            # --- Кулак → просто поднять перо ---
            if hand.is_fist():
                state.prev_point = None
            elif not hand.is_ok_sign() and hand.is_pointing_or_index_up():
                self._draw_stroke(hand, state, fw, fh)
            elif not hand.is_ok_sign() and not hand.is_fist():
                state.prev_point = None

        # Руки убраны — разрываем линии
        for i in range(len(hands), 2):
            self._hand_states[i].prev_point = None

        # Очистка холста: OK-жест удерживается OK_CLEAR_FRAMES кадров
        ok_max = max(s.ok_frames for s in self._hand_states)
        if ok_max >= OK_CLEAR_FRAMES and not self._cleared:
            self._canvas[:] = 0
            self._cleared = True
            self._clear_anim_end = time.time() + 0.8

        if not any_ok:
            self._cleared = False

        # Рендер холста
        self._render_canvas(frame)

        # Палитры (для каждой руки — своя позиция)
        palette_rects = self._draw_palette(frame, fw, fh, hands)

        # Курсоры и выбор цвета
        for i, hand in enumerate(hands[:2]):
            state = self._hand_states[i]
            ix, iy = hand.index_tip_px
            cursor_color = HAND_CURSOR_COLORS[i % len(HAND_CURSOR_COLORS)]
            self._ui.draw_finger_cursor(frame, ix, iy, cursor_color, radius=14)

            if hand.is_pointing_or_index_up():
                self._check_palette_selection(ix, iy, palette_rects[i], state)

        # HUD
        n = len(hands)
        mode_str = f"Risovanie ({n} ruk)  |  OK-zhest = ochistit'"
        self._ui.draw_hud(frame, mode_str, 0, "", n)
        self._ui.draw_back_hint(frame)

        # Индикатор очистки OK
        ok_max = max(s.ok_frames for s in self._hand_states)
        if ok_max >= 1:
            progress = min(ok_max / OK_CLEAR_FRAMES, 1.0)
            self._draw_clear_indicator(frame, fw, fh, progress)

        # Вспышка после очистки
        if time.time() < self._clear_anim_end:
            alpha = min((self._clear_anim_end - time.time()) / 0.8, 1.0) * 0.3
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, HUD_HEIGHT), (fw, fh), (255, 255, 255), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Значки пользователей (левый верхний угол)
        self._draw_user_badges(frame, hands)

        return None

    # ------------------------------------------------------------------
    # Рисование
    # ------------------------------------------------------------------

    def _draw_stroke(self, hand: HandData, state: _HandDrawState,
                     fw: int, fh: int):
        ix, iy = hand.index_tip_px
        ix = max(0, min(fw - 1, ix))
        iy = max(0, min(fh - 1, iy))

        hand_size = hand.hand_size_px()
        thickness = int(BRUSH_MIN + (BRUSH_MAX - BRUSH_MIN) * (hand_size / BRUSH_NORM_SIZE))
        thickness = max(BRUSH_MIN, min(BRUSH_MAX, thickness))

        color = BRUSH_COLORS[state.color_idx % len(BRUSH_COLORS)]

        if state.prev_point is not None:
            cv2.line(self._canvas, state.prev_point, (ix, iy),
                     color, thickness, cv2.LINE_AA)
            cv2.circle(self._canvas, (ix, iy), thickness // 2, color, -1, cv2.LINE_AA)

        state.prev_point = (ix, iy)

    def _render_canvas(self, frame: np.ndarray):
        mask = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        frame[mask > 0] = self._canvas[mask > 0]

    # ------------------------------------------------------------------
    # Палитры
    # ------------------------------------------------------------------

    def _draw_palette(self, frame: np.ndarray, fw: int, fh: int,
                      hands: list[HandData]) -> list[list[tuple]]:
        """
        Рисует по одной палитре для каждой руки.
        Рука 0 — внизу по центру. Рука 1 — внизу, правее.
        Возвращает список list[list[(cx, cy, r)]] для каждой руки.
        """
        all_rects = [[], []]
        n_colors   = len(BRUSH_COLORS)

        positions = [
            # Рука 0: центрирована
            (fw // 2, fh - PALETTE_MARGIN - PALETTE_RADIUS),
            # Рука 1: смещена вправо (или ниже если одна камера)
            (fw // 2, fh - PALETTE_MARGIN * 2 - PALETTE_RADIUS * 3 - 10),
        ]

        offsets_x = [
            -(n_colors * (PALETTE_RADIUS * 2 + PALETTE_GAP)) // 2,
            -(n_colors * (PALETTE_RADIUS * 2 + PALETTE_GAP)) // 2,
        ]

        labels = ["Игрок 1", "Игрок 2"]

        for hand_i, (base_cx, base_cy) in enumerate(positions):
            if hand_i >= len(hands):
                break   # Не рисуем палитру для отсутствующей руки

            state = self._hand_states[hand_i]

            # Подпись игрока
            cursor_color = HAND_CURSOR_COLORS[hand_i]
            cv2.putText(frame, labels[hand_i],
                        (fw // 2 - 40, base_cy - PALETTE_RADIUS - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, cursor_color, 1, cv2.LINE_AA)

            sx = offsets_x[hand_i]
            rects = []
            for j, color in enumerate(BRUSH_COLORS):
                cx = base_cx + sx + j * (PALETTE_RADIUS * 2 + PALETTE_GAP) + PALETTE_RADIUS
                cy = base_cy

                cv2.circle(frame, (cx + 2, cy + 2), PALETTE_RADIUS, (0,0,0), -1)
                cv2.circle(frame, (cx, cy), PALETTE_RADIUS, color, -1, cv2.LINE_AA)

                border = COLOR_WHITE if j == state.color_idx else (80, 80, 80)
                thick  = 3 if j == state.color_idx else 1
                cv2.circle(frame, (cx, cy), PALETTE_RADIUS, border, thick, cv2.LINE_AA)
                rects.append((cx, cy, PALETTE_RADIUS))

            all_rects[hand_i] = rects

        return all_rects

    def _check_palette_selection(self, ix: int, iy: int,
                                  rects: list[tuple], state: _HandDrawState):
        for j, (cx, cy, r) in enumerate(rects):
            if ((ix - cx)**2 + (iy - cy)**2) ** 0.5 <= r:
                state.color_idx = j
                break

    # ------------------------------------------------------------------
    # Юзер-бейджи
    # ------------------------------------------------------------------

    def _draw_user_badges(self, frame: np.ndarray, hands: list[HandData]):
        """Рисует бейджи «Игрок 1», «Игрок 2» в верхнем левом углу."""
        for i, hand in enumerate(hands[:2]):
            state  = self._hand_states[i]
            color  = BRUSH_COLORS[state.color_idx % len(BRUSH_COLORS)]
            cursor = HAND_CURSOR_COLORS[i % len(HAND_CURSOR_COLORS)]
            y      = HUD_HEIGHT + 30 + i * 32
            cv2.circle(frame, (16, y), 10, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (16, y), 10, cursor,  1, cv2.LINE_AA)
            cv2.putText(frame, f"P{i+1}  {hand.count_fingers()} paltsa",
                        (32, y + 5), cv2.FONT_HERSHEY_PLAIN, 1.1, COLOR_WHITE, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Индикатор очистки
    # ------------------------------------------------------------------

    def _draw_clear_indicator(self, frame: np.ndarray, fw: int, fh: int,
                               progress: float):
        """Показывает прогресс заряда жеста OK перед очисткой."""
        if progress < 1.0:
            text = f"OK 👌  {'█' * int(progress*10)}{'░'*(10-int(progress*10))}  ochishcheniye..."
            color = (0, int(120 + 135*progress), 255)
        else:
            text = "  KHOLST OCHISHCHEN  "
            color = (0, 255, 80)

        # Фон текста
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        cx = fw//2; cy = fh//2 + 20
        cv2.rectangle(frame, (cx-tw//2-10, cy-th-8), (cx+tw//2+10, cy+8),
                      (10,10,20), -1)
        put_text_centered(frame, text, cx, cy, FONT_BOLD, 0.65, color, 1)

        # Кольцо-прогресс
        angle = int(360 * progress)
        cv2.ellipse(frame, (fw//2, fh//2 - 40), (30, 30), -90, 0, angle,
                    color, 4, cv2.LINE_AA)
        cv2.ellipse(frame, (fw//2, fh//2 - 40), (30, 30), 0, 0, 360,
                    (40,40,40), 1, cv2.LINE_AA)
