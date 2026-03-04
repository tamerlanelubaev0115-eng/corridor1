# =============================================================================
# modules/ui.py — UI-компоненты для OpenCV-окна
# =============================================================================
"""
UIRenderer предоставляет:
  - Кнопки с таймером удержания (визуальный прогресс-круг)
  - HUD верхняя полоса (FPS, режим, жест)
  - Overlay «Назад в меню»
  - Рисование скруглённых прямоугольников (OpenCV не умеет из коробки)
  - Полупрозрачные overlay-прямоугольники
  - Idle-индикатор (обратный отсчёт на экране)
"""

import cv2
import numpy as np
import math
import time
from typing import Optional
from config import (
    COLOR_WHITE, COLOR_BLACK, COLOR_ACCENT, COLOR_ACCENT2,
    COLOR_GREEN, COLOR_RED, COLOR_DARK, COLOR_GRAY,
    BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_RADIUS, TIMER_RADIUS,
    HUD_HEIGHT, HOVER_CONFIRM_SECONDS, FONT_MAIN, FONT_BOLD, FONT_SMALL,
    IDLE_TIMEOUT_SECONDS
)


# =============================================================================
# Утилиты рисования
# =============================================================================

def draw_rounded_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    radius: int,
    color: tuple,
    thickness: int = -1,
    alpha: float = 1.0,
):
    """
    Рисует прямоугольник со скруглёнными углами.
    thickness=-1 → заливка; alpha < 1 → полупрозрачность.
    """
    overlay = img.copy() if alpha < 1.0 else img

    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    # Заливка центра и полос
    if thickness == -1:
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
        # Углы
        cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90,  color, -1)
        cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90,  color, -1)
        cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r),  90, 0, 90,  color, -1)
        cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r),   0, 0, 90,  color, -1)
    else:
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.rectangle(overlay, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.rectangle(overlay, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90,  color, thickness)
        cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90,  color, thickness)
        cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r),  90, 0, 90,  color, thickness)
        cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r),   0, 0, 90,  color, thickness)

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def draw_circle_progress(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    progress: float,        # 0.0 .. 1.0
    bg_color: tuple  = (60, 60, 60),
    fg_color: tuple  = COLOR_ACCENT,
    thickness: int   = 4,
):
    """
    Рисует круговой прогресс-бар (дуга от верха по часовой стрелке).
    progress: 0.0 = пусто, 1.0 = полный круг.
    """
    # Фоновый круг (серый)
    cv2.circle(img, (cx, cy), radius, bg_color, thickness)

    if progress > 0.01:
        # Дуга: начало −90° (верх), по часовой стрелке
        angle = int(360 * progress)
        cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, angle,
                    fg_color, thickness, cv2.LINE_AA)

    # Центральная «точка» прогресса
    if progress >= 1.0:
        cv2.circle(img, (cx, cy), radius - thickness - 2, COLOR_GREEN, -1)


def put_text_centered(
    img: np.ndarray,
    text: str,
    cx: int, cy: int,
    font=FONT_MAIN,
    scale: float = 1.0,
    color: tuple = COLOR_WHITE,
    thickness: int = 1,
):
    """Рисует текст с выравниванием по центру точки (cx, cy)."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = cx - tw // 2
    y = cy + th // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# =============================================================================
# HoverButton — кнопка с таймером удержания
# =============================================================================

class HoverButton:
    """
    Кнопка, которая срабатывает после HOVER_CONFIRM_SECONDS под пальцем.

    Состояния:
      IDLE     → HOVERING (палец над кнопкой) → TRIGGERED → IDLE
    """

    STATE_IDLE      = "idle"
    STATE_HOVERING  = "hovering"
    STATE_TRIGGERED = "triggered"

    def __init__(
        self,
        x: int, y: int,
        width: int  = BUTTON_WIDTH,
        height: int = BUTTON_HEIGHT,
        label: str  = "",
        icon: str   = "",          # Юникод-символ (emoji)
        color_bg: tuple     = (40, 40, 55),
        color_hover: tuple  = (60, 60, 80),
        color_accent: tuple = COLOR_ACCENT,
    ):
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        self.label       = label
        self.icon        = icon
        self.color_bg    = color_bg
        self.color_hover = color_hover
        self.color_accent= color_accent

        self._state      = self.STATE_IDLE
        self._hover_start: Optional[float] = None
        self._triggered  = False

    # ------------------------------------------------------------------
    def update(self, finger_x: int, finger_y: int) -> bool:
        """
        Обновляет состояние кнопки по позиции пальца.
        Возвращает True в момент срабатывания (один раз).
        """
        inside = (
            self.x <= finger_x <= self.x + self.w
            and self.y <= finger_y <= self.y + self.h
        )

        if inside:
            if self._state == self.STATE_IDLE:
                self._state = self.STATE_HOVERING
                self._hover_start = time.time()
                self._triggered = False

            elif self._state == self.STATE_HOVERING:
                elapsed = time.time() - self._hover_start
                if elapsed >= HOVER_CONFIRM_SECONDS:
                    self._state    = self.STATE_TRIGGERED
                    self._triggered = True
                    return True

        else:
            # Палец ушёл
            if self._state == self.STATE_TRIGGERED:
                self._state = self.STATE_IDLE
            elif self._state == self.STATE_HOVERING:
                self._state = self.STATE_IDLE
                self._hover_start = None

        return False

    def reset(self):
        """Сбросить в исходное состояние (например, после входа в режим)."""
        self._state     = self.STATE_IDLE
        self._hover_start = None
        self._triggered = False

    def get_progress(self) -> float:
        """0.0..1.0 — прогресс удержания."""
        if self._state != self.STATE_HOVERING or self._hover_start is None:
            return 0.0
        elapsed = time.time() - self._hover_start
        return min(elapsed / HOVER_CONFIRM_SECONDS, 1.0)

    # ------------------------------------------------------------------
    def draw(self, img: np.ndarray, cursor_pos: Optional[tuple] = None):
        """
        Рисует кнопку на img.
        cursor_pos — (x, y) пальца для эффекта hover.
        """
        is_hover = self._state in (self.STATE_HOVERING, self.STATE_TRIGGERED)
        bg = self.color_hover if is_hover else self.color_bg

        # Тень (смещённый прямоугольник)
        draw_rounded_rect(img, self.x+4, self.y+4, self.x+self.w+4, self.y+self.h+4,
                          BUTTON_RADIUS, (10, 10, 15), alpha=0.6)

        # Фон кнопки
        draw_rounded_rect(img, self.x, self.y, self.x+self.w, self.y+self.h,
                          BUTTON_RADIUS, bg, alpha=0.92)

        # Акцентная левая полоса
        draw_rounded_rect(img, self.x, self.y, self.x+6, self.y+self.h,
                          3, self.color_accent)

        # Иконка
        cx = self.x + self.w // 2
        cy = self.y + self.h // 2

        if self.icon:
            put_text_centered(img, self.icon, cx - 50, cy,
                              FONT_BOLD, 1.4, COLOR_WHITE, 2)

        # Надпись
        put_text_centered(img, self.label, cx + (20 if self.icon else 0), cy,
                          FONT_MAIN, 0.9, COLOR_WHITE, 1)

        # Прогресс-круг (справа от кнопки)
        progress = self.get_progress()
        if is_hover:
            timer_x = self.x + self.w + TIMER_RADIUS + 10
            timer_y = self.y + self.h // 2
            draw_circle_progress(img, timer_x, timer_y, TIMER_RADIUS, progress)

            # Цифра секунд внутри круга
            remaining = HOVER_CONFIRM_SECONDS * (1.0 - progress)
            put_text_centered(img, f"{remaining:.0f}",
                              timer_x, timer_y, FONT_BOLD, 0.7, COLOR_WHITE, 1)

        # Рамка при hover
        if is_hover:
            draw_rounded_rect(img, self.x, self.y, self.x+self.w, self.y+self.h,
                              BUTTON_RADIUS, self.color_accent, thickness=2)


# =============================================================================
# UIRenderer — глобальный рендерер интерфейса
# =============================================================================

class UIRenderer:
    """Отрисовывает общие UI-элементы: HUD, статус жеста, idle-таймер."""

    def __init__(self):
        self._gesture_label = ""
        self._mode_label    = ""
        self._hint_text     = ""

    def draw_hud(
        self,
        frame: np.ndarray,
        mode_label: str,
        fps: float,
        gesture_label: str = "",
        n_hands: int = 0,
    ):
        """
        Рисует HUD-полосу сверху.
        Содержит: название режима | FPS | кол-во рук | текущий жест.
        """
        fh, fw = frame.shape[:2]

        # Полупрозрачный тёмный фон HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, HUD_HEIGHT), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Разделительная линия
        cv2.line(frame, (0, HUD_HEIGHT), (fw, HUD_HEIGHT), COLOR_ACCENT, 1)

        # Слева: название режима
        cv2.putText(frame, mode_label, (16, HUD_HEIGHT - 16),
                    FONT_MAIN, 0.75, COLOR_ACCENT, 1, cv2.LINE_AA)

        # По центру: жест
        if gesture_label:
            put_text_centered(frame, f"[ {gesture_label} ]",
                              fw // 2, HUD_HEIGHT // 2,
                              FONT_BOLD, 0.65, COLOR_WHITE, 1)

        # Справа: руки + FPS
        fps_str   = f"FPS: {fps:.0f}"
        hands_str = f"Руки: {n_hands}"
        cv2.putText(frame, fps_str,   (fw - 130, HUD_HEIGHT - 16),
                    FONT_SMALL, 1.4, COLOR_GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, hands_str, (fw - 130, 22),
                    FONT_SMALL, 1.2, COLOR_GRAY if n_hands == 0 else COLOR_WHITE,
                    1, cv2.LINE_AA)

    def draw_back_hint(self, frame: np.ndarray):
        """
        Рисует подсказку «✋✋ — Назад в меню» внизу экрана.
        """
        fh, fw = frame.shape[:2]
        text = "Две ладони   Назад в меню"
        (tw, _), _ = cv2.getTextSize(text, FONT_SMALL, 1.2, 1)
        x = (fw - tw) // 2
        y = fh - 12

        # Полупрозрачный фон под текстом
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 22), (x + tw + 10, y + 6),
                      (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, text, (x, y),
                    FONT_SMALL, 1.2, COLOR_GRAY, 1, cv2.LINE_AA)

    def draw_idle_warning(self, frame: np.ndarray, elapsed: float):
        """
        Рисует предупреждение об idle и обратный отсчёт.
        Вызывается когда elapsed > IDLE_TIMEOUT_SECONDS * 0.7.
        """
        fh, fw = frame.shape[:2]
        remaining = max(0.0, IDLE_TIMEOUT_SECONDS - elapsed)
        progress = elapsed / IDLE_TIMEOUT_SECONDS

        # Тёмный overlay по всему кадру
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (5, 5, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Центральный круг обратного отсчёта
        cx, cy = fw // 2, fh // 2
        draw_circle_progress(frame, cx, cy, 80, progress,
                             (40, 40, 50), COLOR_RED, thickness=8)
        put_text_centered(frame, f"{remaining:.0f}",
                          cx, cy, FONT_BOLD, 2.5, COLOR_WHITE, 3)

        put_text_centered(frame, "Нет движения — возврат в меню",
                          cx, cy + 110, FONT_SMALL, 1.5, COLOR_GRAY, 1)

    def draw_finger_cursor(
        self, frame: np.ndarray, x: int, y: int,
        color: tuple = COLOR_ACCENT, radius: int = 12
    ):
        """Рисует красивый курсор-кружок на кончике пальца."""
        cv2.circle(frame, (x, y), radius,     color,       2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius // 3, color,      -1, cv2.LINE_AA)
        # Крестик
        cv2.line(frame, (x - radius, y), (x + radius, y), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x, y - radius), (x, y + radius), color, 1, cv2.LINE_AA)

    def draw_status_bar(self, frame: np.ndarray, text: str, y_offset: int = 0):
        """Полоса с текстом статуса под HUD."""
        fh, fw = frame.shape[:2]
        y = HUD_HEIGHT + 30 + y_offset
        put_text_centered(frame, text, fw // 2, y,
                          FONT_BOLD, 0.7, COLOR_ACCENT, 1)
