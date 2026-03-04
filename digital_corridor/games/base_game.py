# =============================================================================
# games/base_game.py — Абстрактный базовый класс встроенной игры
# =============================================================================
"""
Каждая встроенная игра наследуется от BaseGame и обязана реализовать:
  - reset()   — сброс игры в начальное состояние
  - update(dt, zone, is_open_palm) — обновить физику/логику
  - render(canvas)  — нарисовать игру на numpy-массиве (BGR)

Рендер происходит в виртуальном разрешении (VIRT_W × VIRT_H),
затем масштабируется на реальный кадр.
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2


VIRT_W = 640   # Виртуальная ширина игровой области
VIRT_H = 360   # Виртуальная высота игровой области

# Цвета (BGR)
C_BG      = (18,  18,  28 )
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0  )
C_GREEN   = (0,   220, 80 )
C_RED     = (0,   50,  230)
C_YELLOW  = (0,   220, 255)
C_CYAN    = (220, 200, 0  )
C_ORANGE  = (0,   140, 255)
C_GRAY    = (80,  80,  80 )
C_DKGRAY  = (40,  40,  50 )


class BaseGame(ABC):
    """Базовый класс встроенной мини-игры."""

    def __init__(self):
        self.score      = 0
        self.high_score = 0
        self.lives      = 3
        self.game_over  = False
        self.paused     = False

    @abstractmethod
    def reset(self):
        """Сброс в начальное состояние."""
        pass

    @abstractmethod
    def update(self, dt: float, zone: str, is_open_palm: bool):
        """
        dt           — delta time в секундах
        zone         — 'up'/'down'/'left'/'right'/'center'
        is_open_palm — True если открытая ладонь (огонь/прыжок)
        """
        pass

    @abstractmethod
    def render(self, canvas: np.ndarray):
        """Рисует игру на canvas (VIRT_W × VIRT_H, BGR)."""
        pass

    # ------------------------------------------------------------------
    # Утилиты рисования
    # ------------------------------------------------------------------

    @staticmethod
    def fill_bg(canvas: np.ndarray, color=C_BG):
        canvas[:] = color

    @staticmethod
    def draw_text(canvas, text, x, y, scale=0.6, color=C_WHITE, thickness=1):
        cv2.putText(canvas, text, (int(x), int(y)),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_text_center(canvas, text, cx, cy, scale=0.6, color=C_WHITE, thickness=1):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        cv2.putText(canvas, text, (int(cx - tw/2), int(cy + th/2)),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

    def draw_score(self, canvas):
        self.draw_text(canvas, f"Score: {self.score}", 10, 24, 0.6, C_YELLOW)
        self.draw_text(canvas, f"Best:  {self.high_score}", 10, 46, 0.5, C_GRAY)

    def draw_game_over(self, canvas):
        h, w = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0,0),(w,h),(5,5,10),-1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        self.draw_text_center(canvas, "GAME OVER", w//2, h//2 - 30, 1.2, C_RED, 2)
        self.draw_text_center(canvas, f"Score: {self.score}", w//2, h//2 + 10, 0.8, C_WHITE, 1)
        self.draw_text_center(canvas, "Raise OPEN PALM to restart", w//2, h//2 + 45, 0.55, C_GRAY, 1)
