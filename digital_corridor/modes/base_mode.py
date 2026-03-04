# =============================================================================
# modes/base_mode.py — Абстрактный базовый класс режима
# =============================================================================
"""
Каждый режим (меню, рисование, слайды, игры) наследуется от BaseMode
и обязан реализовать три метода:
  - on_enter()   — вызывается при входе в режим
  - update()     — вызывается каждый кадр, возвращает next_mode или None
  - on_exit()    — вызывается при выходе из режима (очистка)

Общий ресурс UIRenderer передаётся через конструктор.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from modules.gesture_detector import GestureDetector, Gesture
from modules.hand_tracker import HandData
from modules.ui import UIRenderer


class BaseMode(ABC):
    """
    Абстрактный базовый класс для всех режимов.

    Args:
        ui: экземпляр UIRenderer (общий для всех режимов)
    """

    def __init__(self, ui: UIRenderer):
        self._ui = ui
        self._active = False

    # ------------------------------------------------------------------
    # Жизненный цикл режима
    # ------------------------------------------------------------------

    def enter(self):
        """Переход в этот режим — инициализация."""
        self._active = True
        self.on_enter()

    def exit(self):
        """Выход из режима — очистка ресурсов."""
        self._active = False
        self.on_exit()

    @abstractmethod
    def on_enter(self):
        """Вызывается при входе в режим. Переопределить в наследнике."""
        pass

    @abstractmethod
    def on_exit(self):
        """Вызывается при выходе из режима. Переопределить в наследнике."""
        pass

    @abstractmethod
    def update(
        self,
        frame: np.ndarray,
        hands: list[HandData],
        gesture: str,
        primary_hand: Optional[HandData],
    ) -> Optional[str]:
        """
        Основной метод — вызывается каждый кадр.

        Args:
            frame        — текущий кадр (BGR), можно рисовать поверх
            hands        — список обнаруженных рук
            gesture      — текущий распознанный жест (Gesture.*)
            primary_hand — «ведущая» рука или None

        Returns:
            Название следующего режима (config.MODE_*) для переключения,
            или None если режим не меняется.
        """
        pass

    # ------------------------------------------------------------------
    # Утилита: возврат в меню по жесту «две ладони»
    # ------------------------------------------------------------------

    def _check_back_gesture(self, gesture: str) -> Optional[str]:
        """
        Если жест BOTH_OPEN — возвращаем MODE_MENU.
        Удобный хелпер для всех наследников.
        """
        from config import MODE_MENU
        if gesture == Gesture.BOTH_OPEN:
            return MODE_MENU
        return None
