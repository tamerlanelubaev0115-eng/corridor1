# =============================================================================
# modules/gesture_detector.py — Детектор жестов высокого уровня
# =============================================================================
"""
GestureDetector принимает список HandData и классифицирует жесты:

  - POINTING      : только указательный палец выпрямлен
  - FIST          : кулак (очистка холста, подтверждение)
  - OPEN_PALM     : все 4 пальца выпрямлены
  - SWIPE_LEFT    : быстрый взмах открытой ладони влево
  - SWIPE_RIGHT   : быстрый взмах открытой ладони вправо
  - BOTH_OPEN     : две открытые ладони → возврат в меню
  - NONE          : нет определённого жеста / нет рук

Особенности:
  - Свайп ТОЛЬКО при открытой ладони (нет конфликта с рисованием)
  - EMA-сглаживание позиции запястья для надёжного свайпа
  - Hysteresis для кулака (нужно N кадров подряд)
  - Idle Timer: отслеживает время отсутствия рук
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from modules.hand_tracker import HandData
from config import (
    SWIPE_MIN_SPEED, SWIPE_HISTORY_FRAMES,
    FIST_FRAMES_THRESHOLD, BOTH_HANDS_BACK_FRAMES,
    IDLE_TIMEOUT_SECONDS, EMA_ALPHA, FRAME_WIDTH
)


# =============================================================================
# Константы жестов
# =============================================================================

class Gesture:
    NONE        = "none"
    POINTING    = "pointing"       # Указательный палец
    FIST        = "fist"           # Кулак
    OPEN_PALM   = "open_palm"      # Открытая ладонь
    SWIPE_LEFT  = "swipe_left"     # Взмах влево
    SWIPE_RIGHT = "swipe_right"    # Взмах вправо
    BOTH_OPEN   = "both_open"      # Две ладони → назад в меню


# =============================================================================
# HandState — состояние одной руки между кадрами
# =============================================================================

@dataclass
class HandState:
    """Внутреннее состояние для одной руки между кадрами."""
    # История позиций запястья (нормализованные x, метка времени)
    wrist_x_history: deque = field(
        default_factory=lambda: deque(maxlen=SWIPE_HISTORY_FRAMES)
    )
    # EMA-сглаженная позиция запястья
    smooth_x: Optional[float] = None
    smooth_y: Optional[float] = None

    # Счётчики hysteresis
    fist_frames: int = 0
    open_frames: int = 0
    both_open_frames: int = 0

    # Последний зарегистрированный свайп (защита от двойного срабатывания)
    last_swipe_time: float = 0.0
    SWIPE_COOLDOWN   = 0.8  # секунд между свайпами


# =============================================================================
# GestureDetector
# =============================================================================

class GestureDetector:
    """
    Определяет жесты верхнего уровня из списка HandData.

    Использование:
        detector = GestureDetector()
        while True:
            hands = tracker.process(frame)
            gesture, primary_hand = detector.update(hands)
    """

    def __init__(self):
        # Состояние для каждой руки (индекс 0 и 1)
        self._states = [HandState(), HandState()]

        # Время последнего обнаружения рук (для Idle Timer)
        self._last_hand_seen: float = time.time()

        # Флаг подтверждённого возврата в меню (both_open сработал)
        self._back_triggered: bool = False

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def update(self, hands: list[HandData]) -> tuple[str, Optional[HandData]]:
        """
        Обновляет состояние и возвращает кортеж (gesture, primary_hand).

        gesture      — одна из констант Gesture.*
        primary_hand — HandData «ведущей» руки (None если рук нет)
        """
        now = time.time()

        # Обновляем idle-таймер
        if hands:
            self._last_hand_seen = now

        # Нет рук в кадре
        if not hands:
            self._reset_all_counters()
            return Gesture.NONE, None

        # Первичная рука — первая из списка
        primary = hands[0]
        state   = self._states[0]

        # Обновляем EMA позиции запястья
        wx_norm = primary.landmarks_norm[HandData.WRIST][0]
        wy_norm = primary.landmarks_norm[HandData.WRIST][1]
        state.smooth_x = self._ema(state.smooth_x, wx_norm)
        state.smooth_y = self._ema(state.smooth_y, wy_norm)

        # --- Проверка обеих рук для возврата в меню ---
        if len(hands) >= 2:
            gesture = self._check_both_open(hands, state, now)
            if gesture == Gesture.BOTH_OPEN:
                return gesture, primary
        else:
            state.both_open_frames = 0
            self._back_triggered = False

        # --- Свайп (только при открытой ладони) ---
        swipe = self._check_swipe(primary, state, now)
        if swipe in (Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT):
            return swipe, primary

        # --- Кулак (с hysteresis) ---
        if primary.is_fist():
            state.fist_frames += 1
            state.open_frames  = 0
            if state.fist_frames >= FIST_FRAMES_THRESHOLD:
                return Gesture.FIST, primary
        else:
            state.fist_frames = max(0, state.fist_frames - 1)

        # --- Открытая ладонь ---
        if primary.is_open_palm():
            state.open_frames += 1
            return Gesture.OPEN_PALM, primary
        else:
            state.open_frames = 0

        # --- Указательный жест ---
        if primary.is_pointing_or_index_up():
            return Gesture.POINTING, primary

        return Gesture.NONE, primary

    # ------------------------------------------------------------------
    # Вспомогательные проверки
    # ------------------------------------------------------------------

    def _check_swipe(
        self, hand: HandData, state: HandState, now: float
    ) -> str:
        """
        Определяет свайп ТОЛЬКО при открытой ладони.
        Анализирует изменение нормализованной X-позиции запястья за N кадров.
        """
        wx_norm = hand.landmarks_norm[HandData.WRIST][0]

        # Добавляем в историю только при открытой ладони
        if hand.is_open_palm():
            state.wrist_x_history.append((wx_norm, now))
        else:
            # Рука не открыта → сбрасываем историю свайпа
            state.wrist_x_history.clear()
            return Gesture.NONE

        # Недостаточно точек для анализа
        if len(state.wrist_x_history) < 4:
            return Gesture.NONE

        # Проверяем cooldown между свайпами
        if now - state.last_swipe_time < HandState.SWIPE_COOLDOWN:
            return Gesture.NONE

        # Вычисляем скорость: (delta_x_norm) / (delta_t)
        oldest_x, oldest_t = state.wrist_x_history[0]
        newest_x, newest_t = state.wrist_x_history[-1]
        dt = newest_t - oldest_t
        if dt < 0.05:  # Слишком мало времени
            return Gesture.NONE

        speed = (newest_x - oldest_x) / dt  # нормализованные единицы/сек

        if speed > SWIPE_MIN_SPEED:         # Быстро вправо
            state.wrist_x_history.clear()
            state.last_swipe_time = now
            return Gesture.SWIPE_RIGHT

        elif speed < -SWIPE_MIN_SPEED:      # Быстро влево
            state.wrist_x_history.clear()
            state.last_swipe_time = now
            return Gesture.SWIPE_LEFT

        return Gesture.NONE

    def _check_both_open(
        self, hands: list[HandData], state: HandState, now: float
    ) -> str:
        """Две открытые ладони подряд N кадров → BOTH_OPEN (возврат в меню)."""
        both_open = all(h.is_open_palm() for h in hands[:2])

        if both_open:
            state.both_open_frames += 1
        else:
            state.both_open_frames = 0
            self._back_triggered = False

        if state.both_open_frames >= BOTH_HANDS_BACK_FRAMES and not self._back_triggered:
            self._back_triggered = True
            state.both_open_frames = 0
            return Gesture.BOTH_OPEN

        return Gesture.NONE

    def _reset_all_counters(self):
        """Сбрасываем все счётчики когда рук нет."""
        for s in self._states:
            s.fist_frames      = 0
            s.open_frames      = 0
            s.both_open_frames = 0
            s.wrist_x_history.clear()
            s.smooth_x = None
            s.smooth_y = None
        self._back_triggered = False

    @staticmethod
    def _ema(prev: Optional[float], new: float) -> float:
        """Exponential Moving Average сглаживание."""
        if prev is None:
            return new
        return EMA_ALPHA * new + (1.0 - EMA_ALPHA) * prev

    # ------------------------------------------------------------------
    # Idle Timer
    # ------------------------------------------------------------------

    def is_idle(self) -> bool:
        """True если руки не были в кадре дольше IDLE_TIMEOUT_SECONDS."""
        return (time.time() - self._last_hand_seen) >= IDLE_TIMEOUT_SECONDS

    def idle_elapsed(self) -> float:
        """Сколько секунд прошло без рук."""
        return time.time() - self._last_hand_seen

    def reset_idle(self):
        """Сбросить idle-таймер вручную."""
        self._last_hand_seen = time.time()

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    def get_index_tip_norm(self, hand: HandData) -> tuple[float, float]:
        """Нормализованные координаты кончика указательного пальца."""
        lm = hand.landmarks_norm[HandData.INDEX_TIP]
        return lm[0], lm[1]

    def get_index_tip_px(self, hand: HandData) -> tuple[int, int]:
        """Пиксельные координаты кончика указательного пальца."""
        return hand.landmarks_px[HandData.INDEX_TIP]
