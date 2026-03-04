# =============================================================================
# modules/hand_tracker.py — MediaPipe Tasks API + One Euro Filter
# =============================================================================
"""
HandTracker — обёртка над MediaPipe HandLandmarker (Tasks API).

Совместим с mediapipe >= 0.10.x (новый Tasks API).
Файл модели: hand_landmarker.task (скачивается автоматически при первом запуске).

Ключевые особенности для условий коридора:
  1. Низкий порог уверенности (0.55) → ловит руки издалека
  2. NUM_HANDS=2, RUNNING_MODE=VIDEO — стабильный трекинг
  3. One Euro Filter на каждый landmark → идеально плавные линии
  4. Рисование скелета в мини-превью для визуального фидбека
"""

import cv2
import numpy as np
import math
import time
import os
import urllib.request
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from config import (
    MP_MAX_HANDS, MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE,
    MP_MODEL_COMPLEXITY, ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF, PREVIEW_SIZE, PREVIEW_MARGIN,
    COLOR_ACCENT, COLOR_GREEN, COLOR_WHITE,
    FRAME_WIDTH, FRAME_HEIGHT, BASE_DIR
)

# Путь к файлу модели (лежит рядом с модулями в корне проекта)
_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# Соединения скелета (21 landmark) — те же что были в mp.solutions.hands.HAND_CONNECTIONS
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # Большой палец
    (0,5),(5,6),(6,7),(7,8),          # Указательный
    (0,9),(9,10),(10,11),(11,12),     # Средний
    (0,13),(13,14),(14,15),(15,16),   # Безымянный
    (0,17),(17,18),(18,19),(19,20),   # Мизинец
    (5,9),(9,13),(13,17),             # Поперечные ладонь
]


def _ensure_model():
    """Скачивает файл модели, если его нет."""
    if not os.path.isfile(_MODEL_PATH):
        print(f"[HandTracker] Скачиваю модель: {_MODEL_URL}")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print(f"[HandTracker] Модель сохранена: {_MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(
                f"Не удалось скачать модель MediaPipe: {e}\n"
                f"Скачайте вручную: {_MODEL_URL}\n"
                f"И положите как: {_MODEL_PATH}"
            )


# =============================================================================
# One Euro Filter
# =============================================================================

class _LowPassFilter:
    """Простой экспоненциальный LPF."""

    def __init__(self, alpha: float = 1.0):
        self._alpha = alpha
        self._y: Optional[float] = None
        self._s: Optional[float] = None

    def set_alpha(self, alpha: float):
        self._alpha = max(1e-7, min(alpha, 1.0))

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.set_alpha(alpha)
        if self._y is None:
            self._y = value
            self._s = value
        else:
            self._y = self._alpha * value + (1.0 - self._alpha) * self._s
            self._s = self._y
        return self._y

    @property
    def last_value(self) -> Optional[float]:
        return self._s

    def reset(self):
        self._y = None
        self._s = None


class OneEuroFilter:
    """
    One Euro Filter (Casiez et al., CHI 2012).
    Адаптивное сглаживание: плавно при медленном, отзывчиво при быстром.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        beta: float = ONE_EURO_BETA,
        d_cutoff: float = ONE_EURO_D_CUTOFF,
    ):
        self._freq       = max(1.0, freq)
        self._min_cutoff = min_cutoff
        self._beta       = beta
        self._d_cutoff   = d_cutoff
        self._x_filter   = _LowPassFilter()
        self._dx_filter  = _LowPassFilter()
        self._last_time: Optional[float] = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te  = 1.0 / self._freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        now = timestamp if timestamp is not None else time.time()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                self._freq = 1.0 / dt
        self._last_time = now

        prev_x = self._x_filter.last_value
        dx     = 0.0 if prev_x is None else (x - prev_x) * self._freq
        edx    = self._dx_filter.filter(dx, self._alpha(self._d_cutoff))
        cutoff = self._min_cutoff + self._beta * abs(edx)
        return self._x_filter.filter(x, self._alpha(cutoff))

    def reset(self):
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None


class PointFilter2D:
    """One Euro Filter для пары (x, y)."""

    def __init__(self, **kwargs):
        self.fx = OneEuroFilter(**kwargs)
        self.fy = OneEuroFilter(**kwargs)

    def filter(self, x: float, y: float, ts: Optional[float] = None):
        return self.fx.filter(x, ts), self.fy.filter(y, ts)

    def reset(self):
        self.fx.reset()
        self.fy.reset()


# =============================================================================
# HandData — данные одной руки
# =============================================================================

class HandData:
    """
    Нормализованные и пиксельные координаты одной обнаруженной руки
    + методы распознавания жестов по геометрии.
    """

    # Индексы landmark (совпадают с MediaPipe)
    WRIST      = 0
    THUMB_TIP  = 4
    THUMB_IP   = 3
    INDEX_MCP  = 5; INDEX_PIP  = 6; INDEX_DIP  = 7; INDEX_TIP  = 8
    MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
    RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
    PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

    def __init__(self, landmarks_norm: list, handedness: str,
                 frame_w: int, frame_h: int):
        """
        landmarks_norm: список (x, y, z) в [0..1] (уже отфильтрованные)
        handedness: 'Left' или 'Right'
        """
        self.handedness      = handedness
        self.landmarks_norm  = landmarks_norm   # list of (x, y, z)
        self.landmarks_px    = [
            (int(x * frame_w), int(y * frame_h))
            for x, y, z in landmarks_norm
        ]
        self.wrist_px     = self.landmarks_px[self.WRIST]
        self.index_tip_px = self.landmarks_px[self.INDEX_TIP]

    # ------------------------------------------------------------------
    # Жесты
    # ------------------------------------------------------------------

    def _finger_up(self, tip_id: int, pip_id: int) -> bool:
        """Палец выпрямлен: tip.y < pip.y (y растёт вниз)."""
        return self.landmarks_norm[tip_id][1] < self.landmarks_norm[pip_id][1]

    def _thumb_up(self) -> bool:
        """Большой палец вытянут (определяется по оси X)."""
        tip_x = self.landmarks_norm[self.THUMB_TIP][0]
        ip_x  = self.landmarks_norm[self.THUMB_IP][0]
        return tip_x > ip_x if self.handedness == 'Right' else tip_x < ip_x

    def fingers_state(self) -> list:
        """[Thumb, Index, Middle, Ring, Pinky] — True = выпрямлен."""
        return [
            self._thumb_up(),
            self._finger_up(self.INDEX_TIP,  self.INDEX_PIP),
            self._finger_up(self.MIDDLE_TIP, self.MIDDLE_PIP),
            self._finger_up(self.RING_TIP,   self.RING_PIP),
            self._finger_up(self.PINKY_TIP,  self.PINKY_PIP),
        ]

    def is_fist(self) -> bool:
        """Кулак: все 4 пальца (не большой) закрыты."""
        return not any(self.fingers_state()[1:])

    def is_open_palm(self) -> bool:
        """Открытая ладонь: все 4 пальца выпрямлены."""
        return all(self.fingers_state()[1:])

    def is_pointing_or_index_up(self) -> bool:
        """Как минимум указательный выпрямлен."""
        return self.fingers_state()[1]

    def count_fingers(self) -> int:
        return sum(self.fingers_state())

    def palm_center_px(self) -> tuple:
        wx, wy = self.landmarks_px[self.WRIST]
        mx, my = self.landmarks_px[self.MIDDLE_MCP]
        return (wx + mx) // 2, (wy + my) // 2

    def hand_size_px(self) -> float:
        wx, wy = self.landmarks_px[self.WRIST]
        mx, my = self.landmarks_px[self.MIDDLE_MCP]
        return math.hypot(mx - wx, my - wy)

    def is_ok_sign(self) -> bool:
        """
        Жест OK (👌): большой и указательный пальцы образуют кольцо
        (расстояние между кончиками мало), остальные три пальца выпрямлены.
        """
        # Расстояние кончик большого — кончик указательного
        tx, ty = self.landmarks_px[self.THUMB_TIP]
        ix, iy = self.landmarks_px[self.INDEX_TIP]
        dist   = math.hypot(tx - ix, ty - iy)
        size   = max(self.hand_size_px(), 1.0)
        pinched = dist / size < 0.28   # ~28% размера руки

        # Средний, безымянный, мизинец — выпрямлены
        state = self.fingers_state()
        others_up = state[2] and state[3] and state[4]

        return pinched and others_up



# =============================================================================
# HandTracker — Tasks API
# =============================================================================

class HandTracker:
    """
    Обёртка над MediaPipe HandLandmarker (Tasks API, mediapipe >= 0.10).

    Использование:
        tracker = HandTracker(frame_w=1280, frame_h=720)
        hands = tracker.process(frame_bgr)   # list[HandData]
        tracker.draw_preview_on_frame(frame)
    """

    def __init__(self, frame_w: int = FRAME_WIDTH, frame_h: int = FRAME_HEIGHT):
        self._frame_w = frame_w
        self._frame_h = frame_h

        # Убеждаемся что модель скачана
        _ensure_model()

        # One Euro Filter: ключ (hand_idx, lm_idx)
        self._filters: dict = {}

        # Превью скелета
        self._preview_img: Optional[np.ndarray] = None
        self._last_hands: list = []

        # Инициализируем HandLandmarker
        base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options   = HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=MP_MAX_HANDS,
            min_hand_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MP_TRACKING_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_ts   = 0       # Монотонный счётчик timestamp (мс)

        print(f"[HandTracker] Инициализирован. Модель: {_MODEL_PATH}")

    # ------------------------------------------------------------------
    # Основная обработка
    # ------------------------------------------------------------------

    def process(self, frame_bgr: np.ndarray) -> list:
        """
        Обрабатывает кадр. Возвращает список HandData (0–2 элемента).
        Применяет One Euro Filter к координатам.
        Строит превью скелета.
        """
        # Tasks API требует mp.Image в RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Монотонный timestamp в миллисекундах
        self._frame_ts += 33  # ~30fps
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        detected: list = []
        ts = time.time()

        if result.hand_landmarks and result.handedness:
            for i, (lm_list, hand_info) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                hand_label = hand_info[0].display_name  # 'Left' / 'Right'

                # Применяем One Euro Filter
                filtered_norm = self._apply_filters(i, lm_list, ts)

                hand = HandData(filtered_norm, hand_label,
                                self._frame_w, self._frame_h)
                detected.append(hand)
        else:
            self._reset_filters()

        self._last_hands = detected
        self._build_preview(frame_bgr, result)
        return detected

    def _apply_filters(self, hand_idx: int, lm_list: list, ts: float) -> list:
        """Применяет One Euro Filter к каждому landmark. Возвращает list[(x,y,z)]."""
        filtered = []
        for lm_idx, lm in enumerate(lm_list):
            key = (hand_idx, lm_idx)
            if key not in self._filters:
                self._filters[key] = PointFilter2D(
                    freq=30.0,
                    min_cutoff=ONE_EURO_MIN_CUTOFF,
                    beta=ONE_EURO_BETA,
                    d_cutoff=ONE_EURO_D_CUTOFF,
                )
            fx, fy = self._filters[key].filter(lm.x, lm.y, ts)
            filtered.append((fx, fy, lm.z))
        return filtered

    def _reset_filters(self):
        for f in self._filters.values():
            f.reset()

    # ------------------------------------------------------------------
    # Превью скелета
    # ------------------------------------------------------------------

    def _build_preview(self, frame_bgr: np.ndarray, result):
        """Строит PREVIEW_SIZE × PREVIEW_SIZE изображение со скелетом рук."""
        preview = np.full((PREVIEW_SIZE, PREVIEW_SIZE, 3), 20, dtype=np.uint8)

        has_hands = bool(result.hand_landmarks)

        if has_hands:
            scale = PREVIEW_SIZE / max(self._frame_w, self._frame_h)
            pw    = int(self._frame_w * scale)
            ph    = int(self._frame_h * scale)
            x_off = (PREVIEW_SIZE - pw) // 2
            y_off = (PREVIEW_SIZE - ph) // 2

            # Затемнённый уменьшенный кадр
            small = cv2.resize(frame_bgr, (pw, ph))
            small = cv2.addWeighted(small, 0.3, np.zeros_like(small), 0.7, 0)
            preview[y_off:y_off+ph, x_off:x_off+pw] = small

            for lm_list in result.hand_landmarks:
                # Скелет
                for idx_a, idx_b in HAND_CONNECTIONS:
                    xa = int(lm_list[idx_a].x * pw) + x_off
                    ya = int(lm_list[idx_a].y * ph) + y_off
                    xb = int(lm_list[idx_b].x * pw) + x_off
                    yb = int(lm_list[idx_b].y * ph) + y_off
                    cv2.line(preview, (xa, ya), (xb, yb), (80, 80, 80), 1)

                # Цветные точки по пальцам
                colors = [(255,255,255),(255,120,0),(0,200,255),
                          (0,255,100),(200,0,255),(255,50,50)]
                groups = [[0],[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
                for g, group in enumerate(groups):
                    for idx in group:
                        px = int(lm_list[idx].x * pw) + x_off
                        py = int(lm_list[idx].y * ph) + y_off
                        cv2.circle(preview, (px, py), 3, colors[g], -1)

        # Рамка и счётчик рук
        cv2.rectangle(preview, (0,0), (PREVIEW_SIZE-1, PREVIEW_SIZE-1), COLOR_ACCENT, 1)
        n = len(result.hand_landmarks) if has_hands else 0
        cv2.putText(preview, f"Ruki: {n}", (5, PREVIEW_SIZE-8),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, COLOR_ACCENT, 1, cv2.LINE_AA)

        self._preview_img = preview

    def get_preview(self) -> Optional[np.ndarray]:
        return self._preview_img

    def draw_preview_on_frame(self, frame: np.ndarray):
        """Накладывает превью в правый нижний угол кадра."""
        if self._preview_img is None:
            return
        fh, fw = frame.shape[:2]
        x1 = fw - PREVIEW_SIZE - PREVIEW_MARGIN
        y1 = fh - PREVIEW_SIZE - PREVIEW_MARGIN
        # Защита от выхода за границы
        if y1 >= 0 and x1 >= 0:
            frame[y1:y1+PREVIEW_SIZE, x1:x1+PREVIEW_SIZE] = self._preview_img

    def get_last_hands(self) -> list:
        return self._last_hands

    def close(self):
        """Освобождает ресурсы."""
        try:
            self._landmarker.close()
        except Exception:
            pass
