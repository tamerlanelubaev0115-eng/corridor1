# =============================================================================
# modules/camera.py — Singleton захват видео с фоновым потоком
# =============================================================================
"""
Реализует паттерн Singleton для cv2.VideoCapture.
Фоновый поток постоянно читает кадры — основной поток всегда получает
свежий кадр без ожидания. Это устраняет главный ботлнек FPS.

Настройки оптимизированы для:
  - Плохого освещения (яркость, контраст через CLAHE при необходимости)
  - Дальней камеры (высокое разрешение → медиапайп видит мелкие руки)
  - Стабильного FPS (буфер кадров = 1, MJPG кодек)
"""

import cv2
import threading
import time
import sys
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS
)


class CameraCapture:
    """
    Singleton-класс для захвата видео с камеры.

    Использование:
        cam = CameraCapture.get_instance()
        frame = cam.read()
        cam.release()
    """

    _instance = None          # Единственный экземпляр (Singleton)
    _lock = threading.Lock()  # Блокировка для потокобезопасной инициализации

    @classmethod
    def get_instance(cls) -> "CameraCapture":
        """Возвращает единственный экземпляр камеры (создаёт при первом вызове)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Двойная проверка (double-checked locking)
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Инициализация камеры — вызывается только один раз из get_instance()."""
        if CameraCapture._instance is not None:
            raise RuntimeError(
                "CameraCapture — Singleton! Используйте CameraCapture.get_instance()"
            )

        self._cap = None
        self._frame = None          # Последний полученный кадр
        self._frame_lock = threading.Lock()
        self._running = False       # Флаг работы фонового потока
        self._thread = None
        self._initialized = False
        self._fps_counter = 0
        self._fps_time = time.time()
        self.current_fps = 0.0

        self._open_camera()

    # ------------------------------------------------------------------
    # Открытие камеры
    # ------------------------------------------------------------------
    def _open_camera(self):
        """Открывает камеру и применяет оптимальные настройки."""
        # Пробуем несколько backend'ов для надёжности на Windows
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            self._cap = cv2.VideoCapture(CAMERA_INDEX, backend)
            if self._cap.isOpened():
                break

        if not self._cap.isOpened():
            print(
                f"[ОШИБКА] Камера с индексом {CAMERA_INDEX} не найдена.\n"
                "Проверьте подключение камеры и перезапустите программу."
            )
            sys.exit(1)

        # --- Настройки разрешения и FPS ---
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

        # --- Кодек MJPG: значительно ускоряет USB-камеры ---
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # --- Буфер = 1 кадр → всегда получаем свежий кадр ---
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # --- Автофокус (если камера поддерживает) ---
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # --- Автоэкспозиция — активируем для тёмных помещений ---
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        # Читаем реальные размеры (камера могла не поддержать запрошенные)
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[Камера] Открыта: {self.width}x{self.height} @ {TARGET_FPS}fps")

        # Запускаем фоновый поток чтения кадров
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="CameraThread", daemon=True
        )
        self._thread.start()
        self._initialized = True

    # ------------------------------------------------------------------
    # Фоновый поток
    # ------------------------------------------------------------------
    def _capture_loop(self):
        """
        Фоновый поток: постоянно захватывает кадры и сохраняет последний.
        Использование отдельного потока устраняет задержку декодирования USB.
        """
        while self._running:
            if not self._cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if ret and frame is not None:
                # Зеркалим по горизонтали (естественное для пользователя)
                frame = cv2.flip(frame, 1)

                with self._frame_lock:
                    self._frame = frame

                # Подсчёт FPS фонового потока
                self._fps_counter += 1
                now = time.time()
                if now - self._fps_time >= 1.0:
                    self.current_fps = self._fps_counter / (now - self._fps_time)
                    self._fps_counter = 0
                    self._fps_time = now
            else:
                # Небольшая пауза при ошибке чтения
                time.sleep(0.005)

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------
    def read(self):
        """
        Возвращает последний захваченный кадр (BGR, уже зеркальный).
        Возвращает (True, frame) или (False, None) если кадр ещё не готов.
        """
        with self._frame_lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def is_opened(self) -> bool:
        """Проверяет, открыта ли камера."""
        return self._cap is not None and self._cap.isOpened()

    def release(self):
        """Останавливает фоновый поток и освобождает камеру."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        CameraCapture._instance = None
        print("[Камера] Освобождена.")

    def get_fps(self) -> float:
        """Возвращает текущий FPS захвата."""
        return self.current_fps
