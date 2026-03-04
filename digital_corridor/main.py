# =============================================================================
# main.py — Точка входа «Цифровой коридор»
# =============================================================================
"""
Главный файл. Запускает инсталляцию «Цифровой коридор».

Архитектура:
  - FSM (конечный автомат) для переключения между режимами
  - Singleton камера (CameraCapture) — не переинициализируется
  - HandTracker на базе MediaPipe — трекает руки каждый кадр
  - GestureDetector — определяет жест из HandData
  - Idle Timer — 60 сек без рук → возврат в меню
  - Превью скелета руки — рисуется в правом нижнем углу

Запуск:
  python main.py

Выход:
  Нажмите ESC или Q в окне OpenCV

Требования:
  Установить зависимости: pip install -r requirements.txt
"""

import sys
import os
import time
import cv2
import numpy as np

# Добавляем корень проекта в sys.path для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODE_MENU, MODE_DRAWING, MODE_SLIDES, MODE_GAMES,
    MODE_LABELS, HUD_HEIGHT, COLOR_RED, COLOR_WHITE,
    FONT_BOLD, IDLE_TIMEOUT_SECONDS
)
from modules.camera        import CameraCapture
from modules.hand_tracker  import HandTracker
from modules.gesture_detector import GestureDetector, Gesture
from modules.ui            import UIRenderer

from modes.menu_mode       import MenuMode
from modes.drawing_mode    import DrawingMode
from modes.slides_mode     import SlidesMode
from modes.games_mode      import GamesMode


# =============================================================================
# Приложение
# =============================================================================

class DigitalCorridorApp:
    """
    Главный класс приложения «Цифровой коридор».
    Управляет FSM режимов, захватом кадров, трекингом и отрисовкой.
    """

    WINDOW_NAME = "Cifrovoy Koridor"

    def __init__(self):
        # --- Инициализация камеры (Singleton) ---
        print("[Main] Инициализация камеры...")
        self._cam = CameraCapture.get_instance()

        # Ждём первый кадр
        timeout = 5.0
        start   = time.time()
        while time.time() - start < timeout:
            ok, frame = self._cam.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        else:
            print("[ОШИБКА] Камера не даёт кадров. Проверьте подключение.")
            sys.exit(1)

        fh, fw = frame.shape[:2]
        print(f"[Main] Камера готова: {fw}x{fh}")

        # --- Трекер рук ---
        self._tracker  = HandTracker(frame_w=fw, frame_h=fh)

        # --- Детектор жестов ---
        self._detector = GestureDetector()

        # --- UI ---
        self._ui = UIRenderer()

        # --- Режимы ---
        self._modes = {
            MODE_MENU:    MenuMode(self._ui),
            MODE_DRAWING: DrawingMode(self._ui),
            MODE_SLIDES:  SlidesMode(self._ui),
            MODE_GAMES:   GamesMode(self._ui),
        }

        # --- FSM: начинаем с меню ---
        self._current_mode_key = MODE_MENU
        self._modes[MODE_MENU].enter()

        # --- FPS-счётчик (основного цикла) ---
        self._fps_display = 0.0
        self._fps_frames  = 0
        self._fps_time    = time.time()

        # --- Создаём окно OpenCV ---
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

        print("[Main] Инициализация завершена. Старт!")

    # ------------------------------------------------------------------
    # Главный цикл
    # ------------------------------------------------------------------

    def run(self):
        """Основной цикл приложения."""
        while True:
            # --- Читаем кадр ---
            ok, frame = self._cam.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            fh, fw = frame.shape[:2]

            # --- Трекинг рук ---
            try:
                hands = self._tracker.process(frame)
            except Exception as e:
                print(f"[HandTracker] Ошибка: {e}")
                hands = []

            # --- Распознавание жеста ---
            try:
                gesture, primary_hand = self._detector.update(hands)
            except Exception as e:
                print(f"[GestureDetector] Ошибка: {e}")
                gesture, primary_hand = Gesture.NONE, None

            # --- Idle Timer ---
            idle_elapsed = self._detector.idle_elapsed()
            if self._detector.is_idle():
                # Время вышло → сброс в меню
                if self._current_mode_key != MODE_MENU:
                    self._switch_mode(MODE_MENU)
                self._detector.reset_idle()
            elif idle_elapsed > IDLE_TIMEOUT_SECONDS * 0.7:
                # Предупреждение за 30% оставшегося времени
                self._ui.draw_idle_warning(frame, idle_elapsed)

            # --- Обновляем текущий режим ---
            try:
                next_mode = self._modes[self._current_mode_key].update(
                    frame, hands, gesture, primary_hand
                )
            except Exception as e:
                print(f"[Mode:{self._current_mode_key}] Ошибка: {e}")
                next_mode = None

            # --- Переключение режима ---
            if next_mode and next_mode != self._current_mode_key:
                self._switch_mode(next_mode)

            # --- Превью скелета рук (правый нижний угол) ---
            self._tracker.draw_preview_on_frame(frame)

            # --- FPS ---
            self._update_fps()
            # Рисуем FPS поверх HUD (обновляем значение в HUD)
            cv2.putText(
                frame,
                f"FPS: {self._fps_display:.0f}",
                (fw - 120, HUD_HEIGHT - 16),
                FONT_BOLD, 0.65, (100, 220, 100), 1, cv2.LINE_AA
            )

            # --- Отображаем кадр ---
            cv2.imshow(self.WINDOW_NAME, frame)

            # --- Обработка клавиш ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q или ESC — выход
                break
            elif key == ord('m'):                # M — принудительно в меню
                self._switch_mode(MODE_MENU)

        self._shutdown()

    # ------------------------------------------------------------------
    # FSM Переключение
    # ------------------------------------------------------------------

    def _switch_mode(self, new_mode: str):
        """Безопасно переключает режим: вызывает exit() у старого, enter() у нового."""
        if new_mode not in self._modes:
            print(f"[Main] Неизвестный режим: {new_mode}")
            return

        print(f"[Main] Переключение: {self._current_mode_key} → {new_mode}")

        try:
            self._modes[self._current_mode_key].exit()
        except Exception as e:
            print(f"[Main] Ошибка при выходе из {self._current_mode_key}: {e}")

        self._current_mode_key = new_mode

        try:
            self._modes[new_mode].enter()
        except Exception as e:
            print(f"[Main] Ошибка при входе в {new_mode}: {e}")

        # Сбрасываем idle-таймер при входе в любой режим
        self._detector.reset_idle()

    # ------------------------------------------------------------------
    # FPS
    # ------------------------------------------------------------------

    def _update_fps(self):
        """Подсчёт FPS основного цикла (обновляется раз в секунду)."""
        self._fps_frames += 1
        now = time.time()
        dt  = now - self._fps_time
        if dt >= 1.0:
            self._fps_display = self._fps_frames / dt
            self._fps_frames  = 0
            self._fps_time    = now

    # ------------------------------------------------------------------
    # Завершение
    # ------------------------------------------------------------------

    def _shutdown(self):
        """Корректное завершение: закрываем окна, освобождаем ресурсы."""
        print("[Main] Завершение работы...")
        try:
            self._modes[self._current_mode_key].exit()
        except Exception:
            pass
        self._tracker.close()
        self._cam.release()
        cv2.destroyAllWindows()
        print("[Main] До свидания!")


# =============================================================================
# Точка входа
# =============================================================================

def main():
    try:
        app = DigitalCorridorApp()
        app.run()
    except KeyboardInterrupt:
        print("\n[Main] Прервано пользователем (Ctrl+C)")
    except SystemExit as e:
        print(f"[Main] Завершение: {e}")
    except Exception as e:
        print(f"[Main] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
