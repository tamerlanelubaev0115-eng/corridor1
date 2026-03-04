# =============================================================================
# games/dino_game.py — Chrome Dino клон (встроенный, OpenCV)
# =============================================================================
"""
Dino Runner — бесконечный раннер.

Управление:
  zone='up' ИЛИ open_palm → прыжок
  zone='down'              → присесть (duck)

Механика:
  - Динозавр прыгает/приседает, уклоняясь от кактусов и птеродактилей
  - Скорость растёт со временем
  - Счёт = время выживания
  - Game Over → collision → показать экран; open_palm → рестарт
"""

import numpy as np
import random
import math
from games.base_game import BaseGame, VIRT_W, VIRT_H
from games.base_game import C_BG, C_WHITE, C_GREEN, C_GRAY, C_YELLOW, C_RED, C_ORANGE


# Константы
GND_Y      = VIRT_H - 60    # Y линии земли
DINO_X     = 110            # X позиция динозавра (фиксированная)
GRAVITY    = 1800           # px/s²
JUMP_VEL   = -660           # начальная вертикальная скорость при прыжке
DUCK_H     = 22             # Высота приседания
STAND_H    = 44             # Высота стоя
DINO_W     = 36             # Ширина динозавра

class Obstacle:
    def __init__(self, x, kind="cactus"):
        self.x    = x
        self.kind = kind   # "cactus" | "bird"
        if kind == "cactus":
            self.w = random.choice([24, 32, 44])
            self.h = random.choice([40, 52, 64])
            self.y = GND_Y - self.h
        else:
            self.w = 38
            self.h = 20
            self.y = GND_Y - random.choice([24, 56, 90])  # разные высоты птицы

    def move(self, speed, dt):
        self.x -= speed * dt

    def rect(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class DinoGame(BaseGame):
    """«Динозавр» — встроенный в OpenCV."""

    def __init__(self):
        super().__init__()
        self._clouds: list = []
        self.reset()

    def reset(self):
        self.score      = 0
        self.game_over  = False
        self._speed     = 280        # начальная скорость (px/s)
        self._dino_y    = float(GND_Y - STAND_H)
        self._dino_vy   = 0.0
        self._on_ground = True
        self._ducking   = False
        self._obstacles: list[Obstacle] = []
        self._next_obs  = random.uniform(0.8, 1.5)  # s до следующего препятствия
        self._time_acc  = 0.0
        self._score_acc = 0.0
        self._jump_held = False
        self._clouds    = [
            [random.uniform(0, VIRT_W), random.uniform(20, 100), random.uniform(0.5, 1.5)]
            for _ in range(5)
        ]

    # ------------------------------------------------------------------
    def update(self, dt: float, zone: str, is_open_palm: bool):
        if self.game_over:
            if is_open_palm:
                self.high_score = max(self.high_score, self.score)
                self.reset()
            return

        want_jump = (zone == "up" or is_open_palm)
        want_duck = (zone == "down")

        # --- Физика ---
        self._ducking = want_duck and self._on_ground

        if want_jump and self._on_ground and not self._ducking:
            self._dino_vy   = JUMP_VEL
            self._on_ground = False

        self._dino_y  += self._dino_vy * dt
        self._dino_vy += GRAVITY * dt

        dino_h = DUCK_H if self._ducking else STAND_H
        if self._dino_y >= GND_Y - dino_h:
            self._dino_y    = GND_Y - dino_h
            self._dino_vy   = 0.0
            self._on_ground = True

        # --- Скорость нарастает ---
        self._speed = min(280 + self._time_acc * 60, 900)
        self._time_acc += dt

        # --- Счёт ---
        self._score_acc += dt
        self.score = int(self._score_acc * 10)

        # --- Спавн препятствий ---
        self._next_obs -= dt
        if self._next_obs <= 0:
            kind = "bird" if random.random() < 0.25 and self._time_acc > 5 else "cactus"
            self._obstacles.append(Obstacle(VIRT_W + 20, kind))
            gap = random.uniform(1.0, 2.2) * (280 / self._speed)
            self._next_obs = gap

        # --- Перемещение / удаление ---
        for obs in self._obstacles:
            obs.move(self._speed, dt)
        self._obstacles = [o for o in self._obstacles if o.x + o.w > -10]

        # --- Коллизии ---
        dino_rect = (DINO_X + 4, int(self._dino_y) + 4,
                     DINO_X + DINO_W - 4, int(self._dino_y) + dino_h - 4)
        for obs in self._obstacles:
            ox1, oy1, ox2, oy2 = obs.rect()
            dx1, dy1, dx2, dy2 = dino_rect
            if dx1 < ox2 and dx2 > ox1 and dy1 < oy2 and dy2 > oy1:
                self.game_over = True
                self.high_score = max(self.high_score, self.score)
                break

        # --- Облака ---
        for c in self._clouds:
            c[0] -= 40 * dt
        for c in self._clouds:
            if c[0] + 80 < 0:
                c[0] = VIRT_W + 20
                c[1] = random.uniform(20, 100)
                c[2] = random.uniform(0.5, 1.5)

    # ------------------------------------------------------------------
    def render(self, canvas: np.ndarray):
        import cv2
        self.fill_bg(canvas)

        # Облака
        for cx, cy, sz in self._clouds:
            r = int(12 * sz)
            cv2.ellipse(canvas, (int(cx), int(cy)), (int(r*2.5), r), 0, 0, 360, (40,40,50), -1)
            cv2.ellipse(canvas, (int(cx+r), int(cy)-r//2), (int(r*1.5), r), 0, 0, 360, (40,40,50), -1)

        # Земля
        import cv2
        cv2.line(canvas, (0, GND_Y), (VIRT_W, GND_Y), (90, 90, 90), 2)
        # Пунктир земли
        for x in range(0, VIRT_W, 20):
            cv2.line(canvas, (x, GND_Y+5), (x+10, GND_Y+5), (50,50,60), 1)

        # Динозавр
        dino_h = DUCK_H if self._ducking else STAND_H
        dy     = int(self._dino_y)
        color  = C_GREEN if not self.game_over else C_RED
        # Тело
        cv2.rectangle(canvas, (DINO_X, dy), (DINO_X+DINO_W, dy+dino_h), color, -1)
        # Голова (только если не пригнулся)
        if not self._ducking:
            cv2.rectangle(canvas, (DINO_X+10, dy-18), (DINO_X+DINO_W+6, dy+4), color, -1)
            # Глаз
            cv2.circle(canvas, (DINO_X+DINO_W+2, dy-10), 4, C_WHITE, -1)
            cv2.circle(canvas, (DINO_X+DINO_W+3, dy-10), 2, C_BG,    -1)
            # Рот
            cv2.line(canvas, (DINO_X+DINO_W, dy-2), (DINO_X+DINO_W+8, dy-2), color, 2)
        # Ноги (анимация)
        leg_anim = int(self._time_acc * 12) % 2 if self._on_ground else 0
        lx = DINO_X + 6 + leg_anim * 14
        cv2.rectangle(canvas, (lx, dy+dino_h), (lx+10, dy+dino_h+10), color, -1)
        cv2.rectangle(canvas, (DINO_X+DINO_W-16+leg_anim*(-14), dy+dino_h),
                              (DINO_X+DINO_W-6+leg_anim*(-14), dy+dino_h+10), color, -1)

        # Препятствия
        for obs in self._obstacles:
            ox = int(obs.x)
            if obs.kind == "cactus":
                cv2.rectangle(canvas, (ox, int(obs.y)), (ox+obs.w, GND_Y), C_GREEN, -1)
                # Рожки кактуса
                cv2.rectangle(canvas, (ox-8, int(obs.y)+obs.h//4), (ox, int(obs.y)+obs.h//3), C_GREEN, -1)
                cv2.rectangle(canvas, (ox+obs.w, int(obs.y)+obs.h//3), (ox+obs.w+8, int(obs.y)+obs.h//2), C_GREEN, -1)
            else:
                # Ptero
                oy = int(obs.y)
                cv2.ellipse(canvas, (ox+19, oy+10), (16, 10), 0, 0, 360, C_ORANGE, -1)
                wing_y = oy + 2 + int(4 * math.sin(self._time_acc * 10))
                cv2.line(canvas, (ox, wing_y), (ox+12, oy+10), C_ORANGE, 3)
                cv2.line(canvas, (ox+26, oy+10), (ox+38, wing_y), C_ORANGE, 3)

        # Счёт
        self.draw_score(canvas)

        # Подсказка управления (первые 3 секунды)
        if self._time_acc < 3.0:
            self.draw_text_center(canvas, "Ruka vyshe = JUMP  |  Nizhe = DUCK",
                                  VIRT_W//2, VIRT_H-20, 0.5, (80,80,80))

        if self.game_over:
            self.draw_game_over(canvas)
