# =============================================================================
# games/mario_game.py — Super Mario клон (встроенный OpenCV)
# =============================================================================
"""
Mario Platformer — упрощённый платформер.

Управление:
  zone='left'  → бег влево  (A)
  zone='right' → бег вправо (D)
  zone='up' / open_palm → прыжок (Space)

Механика:
  - Гравитация, платформы, марио
  - Враги (Гумбы) — ходят туда/обратно, убиваются прыжком сверху
  - Монеты — собирать, +100 очков
  - Флаг — достичь правого края → следующий уровень
  - 3 жизни
"""

import numpy as np
import cv2
import random
import math
from games.base_game import BaseGame, VIRT_W, VIRT_H
from games.base_game import C_BG, C_WHITE, C_GREEN, C_RED, C_YELLOW, C_ORANGE, C_GRAY


GRAVITY     = 1400.0     # px/s²
JUMP_VEL    = -540.0
WALK_SPEED  = 160.0
WORLD_W     = VIRT_W * 3   # Мир шире экрана — скроллинг

# Цвета Марио-стиля
C_MARIO    = (40,  80,  200)   # синеватый (замена красному — для BGR)
C_GROUND   = (30,  90,  160)
C_PLATFORM = (40,  130, 200)
C_ENEMY    = (20,  40,  180)
C_COIN     = (0,   210, 255)
C_SKY      = (160, 90,  10)    # BGR тёмно-синее небо


class Platform:
    def __init__(self, x, y, w, h=16, color=None):
        self.x = x; self.y = y
        self.w = w; self.h = h
        self.color = color or C_PLATFORM


class Enemy:
    SPEED = 60.0
    W = 24; H = 24

    def __init__(self, x, y):
        self.x   = float(x)
        self.y   = float(y)
        self.vx  = self.SPEED
        self.vy  = 0.0
        self.alive = True
        self.squish_timer = 0.0   # Анимация смерти

    def rect(self):
        return (self.x, self.y, self.x+self.W, self.y+self.H)


class Coin:
    R = 7
    def __init__(self, x, y):
        self.x = x; self.y = y
        self.collected = False
        self.anim = 0.0


class MarioGame(BaseGame):

    def __init__(self):
        super().__init__()
        self._level = 1
        self.reset()

    def reset(self):
        self.score     = 0
        self.lives     = 3
        self.game_over = False
        self._level    = 1
        self._load_level()

    def _load_level(self):
        self._cam_x  = 0.0   # Камера
        self._mario_x = 60.0
        self._mario_y = float(VIRT_H - 100)
        self._mario_vx = 0.0
        self._mario_vy = 0.0
        self._on_ground = False
        self._facing    = 1   # +1=right, -1=left
        self._walk_anim = 0.0

        # Платформы (x, y, w)
        gnd_h = 24
        self._platforms = [
            Platform(0, VIRT_H - gnd_h, WORLD_W, gnd_h, C_GROUND),   # Земля
            Platform(300, VIRT_H-130, 120),
            Platform(500, VIRT_H-190, 140),
            Platform(700, VIRT_H-140, 100),
            Platform(900, VIRT_H-200, 160),
            Platform(1100,VIRT_H-160, 120),
            Platform(1300,VIRT_H-220, 100),
            Platform(1500,VIRT_H-150, 140),
            Platform(1700,VIRT_H-200, 120),
        ]

        # Враги
        self._enemies = [
            Enemy(400, VIRT_H - gnd_h - Enemy.H),
            Enemy(800, VIRT_H - gnd_h - Enemy.H),
            Enemy(1200,VIRT_H - gnd_h - Enemy.H),
            Enemy(1600,VIRT_H - gnd_h - Enemy.H),
        ]

        # Монеты
        self._coins = []
        for px in range(200, WORLD_W - 200, 250):
            self._coins.append(Coin(px, VIRT_H - 80))

        self._flag_x  = WORLD_W - 80
        self._won     = False
        self._dead    = False
        self._dead_timer = 0.0
        self._jump_held  = False

    # ------------------------------------------------------------------
    def update(self, dt, zone, is_open_palm):
        if self.game_over:
            if is_open_palm:
                self.high_score = max(self.high_score, self.score)
                self.reset()
            return

        if self._won:
            self._level += 1
            self.score  += 1000
            self._load_level()
            return

        if self._dead:
            self._dead_timer -= dt
            if self._dead_timer <= 0:
                self.lives -= 1
                if self.lives <= 0:
                    self.game_over = True
                else:
                    self._load_level()
            return

        want_left  = zone == "left"
        want_right = zone == "right"
        want_jump  = zone == "up" or is_open_palm

        # Движение
        self._mario_vx = 0
        if want_left:
            self._mario_vx = -WALK_SPEED
            self._facing   = -1
            self._walk_anim += dt * 8
        elif want_right:
            self._mario_vx = WALK_SPEED
            self._facing   = 1
            self._walk_anim += dt * 8
        else:
            self._walk_anim = 0

        # Прыжок
        if want_jump and self._on_ground and not self._jump_held:
            self._mario_vy = JUMP_VEL
            self._on_ground = False
        self._jump_held = want_jump

        # Физика
        self._mario_vy += GRAVITY * dt
        nx = self._mario_x + self._mario_vx * dt
        ny = self._mario_y + self._mario_vy * dt

        self._on_ground = False
        MW = 26; MH = 36

        # Коллизия с платформами
        for plat in self._platforms:
            px1, py1 = plat.x, plat.y
            px2, py2 = plat.x + plat.w, plat.y + plat.h

            # Горизонтальный клип
            if (nx < px2 and nx+MW > px1 and
                    self._mario_y+MH > py1 and self._mario_y < py2):
                if self._mario_vx > 0:
                    nx = px1 - MW
                elif self._mario_vx < 0:
                    nx = px2

            # Вертикальный клип
            if (nx < px2 and nx+MW > px1 and
                    ny+MH > py1 and ny < py2):
                if self._mario_vy > 0:   # Падение
                    ny = py1 - MH
                    self._mario_vy = 0
                    self._on_ground = True
                elif self._mario_vy < 0:  # Удар о потолок
                    ny = py2
                    self._mario_vy = 0

        # Границы мира
        nx = max(0, min(nx, WORLD_W - MW))
        if ny > VIRT_H + 100:
            self._dead = True
            self._dead_timer = 1.5

        self._mario_x = nx
        self._mario_y = ny

        # Камера следует за Марио
        target_cam = self._mario_x - VIRT_W // 3
        self._cam_x += (target_cam - self._cam_x) * 8 * dt
        self._cam_x  = max(0, min(self._cam_x, WORLD_W - VIRT_W))

        # Враги
        for e in self._enemies:
            if not e.alive:
                e.squish_timer = max(0, e.squish_timer - dt)
                continue
            e.vy += GRAVITY * dt
            e.y  += e.vy * dt
            e.x  += e.vx * dt

            # Платформы (только Y)
            for plat in self._platforms:
                px1,py1 = plat.x,plat.y
                px2,py2 = plat.x+plat.w, plat.y+plat.h
                if (e.x < px2 and e.x+e.W > px1 and
                        e.y+e.H > py1 and e.y+e.H < py2+20):
                    e.y  = py1 - e.H
                    e.vy = 0

            # Разворот у стен/края
            if e.x <= 0 or e.x + e.W >= WORLD_W:
                e.vx = -e.vx

            # Коллизия с Марио
            if (self._mario_x < e.x+e.W and self._mario_x+MW > e.x and
                    self._mario_y < e.y+e.H and self._mario_y+MH > e.y):
                if self._mario_vy > 0 and self._mario_y+MH < e.y + e.H//2 + 10:
                    # Прыжок сверху — убить
                    e.alive = False
                    e.squish_timer = 0.5
                    self._mario_vy = JUMP_VEL * 0.5
                    self.score += 200
                else:
                    self._dead = True
                    self._dead_timer = 1.5

        # Монеты
        for coin in self._coins:
            if not coin.collected:
                coin.anim += dt * 4
                if (abs(self._mario_x + MW//2 - coin.x) < 20 and
                        abs(self._mario_y + MH//2 - coin.y) < 20):
                    coin.collected = True
                    self.score    += 100

        # Флаг
        if self._mario_x + MW >= self._flag_x:
            self._won = True

        # Очки за время
        self.score += int(dt * 2)

    # ------------------------------------------------------------------
    def render(self, canvas):
        # Небо с градиентом
        for y in range(VIRT_H):
            t   = y / VIRT_H
            r   = int(10 + t*30); g = int(10+t*50); b = int(60+t*80)
            canvas[y, :] = (b, g, r)

        cam = int(self._cam_x)
        MW = 26; MH = 36

        # Параллакс-облака
        for i in range(6):
            cx = int((i * 280 - cam * 0.3) % (VIRT_W + 100) - 50)
            cy = 40 + i % 3 * 20
            cv2.ellipse(canvas,(cx,cy),(55,18),0,0,360,(70,70,100),-1)
            cv2.ellipse(canvas,(cx+30,cy-10),(35,14),0,0,360,(70,70,100),-1)

        # Платформы
        for plat in self._platforms:
            x1 = int(plat.x - cam); y1 = int(plat.y)
            x2 = x1 + plat.w; y2 = y1 + plat.h
            if x2 < -10 or x1 > VIRT_W + 10: continue
            cv2.rectangle(canvas,(x1,y1),(x2,y2),plat.color,-1)
            # Верхняя полоска
            lighter = tuple(min(255,c+40) for c in plat.color)
            cv2.line(canvas,(x1,y1),(x2,y1),lighter,3)

        # Монеты
        for coin in self._coins:
            if coin.collected: continue
            cx = int(coin.x - cam)
            cy = int(coin.y + math.sin(coin.anim)*4)
            if -20 < cx < VIRT_W+20:
                cv2.circle(canvas,(cx,cy),Coin.R,C_COIN,-1)
                cv2.circle(canvas,(cx-2,cy-2),2,(255,255,255),-1)

        # Враги
        for e in self._enemies:
            ex = int(e.x - cam)
            ey = int(e.y)
            if -30 < ex < VIRT_W+30:
                h = e.H if e.alive else max(4, int(e.H * e.squish_timer * 2))
                cv2.rectangle(canvas,(ex,ey+e.H-h),(ex+e.W,ey+e.H),C_ENEMY,-1)
                if e.alive:
                    # Глаза
                    cv2.circle(canvas,(ex+6, ey+8),4,C_WHITE,-1)
                    cv2.circle(canvas,(ex+18,ey+8),4,C_WHITE,-1)
                    cv2.circle(canvas,(ex+7, ey+8),2,(0,0,0),-1)
                    cv2.circle(canvas,(ex+19,ey+8),2,(0,0,0),-1)

        # Флаг
        fx = int(self._flag_x - cam)
        if -10 < fx < VIRT_W+10:
            cv2.line(canvas,(fx,VIRT_H-24),(fx,VIRT_H-120),C_WHITE,3)
            pts = np.array([(fx,VIRT_H-120),(fx+30,VIRT_H-105),(fx,VIRT_H-90)])
            cv2.fillPoly(canvas,[pts],C_GREEN)

        # Марио
        if not self._dead:
            mx = int(self._mario_x - cam)
            my = int(self._mario_y)
            # Тело
            cv2.rectangle(canvas,(mx,my+MH//2),(mx+MW,my+MH),C_MARIO,-1)
            # Голова
            cv2.circle(canvas,(mx+MW//2, my+12), 14, C_ORANGE,-1)
            # Кепка
            cv2.ellipse(canvas,(mx+MW//2,my+6),(14,6),0,180,360,C_MARIO,-1)
            cv2.rectangle(canvas,(mx-2,my+4),(mx+MW+2,my+10),C_MARIO,-1)
            # Глаза
            cv2.circle(canvas,(mx+MW//2-4*self._facing,my+10),3,C_WHITE,-1)
            # Усы
            cv2.line(canvas,(mx+MW//2-8,my+16),(mx+MW//2-2,my+14),C_BG,2)
            cv2.line(canvas,(mx+MW//2,my+14),(mx+MW//2+8,my+16),C_BG,2)

        # HUD
        self.draw_score(canvas)
        self.draw_text(canvas, f"Lives: {'*'*self.lives}", VIRT_W//2-40, 24, 0.6, C_WHITE)
        self.draw_text(canvas, f"Level:{self._level}", VIRT_W-100, 24, 0.6, C_YELLOW)

        if self._won:
            self.draw_text_center(canvas,"LEVEL CLEAR!",VIRT_W//2,VIRT_H//2-20,1.2,C_YELLOW,2)

        if self.game_over:
            self.draw_game_over(canvas)
