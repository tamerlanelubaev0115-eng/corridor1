# =============================================================================
# games/battle_game.py — Battle City клон (встроенный OpenCV)
# =============================================================================
"""
Battle City — упрощённый танковый шутер.

Управление:
  zone='up'    → движение вверх (W)
  zone='down'  → движение вниз  (S)
  zone='left'  → движение влево (A)
  zone='right' → движение вправо(D)
  open_palm    → выстрел

Механика:
  - Игрок управляет зелёным танком
  - Красные вражеские танки движутся и стреляют
  - Кирпичные стены (разрушаемые), стальные (нет)
  - Уничтожь всех врагов → следующий уровень
  - 3 жизни, game over → рестарт open_palm
"""

import numpy as np
import random
import math
import cv2
from games.base_game import BaseGame, VIRT_W, VIRT_H
from games.base_game import C_BG, C_WHITE, C_GREEN, C_RED, C_YELLOW, C_GRAY, C_ORANGE


# Размер клетки тайловой сетки
CELL  = 32
COLS  = VIRT_W // CELL   # 20
ROWS  = VIRT_H // CELL   # ~11

# Тайлы
EMPTY  = 0
BRICK  = 1
STEEL  = 2
BASE   = 3


def _make_map():
    """Генерирует карту уровня."""
    grid = [[EMPTY]*COLS for _ in range(ROWS)]
    # Стальные стены по периметру
    for c in range(COLS):
        grid[0][c] = STEEL
        grid[ROWS-1][c] = STEEL
    for r in range(ROWS):
        grid[r][0] = STEEL
        grid[r][COLS-1] = STEEL
    # База (игрок должен защищать)
    base_r = ROWS - 2; base_c = COLS // 2
    grid[base_r][base_c] = BASE
    # Случайные кирпичные стены
    for _ in range(50):
        r = random.randint(1, ROWS-2)
        c = random.randint(1, COLS-2)
        if grid[r][c] == EMPTY:
            grid[r][c] = BRICK
    return grid


class Tank:
    SIZE   = 24   # px (виртуальных)
    SPEED  = 80.0

    def __init__(self, x, y, color, is_player=False):
        self.x     = float(x)
        self.y     = float(y)
        self.dir   = 0   # 0=up,1=right,2=down,3=left
        self.color = color
        self.is_player = is_player
        self.alive = True
        self.fire_cd = 0.0   # cooldown
        self.ai_timer= 0.0
        self.ai_dir  = random.randint(0, 3)

    def rect(self):
        hs = self.SIZE // 2
        return (int(self.x - hs), int(self.y - hs),
                int(self.x + hs), int(self.y + hs))

    def move(self, dx, dy, grid, dt):
        nx = self.x + dx * self.SPEED * dt
        ny = self.y + dy * self.SPEED * dt
        hs = self.SIZE // 2 + 1
        # Проверяем столкновение с тайлами
        for px, py in [(nx-hs, ny-hs),(nx+hs, ny-hs),(nx-hs, ny+hs),(nx+hs, ny+hs)]:
            c = int(px) // CELL; r = int(py) // CELL
            if 0 <= r < ROWS and 0 <= c < COLS:
                if grid[r][c] in (BRICK, STEEL, BASE):
                    return False
            else:
                return False
        self.x = min(max(nx, hs), VIRT_W - hs)
        self.y = min(max(ny, hs), VIRT_H - hs)
        return True


class Bullet:
    SPEED = 260.0
    SIZE  = 5

    def __init__(self, x, y, direction):
        self.x   = float(x)
        self.y   = float(y)
        self.dir = direction   # 0=up,1=right,2=down,3=left
        self.alive = True

    def update(self, dt, grid, tanks: list):
        dx, dy = [(0,-1),(1,0),(0,1),(-1,0)][self.dir]
        self.x += dx * self.SPEED * dt
        self.y += dy * self.SPEED * dt

        # Выход за границы
        if not (0 < self.x < VIRT_W and 0 < self.y < VIRT_H):
            self.alive = False
            return

        # Попадание в тайл
        c = int(self.x) // CELL; r = int(self.y) // CELL
        if 0 <= r < ROWS and 0 <= c < COLS:
            if grid[r][c] == BRICK:
                grid[r][c] = EMPTY
                self.alive = False
                return
            elif grid[r][c] == STEEL:
                self.alive = False
                return

        # Попадание в танк
        for t in tanks:
            if t.alive and abs(t.x - self.x) < t.SIZE//2 + 4 and abs(t.y - self.y) < t.SIZE//2 + 4:
                t.alive = False
                self.alive = False
                return


class BattleGame(BaseGame):
    ENEMY_COUNT = 5

    def __init__(self):
        super().__init__()
        self._grid    = None
        self._player  = None
        self._enemies : list[Tank]  = []
        self._bullets : list[Bullet] = []
        self._level   = 1
        self._fire_pressed = False
        self.reset()

    def reset(self):
        self.score      = 0
        self.lives      = 3
        self.game_over  = False
        self._level     = 1
        self._start_level()

    def _start_level(self):
        self._grid = _make_map()
        cx   = VIRT_W // 2
        self._player = Tank(cx, VIRT_H - CELL*1.5, C_GREEN, is_player=True)
        self._player.dir = 0
        self._enemies = []
        for i in range(self.ENEMY_COUNT + self._level - 1):
            ex = random.choice([CELL*2, VIRT_W//2, VIRT_W-CELL*2])
            ey = CELL * 2
            self._enemies.append(Tank(ex, ey, C_RED))
        self._bullets = []
        self._fire_pressed = False

    # ------------------------------------------------------------------
    def update(self, dt, zone, is_open_palm):
        if self.game_over:
            if is_open_palm:
                self.high_score = max(self.high_score, self.score)
                self.reset()
            return

        p = self._player

        # --- Игрок ---
        dir_map = {"up":(0,-1,0),"down":(0,1,2),"left":(-1,0,3),"right":(1,0,1)}
        dx = dy = 0
        if zone in dir_map:
            dx, dy, d = dir_map[zone]
            p.dir = d
            p.move(dx, dy, self._grid, dt)

        # Выстрел игрока
        p.fire_cd = max(0, p.fire_cd - dt)
        if is_open_palm and not self._fire_pressed and p.fire_cd <= 0:
            offsets = [(0,-p.SIZE),(p.SIZE,0),(0,p.SIZE),(-p.SIZE,0)]
            ox, oy  = offsets[p.dir]
            self._bullets.append(Bullet(p.x+ox, p.y+oy, p.dir))
            p.fire_cd = 0.4
        self._fire_pressed = is_open_palm

        # --- ИИ врагов ---
        for e in self._enemies:
            if not e.alive:
                continue
            e.fire_cd  = max(0, e.fire_cd - dt)
            e.ai_timer = max(0, e.ai_timer - dt)

            # Сменить направление случайно
            if e.ai_timer <= 0:
                e.ai_dir   = random.randint(0, 3)
                e.ai_timer = random.uniform(0.5, 1.5)

            ddx, ddy = [(0,-1),(1,0),(0,1),(-1,0)][e.ai_dir]
            if not e.move(ddx, ddy, self._grid, dt):
                e.ai_timer = 0  # Упёрся — поменять направление

            # Стрелять в игрока
            if e.fire_cd <= 0 and random.random() < 0.02:
                self._bullets.append(Bullet(e.x, e.y, e.ai_dir))
                e.fire_cd = 1.2

        # --- Пули ---
        all_tanks = [p] + self._enemies
        for b in self._bullets:
            b.update(dt, self._grid, all_tanks)
        self._bullets = [b for b in self._bullets if b.alive]

        # --- Проверка смерти игрока ---
        if not p.alive:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else:
                p.x = VIRT_W // 2
                p.y = VIRT_H - CELL * 1.5
                p.alive = True

        # --- Убитые враги ---
        self._enemies = [e for e in self._enemies]  # keep all for tracking
        alive_enemies = [e for e in self._enemies if e.alive]
        killed = [e for e in self._enemies if not e.alive]
        self.score += len(killed) * 100
        self._enemies = alive_enemies

        if not self._enemies:
            self._level += 1
            self.score  += 500
            self._start_level()

    # ------------------------------------------------------------------
    def render(self, canvas):
        self.fill_bg(canvas)

        # Тайлы
        for r in range(ROWS):
            for c in range(COLS):
                cell = self._grid[r][c]
                x1  = c * CELL; y1 = r * CELL
                x2  = x1 + CELL; y2 = y1 + CELL
                if cell == BRICK:
                    cv2.rectangle(canvas, (x1,y1),(x2,y2),(40,80,160),-1)
                    cv2.rectangle(canvas, (x1,y1),(x2,y2),(30,60,120),1)
                    # Кирпичная текстура
                    cv2.line(canvas,(x1,y1+CELL//2),(x2,y1+CELL//2),(30,60,120),1)
                    cv2.line(canvas,(x1+CELL//2,y1),(x1+CELL//2,y1+CELL//2),(30,60,120),1)
                    cv2.line(canvas,(x1,y1+CELL//2),(x1,y2),(30,60,120),1)
                elif cell == STEEL:
                    cv2.rectangle(canvas, (x1,y1),(x2,y2),(80,80,80),-1)
                    cv2.rectangle(canvas, (x1,y1),(x2,y2),(100,100,100),1)
                elif cell == BASE:
                    cv2.rectangle(canvas, (x1,y1),(x2,y2),(0,200,200),-1)
                    self.draw_text_center(canvas,"BASE",x1+CELL//2,y1+CELL//2,0.3,C_BG)

        # Пули
        for b in self._bullets:
            cv2.circle(canvas,(int(b.x),int(b.y)),4,C_YELLOW,-1)

        # Танки
        def draw_tank(t):
            if not t.alive: return
            hs = t.SIZE // 2
            x1,y1,x2,y2 = t.rect()
            cv2.rectangle(canvas,(x1,y1),(x2,y2),t.color,-1)
            cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,0,0),1)
            # Дуло
            mx,my = int(t.x),int(t.y)
            dl = t.SIZE//2 + 8
            brl = [(mx,my-dl),(mx+dl,my),(mx,my+dl),(mx-dl,my)]
            cv2.line(canvas,(mx,my),brl[t.dir],C_WHITE,3)
            # Полоска башни
            cv2.rectangle(canvas,(mx-6,my-6),(mx+6,my+6),
                          (min(255,t.color[0]+40),min(255,t.color[1]+40),min(255,t.color[2]+40)),-1)

        draw_tank(self._player)
        for e in self._enemies: draw_tank(e)

        # HUD
        self.draw_score(canvas)
        self.draw_text(canvas, f"Lives: {'*'*self.lives}", VIRT_W//2-40, 24, 0.6, C_WHITE)
        self.draw_text(canvas, f"Lvl:{self._level}  Enemies:{len(self._enemies)}",
                       VIRT_W-180, 24, 0.55, C_YELLOW)

        if self.game_over:
            self.draw_game_over(canvas)
