"""
game.py
=======
Self-contained Geometry Dash clone.

Human play
----------
    python game.py

Programmatic / headless API
----------------------------
    from game import Game

    g = Game(render=False)
    obs  = g.reset()           # returns dict of game state
    obs, reward, done = g.step(action=0)   # 0 = no-op, 1 = jump
    g.render()                 # call whenever you want a frame drawn
    g.close()

The obs dict contains everything needed for an RL agent or YOLO pipeline:
    {
      "player_y":     float,   # player top-left y (px)
      "player_vy":    float,   # vertical velocity
      "on_ground":    bool,
      "obstacles":    [        # list of upcoming obstacles, sorted by distance
          {
            "type":   str,     # "spike" | "block"
            "x":      float,   # left edge x in screen coords
            "y":      float,   # top edge y
            "w":      float,
            "h":      float,
          },
          ...
      ],
      "scroll_x":     float,   # total pixels scrolled (proxy for progress)
      "alive":        bool,
    }
"""

from __future__ import annotations

import math
import random
import sys
from typing import Optional

import pygame

import constants as C


# ─────────────────────────────────────────────────────────────────────────────
# Entities
# ─────────────────────────────────────────────────────────────────────────────

class Player:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.x   : float = float(C.PLAYER_X)
        self.y   : float = float(C.GROUND_Y - C.PLAYER_SIZE)
        self.vy  : float = 0.0
        self.on_ground: bool = True
        self.alive: bool = True
        self.angle: float = 0.0   # visual rotation

    def jump(self) -> None:
        if self.on_ground:
            self.vy = C.JUMP_VEL
            self.on_ground = False

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        self.vy += C.GRAVITY * dt
        self.y  += self.vy * dt

        ground_top = float(C.GROUND_Y - C.PLAYER_SIZE)
        if self.y >= ground_top:
            self.y        = ground_top
            self.vy       = 0.0
            self.on_ground = True
            self.angle    = 0.0
        else:
            self.on_ground = False
            self.angle = (self.angle - 200 * dt) % 360

    @property
    def hitbox(self) -> pygame.Rect:
        m = C.HITBOX_MARGIN
        return pygame.Rect(
            int(self.x) + m,
            int(self.y) + m,
            C.PLAYER_SIZE - 2 * m,
            C.PLAYER_SIZE - 2 * m,
        )

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive:
            return
        sq = pygame.Surface((C.PLAYER_SIZE, C.PLAYER_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(sq, C.PLAYER_COLOR, sq.get_rect(), border_radius=5)
        # cross detail
        pygame.draw.line(sq, (255,255,255,140), (5,5), (C.PLAYER_SIZE-5, C.PLAYER_SIZE-5), 2)
        pygame.draw.line(sq, (255,255,255,140), (C.PLAYER_SIZE-5,5), (5,C.PLAYER_SIZE-5), 2)
        rot = pygame.transform.rotate(sq, self.angle)
        cx = int(self.x) + C.PLAYER_SIZE // 2
        cy = int(self.y) + C.PLAYER_SIZE // 2
        surface.blit(rot, rot.get_rect(center=(cx, cy)))


class Spike:
    kind = "spike"

    def __init__(self, x: float) -> None:
        self.x = x
        self.w = C.SPIKE_W
        self.h = C.SPIKE_H

    def update(self, dx: float) -> None:
        self.x -= dx

    @property
    def offscreen(self) -> bool:
        return self.x + self.w < 0

    @property
    def hitbox(self) -> pygame.Rect:
        m = self.w // 4
        return pygame.Rect(int(self.x) + m, C.GROUND_Y - self.h, self.w - 2*m, self.h)

    def draw(self, surface: pygame.Surface) -> None:
        tip = (int(self.x) + self.w // 2, C.GROUND_Y - self.h)
        bl  = (int(self.x),          C.GROUND_Y)
        br  = (int(self.x) + self.w, C.GROUND_Y)
        pygame.draw.polygon(surface, C.SPIKE_COLOR, [bl, br, tip])
        pygame.draw.polygon(surface, (255, 130, 130), [bl, br, tip], 2)

    def as_dict(self) -> dict:
        return {"type": self.kind, "x": self.x,
                "y": float(C.GROUND_Y - self.h), "w": float(self.w), "h": float(self.h)}


class Block:
    kind = "block"

    def __init__(self, x: float) -> None:
        self.x = x
        self.w = C.BLOCK_W
        self.h = C.BLOCK_H

    def update(self, dx: float) -> None:
        self.x -= dx

    @property
    def offscreen(self) -> bool:
        return self.x + self.w < 0

    @property
    def hitbox(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), C.GROUND_Y - self.h, self.w, self.h)

    def draw(self, surface: pygame.Surface) -> None:
        r = self.hitbox
        pygame.draw.rect(surface, C.BLOCK_COLOR, r, border_radius=3)
        pygame.draw.rect(surface, (140, 255, 170), r, 2, border_radius=3)
        cx, cy = r.centerx, r.centery
        pygame.draw.line(surface, (80,180,100), (r.left, cy), (r.right, cy), 1)
        pygame.draw.line(surface, (80,180,100), (cx, r.top), (cx, r.bottom), 1)

    def as_dict(self) -> dict:
        return {"type": self.kind, "x": self.x,
                "y": float(C.GROUND_Y - self.h), "w": float(self.w), "h": float(self.h)}


# ─────────────────────────────────────────────────────────────────────────────
# Main Game class
# ─────────────────────────────────────────────────────────────────────────────

class Game:
    """
    Core game.  Works both rendered (human) and headless (agent/YOLO).

    Parameters
    ----------
    render : bool
        Open a pygame window.  Set False for training.
    seed : int | None
        RNG seed for reproducible obstacle sequences.
    """

    def __init__(self, render: bool = True, seed: Optional[int] = None) -> None:
        self._do_render = render
        self._seed      = seed
        self._rng       = random.Random(seed)

        self.player     = Player()
        self.obstacles  : list[Spike | Block] = []
        self._scroll_x  : float = 0.0
        self._next_spawn: float = C.SPAWN_X   # world-x of next obstacle spawn
        self._step_n    : int   = 0

        self.surface : Optional[pygame.Surface] = None
        self.clock   : Optional[pygame.time.Clock] = None
        self.font    : Optional[pygame.font.Font]  = None

        if render:
            self._init_display()

        self._spawn_initial()

    # ── Display ───────────────────────────────────────────────────────────────

    def _init_display(self) -> None:
        if not pygame.get_init():
            pygame.init()
        pygame.font.init()
        self.surface = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))
        pygame.display.set_caption(C.WINDOW_TITLE)
        self.clock   = pygame.time.Clock()
        self.font    = pygame.font.SysFont("monospace", 15, bold=True)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> dict:
        """Restart the level. Returns initial obs dict."""
        if seed is not None:
            self._rng = random.Random(seed)
        self.player.reset()
        self.obstacles.clear()
        self._scroll_x   = 0.0
        self._next_spawn = C.SPAWN_X
        self._step_n     = 0
        self._spawn_initial()
        return self._obs()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int, dt: float = 1.0 / C.FPS) -> tuple[dict, float, bool]:
        """
        Advance simulation one timestep.

        Parameters
        ----------
        action : int   0 = no-op, 1 = jump
        dt     : float seconds per step (default 1/60)

        Returns
        -------
        obs    : dict
        reward : float
        done   : bool
        """
        self._step_n += 1

        if action == 1:
            self.player.jump()

        self.player.update(dt)

        # Scroll world
        dx = C.GAME_SPEED * dt
        self._scroll_x += dx
        for obs in self.obstacles:
            obs.update(dx)

        # Cull off-screen
        self.obstacles = [o for o in self.obstacles if not o.offscreen]

        # Spawn new obstacles as needed
        self._maybe_spawn()

        # Collision
        dead = self._check_collision()

        reward = C.REWARD_DEATH if dead else C.REWARD_ALIVE
        done   = dead

        return self._obs(), reward, done

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self) -> None:
        """Draw current state. Safe to call even if render=False (no-ops)."""
        if self.surface is None:
            return
        self._draw_bg()
        self._draw_ground()
        for obs in self.obstacles:
            obs.draw(self.surface)
        self.player.draw(self.surface)
        self._draw_hud()
        pygame.display.flip()

    def tick(self) -> float:
        """Advance the clock; returns dt in seconds. Call once per frame."""
        if self.clock is None:
            return 1.0 / C.FPS
        ms = self.clock.tick(C.FPS)
        return min(ms / 1000.0, 1.0 / 30.0)

    def close(self) -> None:
        if pygame.get_init():
            pygame.quit()

    # ── Obs dict ──────────────────────────────────────────────────────────────

    def _obs(self) -> dict:
        upcoming = sorted(
            [o for o in self.obstacles if o.x + o.w >= C.PLAYER_X],
            key=lambda o: o.x,
        )
        return {
            "player_y":  self.player.y,
            "player_vy": self.player.vy,
            "on_ground": self.player.on_ground,
            "alive":     self.player.alive,
            "scroll_x":  self._scroll_x,
            "obstacles": [o.as_dict() for o in upcoming],
        }

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _spawn_initial(self) -> None:
        """Pre-populate a handful of obstacles so the screen isn't empty."""
        x = float(C.SCREEN_W + 100)
        for _ in range(6):
            gap = self._rng.randint(C.GAP_MIN, C.GAP_MAX)
            x  += gap
            self._add_obstacle(x)
        self._next_spawn = x + self._rng.randint(C.GAP_MIN, C.GAP_MAX)

    def _maybe_spawn(self) -> None:
        """Spawn a new obstacle once existing ones have scrolled far enough."""
        rightmost = max((o.x + o.w for o in self.obstacles), default=0)
        while rightmost < C.SCREEN_W + C.GAP_MAX:
            self._add_obstacle(self._next_spawn)
            self._next_spawn += self._rng.randint(C.GAP_MIN, C.GAP_MAX)
            rightmost = self._next_spawn

    def _add_obstacle(self, x: float) -> None:
        kind = self._rng.choice(["spike", "spike", "spike", "block"])  # spikes more common
        if kind == "spike":
            self.obstacles.append(Spike(x))
        else:
            self.obstacles.append(Block(x))

    # ── Collision ─────────────────────────────────────────────────────────────

    def _check_collision(self) -> bool:
        pr = self.player.hitbox
        for obs in self.obstacles:
            if pr.colliderect(obs.hitbox):
                self.player.alive = False
                return True
        return False

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_bg(self) -> None:
        self.surface.fill(C.BG_COLOR)
        for y in range(0, C.GROUND_Y, 44):
            pygame.draw.line(self.surface, C.GRID_COLOR, (0, y), (C.SCREEN_W, y), 1)

    def _draw_ground(self) -> None:
        pygame.draw.rect(
            self.surface, C.GROUND_COLOR,
            (0, C.GROUND_Y, C.SCREEN_W, C.SCREEN_H - C.GROUND_Y)
        )
        pygame.draw.line(self.surface, (75, 75, 100), (0, C.GROUND_Y), (C.SCREEN_W, C.GROUND_Y), 2)
        tile   = 36
        offset = int(self._scroll_x) % tile
        for x in range(-tile + offset, C.SCREEN_W, tile):
            pygame.draw.line(self.surface, (60,60,85), (x, C.GROUND_Y), (x, C.SCREEN_H), 1)

    def _draw_hud(self) -> None:
        if self.font is None:
            return
        lines = [
            f"dist : {int(self._scroll_x):>6} px",
            f"steps: {self._step_n:>6}",
        ]
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, C.HUD_COLOR)
            self.surface.blit(surf, (10, 10 + i * 20))


# ─────────────────────────────────────────────────────────────────────────────
# Human play entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    game = Game(render=True, seed=42)
    attempts = 0
    best     = 0

    print("SPACE / UP = jump   |   R = restart   |   Q = quit")

    running = True
    while running:
        dt = game.tick()

        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    action = 1
                elif event.key == pygame.K_r:
                    game.reset()
                    attempts += 1
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        # Also catch held key
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = 1

        obs, reward, done = game.step(action, dt)
        game.render()

        best = max(best, int(obs["scroll_x"]))

        if done:
            attempts += 1
            dist = int(obs["scroll_x"])
            print(f"Attempt {attempts} | dist={dist}px | best={best}px")
            pygame.time.wait(350)
            game.reset()

    game.close()


if __name__ == "__main__":
    main()