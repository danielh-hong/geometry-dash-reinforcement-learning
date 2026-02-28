# =============================================================================
# game.py
# =============================================================================
# Self-contained Geometry Dash clone with accurate physics.
#
# ── HOW TO PLAY (human mode) ─────────────────────────────────────────────────
#
#   1. Make sure your venv is active:
#        source venv/bin/activate        (Mac/Linux)
#        venv\Scripts\activate           (Windows)
#
#   2. Run:
#        python game.py                  # normal play
#        python game.py --debug          # shows hitboxes in yellow
#        python game.py --seed 123       # fixed obstacle layout (reproducible)
#
#   Controls:
#        SPACE or UP ARROW  →  jump
#        R                  →  restart immediately
#        H                  →  toggle hitbox debug overlay
#        Q or ESC           →  quit
#
# ── HOW TO USE PROGRAMMATICALLY (for RL / YOLO) ──────────────────────────────
#
#   from game import Game
#
#   g = Game(render=False, seed=42)   # headless — no window, runs fast
#   obs = g.reset()                   # returns a dict of game state
#
#   while True:
#       action = 1 if your_agent_decides_to_jump(obs) else 0
#       obs, reward, done = g.step(action)
#       if done:
#           obs = g.reset()
#
# ── OBS DICT ─────────────────────────────────────────────────────────────────
#
#   {
#     "player_y"   : float   top-left y of the player cube (px)
#     "player_vy"  : float   vertical velocity — negative = moving up (px/s)
#     "on_ground"  : bool    True if cube is resting on the ground
#     "alive"      : bool    False after a collision
#     "scroll_x"   : float   total pixels scrolled (grows as level progresses)
#     "step"       : int     number of steps taken this episode
#     "obstacles"  : list    upcoming obstacles sorted by distance, each is:
#                            {
#                              "type" : "spike" | "block"
#                              "x"    : float  left edge x in screen coords (px)
#                              "y"    : float  top  edge y in screen coords (px)
#                              "w"    : float  width  (px)
#                              "h"    : float  height (px)
#                            }
#   }
#
# ── STEP() RETURN ─────────────────────────────────────────────────────────────
#
#   obs    : dict   (described above)
#   reward : float  REWARD_ALIVE each step, REWARD_DEATH on collision
#   done   : bool   True when the player has died
#
# =============================================================================

from __future__ import annotations

import argparse
import math
import random
import sys
from typing import Optional

import pygame

import constants as C


# =============================================================================
# Player
# =============================================================================

class Player:
    """
    The player-controlled cube.

    Attributes
    ----------
    x, y       : float   top-left pixel position on screen
    vy         : float   vertical velocity in px/s (negative = upward)
    on_ground  : bool    True when sitting on the ground
    alive      : bool    Set to False on collision
    angle      : float   Visual rotation angle in degrees (cosmetic only)
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Put the player back at the start position."""
        self.x        : float = float(C.PLAYER_X)
        self.y        : float = float(C.GROUND_Y - C.PLAYER_SIZE)
        self.vy       : float = 0.0
        self.on_ground: bool  = True
        self.alive    : bool  = True
        self.angle    : float = 0.0   # degrees, purely visual

    # ── Actions ───────────────────────────────────────────────────────────────

    def jump(self) -> None:
        """
        Jump if currently on the ground.
        Ignored while airborne (no double-jump in basic cube mode).
        """
        if self.on_ground:
            self.vy = C.JUMP_VEL      # negative = upward in pygame coords
            self.on_ground = False

    # ── Physics update ────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """
        Advance player physics by dt seconds.
        Called once per frame from Game.step().
        """
        if not self.alive:
            return

        # Apply gravity: accelerate downward each frame
        self.vy += C.GRAVITY * dt

        # Move vertically
        self.y += self.vy * dt

        # Ground clamp — stop falling through the floor
        ground_top = float(C.GROUND_Y - C.PLAYER_SIZE)
        if self.y >= ground_top:
            self.y         = ground_top
            self.vy        = 0.0
            self.on_ground = True
            self.angle     = 0.0      # snap rotation back when landing
        else:
            self.on_ground = False
            # Spin while airborne — faster when moving upward, slower falling
            # (purely cosmetic, doesn't affect gameplay)
            self.angle = (self.angle - 220 * dt) % 360

    # ── Hitbox ────────────────────────────────────────────────────────────────

    @property
    def hitbox(self) -> pygame.Rect:
        """
        The actual collision rectangle used for death detection.
        Slightly smaller than the visual cube (HITBOX_MARGIN px inward on all sides).
        This matches real GD where the kill hitbox is a bit smaller than the icon.
        """
        m = C.HITBOX_MARGIN
        return pygame.Rect(
            int(self.x) + m,
            int(self.y) + m,
            C.PLAYER_SIZE - 2 * m,
            C.PLAYER_SIZE - 2 * m,
        )

    @property
    def visual_rect(self) -> pygame.Rect:
        """Full visual bounding box (used for rendering, not collision)."""
        return pygame.Rect(int(self.x), int(self.y), C.PLAYER_SIZE, C.PLAYER_SIZE)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface, debug: bool = False) -> None:
        if not self.alive:
            return

        cx = int(self.x) + C.PLAYER_SIZE // 2
        cy = int(self.y) + C.PLAYER_SIZE // 2

        # Draw rotated cube onto a temp surface, then blit to screen
        sq = pygame.Surface((C.PLAYER_SIZE, C.PLAYER_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(sq, C.PLAYER_COLOR, sq.get_rect(), border_radius=4)
        # White cross detail (like real GD cube icons)
        pygame.draw.line(sq, (255, 255, 255, 160),
                         (4, 4), (C.PLAYER_SIZE - 4, C.PLAYER_SIZE - 4), 2)
        pygame.draw.line(sq, (255, 255, 255, 160),
                         (C.PLAYER_SIZE - 4, 4), (4, C.PLAYER_SIZE - 4), 2)

        rotated = pygame.transform.rotate(sq, self.angle)
        surface.blit(rotated, rotated.get_rect(center=(cx, cy)))

        # Debug: draw hitbox outline in yellow
        if debug:
            pygame.draw.rect(surface, (255, 255, 0), self.hitbox, 1)

    # ── State dict ────────────────────────────────────────────────────────────

    def state(self) -> dict:
        """Return player state as a plain dict (used in obs)."""
        return {
            "player_y":  self.y,
            "player_vy": self.vy,
            "on_ground": self.on_ground,
            "alive":     self.alive,
        }


# =============================================================================
# Spike
# =============================================================================

class Spike:
    """
    Triangular hazard sitting on the ground.

    Hitbox is NARROWER than the visual triangle — matching real GD behaviour
    where the outer edges of the spike silhouette are non-lethal.
    Kill zone = inner (1 - 2×SPIKE_HITBOX_MARGIN) fraction of the base width.
    """

    kind = "spike"

    def __init__(self, x: float) -> None:
        self.x = x                  # left edge of the spike base (px)
        self.w = float(C.SPIKE_W)   # base width (px)
        self.h = float(C.SPIKE_H)   # height     (px)

    # ── Scrolling ─────────────────────────────────────────────────────────────

    def update(self, dx: float) -> None:
        """Scroll left by dx pixels (called each frame)."""
        self.x -= dx

    @property
    def offscreen(self) -> bool:
        """True when fully off the left edge — safe to remove."""
        return self.x + self.w < 0

    # ── Hitbox ────────────────────────────────────────────────────────────────

    @property
    def hitbox(self) -> pygame.Rect:
        """
        Narrow rectangle representing the dangerous inner zone of the spike.

        Real GD spikes: you can graze the outer edges of the visual triangle
        and survive. Only the inner ~40% of the base width is actually lethal.

        SPIKE_HITBOX_MARGIN = 0.30 means we trim 30% off each side.
        Kill zone width = SPIKE_W × (1 - 2×0.30) = 40% of SPIKE_W = 12 px.
        """
        margin_px = int(self.w * C.SPIKE_HITBOX_MARGIN)
        return pygame.Rect(
            int(self.x) + margin_px,
            int(C.GROUND_Y - self.h),
            int(self.w - 2 * margin_px),
            int(self.h),
        )

    # ── Rendering ─────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface, debug: bool = False) -> None:
        # Draw the full visual triangle
        tip_x = int(self.x) + int(self.w) // 2
        tip_y = int(C.GROUND_Y - self.h)
        bl    = (int(self.x),          C.GROUND_Y)
        br    = (int(self.x + self.w), C.GROUND_Y)
        pygame.draw.polygon(surface, C.SPIKE_COLOR, [bl, br, (tip_x, tip_y)])
        # Bright outline for visibility
        pygame.draw.polygon(surface, (255, 130, 130), [bl, br, (tip_x, tip_y)], 2)

        # Debug: draw the actual narrow kill hitbox in yellow
        if debug:
            pygame.draw.rect(surface, (255, 255, 0), self.hitbox, 1)

    # ── State dict ────────────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        """Obstacle state for the obs dict returned by Game.step()."""
        return {
            "type": self.kind,
            "x":    self.x,
            "y":    float(C.GROUND_Y - self.h),
            "w":    self.w,
            "h":    self.h,
        }


# =============================================================================
# Block
# =============================================================================

class Block:
    """
    Solid rectangular obstacle sitting on the ground.
    Full rectangle hitbox — any contact is lethal.
    The player must jump OVER the block (can't pass through sides).
    """

    kind = "block"

    def __init__(self, x: float) -> None:
        self.x = x
        self.w = float(C.BLOCK_W)
        self.h = float(C.BLOCK_H)

    def update(self, dx: float) -> None:
        self.x -= dx

    @property
    def offscreen(self) -> bool:
        return self.x + self.w < 0

    @property
    def hitbox(self) -> pygame.Rect:
        """Full rectangle — no margin. Any overlap = death."""
        return pygame.Rect(
            int(self.x),
            int(C.GROUND_Y - self.h),
            int(self.w),
            int(self.h),
        )

    def draw(self, surface: pygame.Surface, debug: bool = False) -> None:
        r = self.hitbox
        pygame.draw.rect(surface, C.BLOCK_COLOR, r, border_radius=3)
        pygame.draw.rect(surface, (140, 255, 170), r, 2, border_radius=3)
        # Inner grid lines (visual detail)
        pygame.draw.line(surface, (80, 180, 100),
                         (r.left, r.centery), (r.right, r.centery), 1)
        pygame.draw.line(surface, (80, 180, 100),
                         (r.centerx, r.top), (r.centerx, r.bottom), 1)
        if debug:
            pygame.draw.rect(surface, (255, 255, 0), self.hitbox, 1)

    def as_dict(self) -> dict:
        return {
            "type": self.kind,
            "x":    self.x,
            "y":    float(C.GROUND_Y - self.h),
            "w":    self.w,
            "h":    self.h,
        }


# =============================================================================
# Game  — the main class
# =============================================================================

class Game:
    """
    Core Geometry Dash simulation.

    Works in two modes:
      render=True   Opens a pygame window. Use for human play or watching the agent.
      render=False  Headless — no window, runs as fast as possible. Use for training.

    Parameters
    ----------
    render : bool
        Whether to open a pygame display window.
    seed : int | None
        Random seed for obstacle generation. Same seed = same level layout.
        None = random layout every reset.
    debug : bool
        Show hitbox outlines (yellow rectangles). Useful for tuning constants.
    """

    def __init__(
        self,
        render : bool          = True,
        seed   : Optional[int] = None,
        debug  : bool          = False,
    ) -> None:
        self._do_render = render
        self._seed      = seed
        self._debug     = debug
        self._rng       = random.Random(seed)

        self.player    = Player()
        self.obstacles : list[Spike | Block] = []
        self._scroll_x : float = 0.0    # total pixels scrolled this episode
        self._step_n   : int   = 0      # step counter
        self._next_x   : float = 0.0    # world-x where next obstacle spawns

        # pygame objects — only created when rendering
        self.surface : Optional[pygame.Surface]     = None
        self.clock   : Optional[pygame.time.Clock]  = None
        self.font    : Optional[pygame.font.Font]   = None

        if render:
            self._init_display()

        self._spawn_initial()

    # ── Display setup ─────────────────────────────────────────────────────────

    def _init_display(self) -> None:
        """Initialise pygame window, clock, and font."""
        if not pygame.get_init():
            pygame.init()
        pygame.font.init()
        self.surface = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))
        pygame.display.set_caption(C.WINDOW_TITLE)
        self.clock   = pygame.time.Clock()
        self.font    = pygame.font.SysFont("monospace", 15, bold=True)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> dict:
        """
        Restart the episode from the beginning.

        Parameters
        ----------
        seed : int | None
            Override the RNG seed for this episode.
            Useful when the gym wrapper wants to vary seeds per episode.

        Returns
        -------
        dict  — initial observation (same structure as step() obs)
        """
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            # Reset to same seed → same level layout every episode
            self._rng = random.Random(self._seed)

        self.player.reset()
        self.obstacles.clear()
        self._scroll_x = 0.0
        self._step_n   = 0
        self._next_x   = float(C.SPAWN_X)

        self._spawn_initial()
        return self._obs()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int, dt: float = 1.0 / C.FPS) -> tuple[dict, float, bool]:
        """
        Advance the simulation by one timestep.

        Parameters
        ----------
        action : int
            0  →  do nothing
            1  →  jump (ignored if already airborne)
        dt : float
            Timestep in seconds. Default = 1/60 s (one frame at 60 fps).
            You can slow down time by passing a smaller dt.

        Returns
        -------
        obs    : dict    current game state (see module docstring for full schema)
        reward : float   REWARD_ALIVE normally, REWARD_DEATH on collision
        done   : bool    True if the player has died
        """
        self._step_n += 1

        # 1. Apply action
        if action == 1:
            self.player.jump()

        # 2. Physics
        self.player.update(dt)

        # 3. Scroll all obstacles left
        dx = C.GAME_SPEED * dt
        self._scroll_x += dx
        for obs in self.obstacles:
            obs.update(dx)

        # 4. Remove obstacles that have scrolled off the left edge
        self.obstacles = [o for o in self.obstacles if not o.offscreen]

        # 5. Spawn new obstacles to keep the level going
        self._maybe_spawn()

        # 6. Collision detection
        dead = self._check_collision()

        # 7. Compute reward
        reward = C.REWARD_DEATH if dead else C.REWARD_ALIVE
        done   = dead

        return self._obs(), reward, done

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self) -> None:
        """
        Draw the current frame to the pygame window.

        In human play mode this is called automatically inside the game loop.
        In agent/headless mode you can call this manually whenever you want
        to visually inspect what the agent is doing.
        """
        if self.surface is None:
            return

        self._draw_bg()
        self._draw_ground()
        for obs in self.obstacles:
            obs.draw(self.surface, debug=self._debug)
        self.player.draw(self.surface, debug=self._debug)
        self._draw_hud()
        pygame.display.flip()

    def tick(self) -> float:
        """
        Advance the pygame clock by one frame and return dt in seconds.
        Call this once per frame in the human play loop.
        Caps dt at 1/30 s to prevent physics explosions if the window lags.
        """
        if self.clock is None:
            return 1.0 / C.FPS
        ms = self.clock.tick(C.FPS)
        return min(ms / 1000.0, 1.0 / 30.0)

    def close(self) -> None:
        """Cleanly shut down pygame. Call when you're done."""
        if pygame.get_init():
            pygame.quit()

    def toggle_debug(self) -> None:
        """Flip the hitbox debug overlay on/off."""
        self._debug = not self._debug

    # ── Observation dict ──────────────────────────────────────────────────────

    def _obs(self) -> dict:
        """
        Build and return the full observation dict.
        obstacles list is sorted by x so index 0 is always the nearest one.
        """
        # Only include obstacles that are ahead of (or at) the player
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
            "step":      self._step_n,
            "obstacles": [o.as_dict() for o in upcoming],
        }

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _spawn_initial(self) -> None:
        """
        Pre-fill the level with obstacles so the screen isn't empty at start.
        Starts spawning from SPAWN_X and places 8 obstacles with random gaps.
        """
        x = float(C.SPAWN_X)
        for _ in range(8):
            gap = self._rng.randint(C.GAP_MIN, C.GAP_MAX)
            x  += gap
            self._add_obstacle(x)
        # Remember where to continue spawning from
        self._next_x = x + self._rng.randint(C.GAP_MIN, C.GAP_MAX)

    def _maybe_spawn(self) -> None:
        """
        Spawn new obstacles once existing ones scroll into view range.
        Keeps a buffer of upcoming obstacles so the agent always has something ahead.
        """
        # Keep spawning until we have obstacles well off the right edge
        while self._next_x - self._scroll_x < C.SCREEN_W + C.GAP_MAX:
            spawn_screen_x = self._next_x - self._scroll_x + C.PLAYER_X
            self._add_obstacle(spawn_screen_x)
            self._next_x += self._rng.randint(C.GAP_MIN, C.GAP_MAX)

    def _add_obstacle(self, x: float) -> None:
        """
        Create a random obstacle at screen x-coordinate x.
        Spikes are 3× more common than blocks (matching real GD early levels).
        """
        kind = self._rng.choice(["spike", "spike", "spike", "block"])
        if kind == "spike":
            self.obstacles.append(Spike(x))
        else:
            self.obstacles.append(Block(x))

    # ── Collision ─────────────────────────────────────────────────────────────

    def _check_collision(self) -> bool:
        """
        Check if the player hitbox overlaps any obstacle hitbox.
        Returns True if a collision occurred (player dies).
        """
        pr = self.player.hitbox
        for obs in self.obstacles:
            if pr.colliderect(obs.hitbox):
                self.player.alive = False
                return True
        return False

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_bg(self) -> None:
        """Dark background with subtle horizontal grid lines."""
        self.surface.fill(C.BG_COLOR)
        for y in range(0, C.GROUND_Y, 44):
            pygame.draw.line(self.surface, C.GRID_COLOR, (0, y), (C.SCREEN_W, y), 1)

    def _draw_ground(self) -> None:
        """Ground panel with scrolling vertical tile lines."""
        # Solid ground rectangle
        pygame.draw.rect(
            self.surface, C.GROUND_COLOR,
            (0, C.GROUND_Y, C.SCREEN_W, C.SCREEN_H - C.GROUND_Y),
        )
        # Ground top edge line
        pygame.draw.line(
            self.surface, (75, 75, 100),
            (0, C.GROUND_Y), (C.SCREEN_W, C.GROUND_Y), 2,
        )
        # Vertical tile lines that scroll with the world
        tile   = C.BLOCK_SIZE                   # one tile = one block width
        offset = int(self._scroll_x) % tile     # offset shifts as world scrolls
        for gx in range(-tile + offset, C.SCREEN_W, tile):
            pygame.draw.line(
                self.surface, (60, 60, 85),
                (gx, C.GROUND_Y), (gx, C.SCREEN_H), 1,
            )

    def _draw_hud(self) -> None:
        """Top-left HUD showing distance and step count."""
        if self.font is None:
            return
        lines = [
            f"dist : {int(self._scroll_x):>7} px",
            f"steps: {self._step_n:>7}",
            f"debug: {'ON ' if self._debug else 'OFF'}  (H to toggle)",
        ]
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, C.HUD_COLOR)
            self.surface.blit(surf, (10, 10 + i * 20))


# =============================================================================
# Human play entry point
# =============================================================================

def main() -> None:
    """
    Run the game in human-playable mode.

    CLI flags:
        --debug         start with hitbox overlay enabled
        --seed INT      fixed random seed (same seed = same level every run)
    """
    parser = argparse.ArgumentParser(description="Geometry Dash RL — human play")
    parser.add_argument("--debug", action="store_true",
                        help="Show hitbox outlines (yellow rectangles)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fixed RNG seed for reproducible level layout")
    args = parser.parse_args()

    game     = Game(render=True, seed=args.seed, debug=args.debug)
    attempts = 0
    best_px  = 0

    print("=" * 50)
    print("  GEOMETRY DASH RL — Human Play Mode")
    print("=" * 50)
    print("  SPACE / UP  →  jump")
    print("  R           →  restart")
    print("  H           →  toggle hitbox debug overlay")
    print("  Q / ESC     →  quit")
    print("=" * 50)

    running = True
    while running:
        dt = game.tick()    # advance clock, get dt

        # ── Event handling ────────────────────────────────────────────────────
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    action = 1
                elif event.key == pygame.K_r:
                    obs = game.reset()
                    attempts += 1
                elif event.key == pygame.K_h:
                    game.toggle_debug()
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        # Also catch keys held down (smoother jump feel)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = 1

        # ── Simulate one frame ────────────────────────────────────────────────
        obs, reward, done = game.step(action, dt)
        game.render()

        best_px = max(best_px, int(obs["scroll_x"]))

        # ── On death: print stats and auto-restart ────────────────────────────
        if done:
            attempts += 1
            dist = int(obs["scroll_x"])
            print(f"  Attempt {attempts:>3}  |  dist = {dist:>6} px  |  best = {best_px:>6} px")
            pygame.time.wait(300)   # brief pause so death feels acknowledged
            game.reset()

    game.close()
    print(f"\nSession over. {attempts} attempts. Best distance: {best_px} px.")


if __name__ == "__main__":
    main()