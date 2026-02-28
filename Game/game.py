# =============================================================================
# game.py
# =============================================================================
# Self-contained Geometry Dash clone with accurate platforming physics.
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
# ── OBS DICT (Fixed-Size for Neural Networks) ────────────────────────────────
#
#   To feed this state into standard RL libraries (SB3, RLlib), the observation
#   space must be a fixed shape. The obstacles are flattened into a 1D array.
#
#   {
#     "player_y"   : float   top-left y of the player cube (px)
#     "player_vy"  : float   vertical velocity — negative = moving up (px/s)
#     "on_ground"  : float   1.0 if resting on ground/block, 0.0 if airborne
#     "obstacles"  : list    Fixed-length list of 25 floats (5 obstacles * 5 features).
#                            If there are fewer than 5 obstacles on screen,
#                            the remaining slots are padded with 0.0.
#                            Features per obstacle:
#                              [ type (0=spike, 1=block), x, y, width, height ]
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
import random
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
    x, y       : float   Top-left pixel position on screen.
    last_y     : float   The y-position from the previous frame (used for collision resolution).
    vy         : float   Vertical velocity in px/s (negative = upward).
    on_ground  : bool    True when sitting on the main floor or a block platform.
    alive      : bool    Set to False on a lethal collision.
    angle      : float   Visual rotation angle in degrees (cosmetic only).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Put the player back at the start position and reset physics state."""
        self.x        : float = float(C.PLAYER_X)
        self.y        : float = float(C.GROUND_Y - C.PLAYER_SIZE)
        self.last_y   : float = self.y  
        self.vy       : float = 0.0
        self.on_ground: bool  = True
        self.alive    : bool  = True
        self.angle    : float = 0.0   

    # ── Actions ───────────────────────────────────────────────────────────────

    def jump(self) -> None:
        """
        Jump if currently supported by the ground or a block.
        Ignored while airborne (no double-jump in basic cube mode).
        """
        if self.on_ground:
            self.vy = C.JUMP_VEL      
            self.on_ground = False

    # ── Physics update ────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """
        Advance player physics by dt seconds.
        Called once per frame from Game.step().
        """
        if not self.alive:
            return

        # Record previous position for continuous collision detection
        self.last_y = self.y 

        # Only apply gravity and spin if we are actually falling/jumping
        if not self.on_ground:
            self.vy += C.GRAVITY * dt
            self.angle = (self.angle - 220 * dt) % 360
        else:
            # Snap angle nicely to the nearest 90 degrees when sliding on a block
            self.angle = round(self.angle / 90.0) * 90.0

        # Move vertically
        self.y += self.vy * dt

        # Main floor collision clamp
        ground_top = float(C.GROUND_Y - C.PLAYER_SIZE)
        if self.y >= ground_top:
            self.y         = ground_top
            self.vy        = 0.0
            self.on_ground = True
            self.angle     = 0.0

    # ── Hitbox ────────────────────────────────────────────────────────────────

    @property
    def hitbox(self) -> pygame.Rect:
        """
        The actual collision rectangle used for death detection.
        Slightly smaller than the visual cube (HITBOX_MARGIN px inward on all sides).
        Matches the forgiving nature of real GD hitboxes.
        """
        m = C.HITBOX_MARGIN
        return pygame.Rect(
            int(self.x) + m,
            int(self.y) + m,
            C.PLAYER_SIZE - 2 * m,
            C.PLAYER_SIZE - 2 * m,
        )

    # ── Rendering ─────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface, debug: bool = False) -> None:
        """Renders the player cube to the provided surface."""
        if not self.alive:
            return

        cx = int(self.x) + C.PLAYER_SIZE // 2
        cy = int(self.y) + C.PLAYER_SIZE // 2

        # Draw rotated cube onto a temp surface, then blit to screen
        sq = pygame.Surface((C.PLAYER_SIZE, C.PLAYER_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(sq, C.PLAYER_COLOR, sq.get_rect(), border_radius=4)
        pygame.draw.line(sq, (255, 255, 255, 160), (4, 4), (C.PLAYER_SIZE - 4, C.PLAYER_SIZE - 4), 2)
        pygame.draw.line(sq, (255, 255, 255, 160), (C.PLAYER_SIZE - 4, 4), (4, C.PLAYER_SIZE - 4), 2)

        rotated = pygame.transform.rotate(sq, self.angle)
        surface.blit(rotated, rotated.get_rect(center=(cx, cy)))

        if debug:
            pygame.draw.rect(surface, (255, 255, 0), self.hitbox, 1)


# =============================================================================
# Spike
# =============================================================================

class Spike:
    """
    Triangular hazard. Any overlap with the kill zone is lethal.
    
    Hitbox is NARROWER than the visual triangle — matching real GD behaviour
    where the outer edges of the spike silhouette are non-lethal.
    Kill zone = inner (1 - 2×SPIKE_HITBOX_MARGIN) fraction of the base width.
    """
    kind = "spike"

    def __init__(self, x: float) -> None:
        self.x = x                  
        self.w = float(C.SPIKE_W)   
        self.h = float(C.SPIKE_H)
        self._y = float(C.GROUND_Y - self.h)

    def update(self, dx: float) -> None:
        """Scroll left by dx pixels."""
        self.x -= dx

    @property
    def offscreen(self) -> bool:
        """True when fully off the left edge — safe to remove from memory."""
        return self.x + self.w < 0

    @property
    def hitbox(self) -> pygame.Rect:
        """Narrow rectangle representing the dangerous inner zone of the spike."""
        margin_px = int(self.w * C.SPIKE_HITBOX_MARGIN)
        return pygame.Rect(
            int(self.x) + margin_px,
            int(self._y),
            int(self.w - 2 * margin_px),
            int(self.h),
        )

    def draw(self, surface: pygame.Surface, debug: bool = False) -> None:
        """Renders the spike triangle and optional debug bounding box."""
        tip_x = int(self.x) + int(self.w) // 2
        tip_y = int(self._y)
        bl    = (int(self.x),          C.GROUND_Y)
        br    = (int(self.x + self.w), C.GROUND_Y)
        pygame.draw.polygon(surface, C.SPIKE_COLOR, [bl, br, (tip_x, tip_y)])
        pygame.draw.polygon(surface, (255, 130, 130), [bl, br, (tip_x, tip_y)], 2)

        if debug:
            pygame.draw.rect(surface, (255, 255, 0), self.hitbox, 1)


# =============================================================================
# Block
# =============================================================================

class Block:
    """
    Solid rectangular obstacle.
    Hitting the sides or bottom is lethal.
    Landing on the top acts as a new floor surface for the player.
    """
    kind = "block"

    def __init__(self, x: float, y: Optional[float] = None, h: Optional[float] = None) -> None:
        self.x = x
        self.w = float(C.BLOCK_W)
        self.h = float(h) if h is not None else float(C.BLOCK_H)
        self._y = float(y) if y is not None else float(C.GROUND_Y - self.h)

    def update(self, dx: float) -> None:
        """Scroll left by dx pixels."""
        self.x -= dx

    @property
    def offscreen(self) -> bool:
        """True when fully off the left edge."""
        return self.x + self.w < 0

    @property
    def hitbox(self) -> pygame.Rect:
        """Full rectangle — no margin."""
        return pygame.Rect(int(self.x), int(self._y), int(self.w), int(self.h))

    def draw(self, surface: pygame.Surface, debug: bool = False) -> None:
        """Renders the solid block with visual styling."""
        r = self.hitbox
        pygame.draw.rect(surface, C.BLOCK_COLOR, r, border_radius=3)
        pygame.draw.rect(surface, (140, 255, 170), r, 2, border_radius=3)
        pygame.draw.line(surface, (80, 180, 100), (r.left, r.centery), (r.right, r.centery), 1)
        pygame.draw.line(surface, (80, 180, 100), (r.centerx, r.top), (r.centerx, r.bottom), 1)
        if debug:
            pygame.draw.rect(surface, (255, 255, 0), self.hitbox, 1)


# =============================================================================
# Game  — the main class
# =============================================================================

class Game:
    """
    Core Geometry Dash simulation environment.

    Parameters
    ----------
    render : bool
        Whether to open a pygame display window. Disable for fast headless training.
    seed : int | None
        Random seed for obstacle generation. Same seed = same level layout.
    debug : bool
        Show hitbox outlines (yellow rectangles).
    """

    def __init__(self, render: bool = True, seed: Optional[int] = None, debug: bool = False) -> None:
        self._do_render = render
        self._seed      = seed
        self._debug     = debug
        self._rng       = random.Random(seed)

        self.player    = Player()
        self.obstacles : list[Spike | Block] = []
        self._scroll_x : float = 0.0
        self._step_n   : int   = 0
        self._next_x   : float = 0.0
        self._fixed_level     : bool = False
        self._level_dicts     : list = []

        self.surface : Optional[pygame.Surface]     = None
        self.clock   : Optional[pygame.time.Clock]  = None
        self.font    : Optional[pygame.font.Font]   = None

        if render:
            self._init_display()

        self._spawn_initial()

    def _init_display(self) -> None:
        """Initialise pygame window, clock, and HUD font."""
        if not pygame.get_init():
            pygame.init()
        pygame.font.init()
        self.surface = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))
        pygame.display.set_caption(C.WINDOW_TITLE)
        self.clock   = pygame.time.Clock()
        self.font    = pygame.font.SysFont("monospace", 15, bold=True)

    def reset(self, seed: Optional[int] = None) -> dict:
        """
        Restart the episode from the beginning.
        Returns the initial fixed-size observation dict.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)

        self.player.reset()
        self.obstacles.clear()
        self._scroll_x = 0.0
        self._step_n   = 0
        self._next_x   = float(C.SPAWN_X)

        if self._fixed_level and self._level_dicts:
            self.obstacles = []
            for d in self._level_dicts:
                if d["type"] == "spike":
                    s = Spike(d["x"]); s.w = d["w"]; s.h = d["h"]; s._y = d.get("y", C.GROUND_Y - s.h)
                    self.obstacles.append(s)
                else:
                    self.obstacles.append(Block(d["x"], y=d["y"], h=d["h"]))
        else:
            self._spawn_initial()
        return self._obs()

    def load_level(self, obstacle_dicts: list[dict]) -> dict:
        """
        Load a pre-generated level from a list of obstacle dicts.
        Locks the environment into fixed-level mode (disables procedural generation).
        """
        self.player.reset()
        self._scroll_x = 0.0
        self._step_n   = 0

        self.obstacles = []
        for d in obstacle_dicts:
            if d["type"] == "spike":
                s = Spike(d["x"]); s.w = d["w"]; s.h = d["h"]; s._y = d.get("y", C.GROUND_Y - s.h)
                self.obstacles.append(s)
            elif d["type"] == "block":
                self.obstacles.append(Block(d["x"], y=d["y"], h=d["h"]))

        self._fixed_level = True
        self._level_dicts = list(obstacle_dicts)   
        return self._obs()

    def step(self, action: int, dt: float = 1.0 / C.FPS) -> tuple[dict, float, bool]:
        """
        Advance the simulation by one timestep.

        Parameters
        ----------
        action : int
            0 = do nothing, 1 = jump.
        dt : float
            Timestep in seconds.

        Returns
        -------
        obs    : dict    Fixed-shape current game state.
        reward : float   REWARD_ALIVE normally, REWARD_DEATH on lethal collision.
        done   : bool    True if the player has died.
        """
        self._step_n += 1

        if action == 1:
            self.player.jump()

        self.player.update(dt)

        dx = C.GAME_SPEED * dt
        self._scroll_x += dx
        for obs in self.obstacles:
            obs.update(dx)

        self.obstacles = [o for o in self.obstacles if not o.offscreen]

        if not self._fixed_level:
            self._maybe_spawn()

        dead = self._check_collision()

        reward = C.REWARD_DEATH if dead else C.REWARD_ALIVE
        done   = dead

        return self._obs(), reward, done

    def render(self) -> None:
        """Draw the current frame to the pygame window."""
        if self.surface is None: return
        self._draw_bg()
        self._draw_ground()
        for obs in self.obstacles: obs.draw(self.surface, debug=self._debug)
        self.player.draw(self.surface, debug=self._debug)
        self._draw_hud()
        pygame.display.flip()

    def tick(self) -> float:
        """Advance the pygame clock by one frame and return dt in seconds."""
        if self.clock is None: return 1.0 / C.FPS
        ms = self.clock.tick(C.FPS)
        return min(ms / 1000.0, 1.0 / 30.0)

    def close(self) -> None:
        """Cleanly shut down pygame window resources."""
        if pygame.get_init(): pygame.quit()

    def toggle_debug(self) -> None:
        """Flip the hitbox debug overlay on/off."""
        self._debug = not self._debug

    # ── Observation dict ──────────────────────────────────────────────────────

    def _obs(self) -> dict:
        """
        Build and return the full observation dict.
        Transforms the variable-length obstacle list into a fixed-length 1D array
        so it can be digested by static Neural Network architectures.
        """
        MAX_OBSTACLES = 5
        upcoming = sorted([o for o in self.obstacles if o.x + o.w >= C.PLAYER_X], key=lambda o: o.x)
        
        obs_array = []
        for i in range(MAX_OBSTACLES):
            if i < len(upcoming):
                o = upcoming[i]
                # Represent type numerically: 0.0 for spike, 1.0 for block
                otype = 0.0 if o.kind == "spike" else 1.0
                obs_array.extend([otype, float(o.x), float(o._y), float(o.w), float(o.h)])
            else:
                # Pad with zeros if there are fewer than MAX_OBSTACLES on screen
                obs_array.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return {
            "player_y":  float(self.player.y),
            "player_vy": float(self.player.vy),
            "on_ground": 1.0 if self.player.on_ground else 0.0,
            "obstacles": obs_array 
        }

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _spawn_initial(self) -> None:
        """Pre-fill the level with 8 obstacles so the screen isn't empty at start."""
        x = float(C.SPAWN_X)
        for _ in range(8):
            gap = self._rng.randint(C.GAP_MIN, C.GAP_MAX)
            x  += gap
            self._add_obstacle(x)
        self._next_x = x + self._rng.randint(C.GAP_MIN, C.GAP_MAX)

    def _maybe_spawn(self) -> None:
        """Spawn new obstacles once existing ones scroll into view range."""
        while self._next_x - self._scroll_x < C.SCREEN_W + C.GAP_MAX:
            spawn_screen_x = self._next_x - self._scroll_x + C.PLAYER_X
            self._add_obstacle(spawn_screen_x)
            self._next_x += self._rng.randint(C.GAP_MIN, C.GAP_MAX)

    def _add_obstacle(self, x: float) -> None:
        """Create a random obstacle at screen x-coordinate x."""
        kind = self._rng.choice(["spike", "spike", "spike", "block"])
        if kind == "spike": self.obstacles.append(Spike(x))
        else: self.obstacles.append(Block(x))

    # ── Collision ─────────────────────────────────────────────────────────────

    def _check_collision(self) -> bool:
        """
        Calculates collision and resolves platforming physics.
        Differentiates between a fatal wall crash and a safe landing on top of a block.
        
        Returns True if a lethal collision occurred.
        """
        pr = self.player.hitbox
        
        # Calculate the lowest point the player was at in the previous frame.
        # This is vital for Continuous Collision Detection (CCD) to ensure the 
        # player doesn't clip through thin platforms when moving fast.
        prev_bottom = self.player.last_y + C.PLAYER_SIZE

        # Tracks if the player is currently resting on a block this frame
        currently_supported = False

        for obs in self.obstacles:
            # 1. Check for overlapping hitboxes
            if pr.colliderect(obs.hitbox):
                
                if obs.kind == "spike":
                    self.player.alive = False
                    return True
                
                elif obs.kind == "block":
                    # LANDING LOGIC:
                    # If the player is falling (vy >= 0) AND their previous bottom edge 
                    # was higher than the top of the block (plus an 8px physics forgiveness margin), 
                    # they safely land.
                    if self.player.vy >= 0 and prev_bottom <= obs._y + 8:
                        self.player.y = obs._y - C.PLAYER_SIZE
                        self.player.vy = 0.0
                        self.player.on_ground = True
                        self.player.angle = 0.0
                        currently_supported = True
                        
                        # Re-calculate the player's bounding box since we snapped them to the platform
                        pr = self.player.hitbox 
                    else:
                        # Player hit the side or bottom of the block — Lethal
                        self.player.alive = False
                        return True

            # 2. Check for ground support even if not strictly colliding
            # This prevents gravity from erroneously pulling the player down 
            # while they are sliding perfectly flush across the top of a platform.
            elif obs.kind == "block":
                if (obs.x <= self.player.x + C.PLAYER_SIZE and 
                    obs.x + obs.w >= self.player.x and
                    abs(self.player.y + C.PLAYER_SIZE - obs._y) < 2):
                    currently_supported = True

        # 3. Handle walking off ledges
        # If the player is not on the main floor and not supported by a block, they fall.
        if not currently_supported and self.player.y < C.GROUND_Y - C.PLAYER_SIZE:
            self.player.on_ground = False

        return False

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_bg(self) -> None:
        """Dark background with subtle horizontal grid lines."""
        self.surface.fill(C.BG_COLOR)
        for y in range(0, C.GROUND_Y, 60):
            pygame.draw.line(self.surface, C.GRID_COLOR, (0, y), (C.SCREEN_W, y), 1)

    def _draw_ground(self) -> None:
        """Ground panel with scrolling vertical tile lines."""
        pygame.draw.rect(self.surface, C.GROUND_COLOR, (0, C.GROUND_Y, C.SCREEN_W, C.SCREEN_H - C.GROUND_Y))
        pygame.draw.line(self.surface, (75, 75, 100), (0, C.GROUND_Y), (C.SCREEN_W, C.GROUND_Y), 2)
        tile   = C.BLOCK_SIZE                  
        offset = int(self._scroll_x) % tile    
        for gx in range(-tile + offset, C.SCREEN_W, tile):
            pygame.draw.line(self.surface, (60, 60, 85), (gx, C.GROUND_Y), (gx, C.SCREEN_H), 1)

    def _draw_hud(self) -> None:
        """Top-left HUD showing distance and debug state."""
        if self.font is None: return
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
    Run the game in human-playable mode for testing.

    CLI flags:
        --debug         start with hitbox overlay enabled
        --seed INT      fixed random seed (same seed = same level every run)
    """
    parser = argparse.ArgumentParser(description="Geometry Dash RL — human play")
    parser.add_argument("--debug", action="store_true", help="Show hitbox outlines (yellow rectangles)")
    parser.add_argument("--seed", type=int, default=None, help="Fixed RNG seed for reproducible level layout")
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
        dt     = game.tick()
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_UP): action = 1
                elif event.key == pygame.K_r:
                    game.reset()
                    attempts += 1
                elif event.key == pygame.K_h: game.toggle_debug()
                elif event.key in (pygame.K_q, pygame.K_ESCAPE): running = False

        # Check held keys so holding space feels responsive
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]: action = 1

        obs, reward, done = game.step(action, dt)
        game.render()

        best_px = max(best_px, int(game._scroll_x))

        if done:
            attempts += 1
            print(f"  Attempt {attempts:>3}  |  dist = {int(game._scroll_x):>6} px  |  best = {best_px:>6} px")
            pygame.time.wait(300)   
            game.reset()

    game.close()
    print(f"\nSession over. {attempts} attempts. Best distance: {best_px} px.")


if __name__ == "__main__":
    main()