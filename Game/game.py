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
#        python game.py --agent --weights logs/checkpoints/policy_final.pth  # AI agent mode
#
#   Controls:
#        SPACE or UP ARROW  →  jump (or agent decides in AI mode)
#        R                  →  restart immediately
#        H                  →  toggle hitbox debug overlay
#        T                  →  toggle inference telemetry (AI mode only)
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
import sys
from typing import Optional
from pathlib import Path

import pygame

import constants as C
from level_generator import LevelGenerator

# Optional: import RL model for AI agent mode
try:
    import torch
    from rl_model import SimplePolicyNetwork
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    SimplePolicyNetwork = None
    torch = None


def _set_windows_dpi_awareness() -> None:
    """Best-effort DPI awareness so window sizes map to physical pixels on Windows."""
    if sys.platform != "win32":
        return

    try:
        import ctypes

        shcore = getattr(ctypes.windll, "shcore", None)
        user32 = getattr(ctypes.windll, "user32", None)

        if shcore is not None:
            try:
                shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass

        if user32 is not None:
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
    except Exception:
        pass


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
            if self.vy < C.MAX_FALL_SPEED:
                self.vy += C.GRAVITY * dt
                if self.vy > C.MAX_FALL_SPEED:
                    self.vy = C.MAX_FALL_SPEED
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
        """Rectangle kill zone with centered width and independent vertical insets."""
        width_frac = max(0.01, min(1.0, C.SPIKE_HITBOX_WIDTH_FRAC))
        top_inset_frac = max(0.0, min(1.0, C.SPIKE_HITBOX_TOP_INSET_FRAC))
        bottom_inset_frac = max(0.0, min(1.0, C.SPIKE_HITBOX_BOTTOM_INSET_FRAC))

        # Keep hitbox centered horizontally while allowing exact width control.
        margin_x = int(self.w * (1.0 - width_frac) * 0.5)
        hitbox_w = max(1, int(self.w * width_frac))

        # Allow independent top/bottom control (not vertically centered by default).
        inset_top = int(self.h * top_inset_frac)
        inset_bottom = int(self.h * bottom_inset_frac)
        hitbox_h = max(1, int(self.h - inset_top - inset_bottom))

        return pygame.Rect(
            int(self.x) + margin_x,
            int(self._y) + inset_top,
            hitbox_w,
            hitbox_h,
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

    def __init__(self, render: bool = True, seed: Optional[int] = None, debug: bool = False, agent_policy: Optional = None) -> None:
        self._do_render = render
        self._seed      = seed
        self._debug     = debug
        self._rng       = random.Random(seed)
        self._telemetry_enabled: bool = False
        self._telemetry_obs: dict = {}
        self._telemetry_reward: float = 0.0
        self._telemetry_done: bool = False
        
        # AI Agent state
        self._agent_policy = agent_policy
        self._agent_enabled = agent_policy is not None
        self._agent_action_probs: list = [0.0, 0.0]  # [no-jump, jump] probabilities
        self._agent_predicted_action: int = 0
        self._agent_confidence: float = 0.0

        self.player    = Player()
        self.obstacles : list[Spike | Block] = []
        self._scroll_x : float = 0.0
        self._step_n   : int   = 0
        self._next_x   : float = 0.0
        self._fixed_level     : bool = False
        self._level_dicts     : list = []
        self._cleared_obstacles : set = set()

        self.surface : Optional[pygame.Surface]     = None
        self.clock   : Optional[pygame.time.Clock]  = None
        self.font    : Optional[pygame.font.Font]   = None

        if render:
            self._init_display()

        self._spawn_initial()

    def _init_display(self) -> None:
        """Initialise pygame window, clock, and HUD font."""
        _set_windows_dpi_awareness()
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
        self._cleared_obstacles.clear()
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
        
        # Update agent inference info if in AI mode
        if self._agent_policy is not None:
            self._update_agent_inference()

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
        
        # Apply sparse rewards for successfully clearing obstacles
        if not dead:
            for obs_obj in self.obstacles:
                if obs_obj not in self._cleared_obstacles and obs_obj.x + obs_obj.w < self.player.x:
                    self._cleared_obstacles.add(obs_obj)
                    reward += C.REWARD_CLEAR

        done   = dead
        obs = self._obs()
        self._telemetry_obs = obs
        self._telemetry_reward = reward
        self._telemetry_done = done
        return obs, reward, done

    def render(self) -> None:
        """Draw the current frame to the pygame window."""
        if self.surface is None: return
        self._draw_bg()
        self._draw_ground()
        for obs in self.obstacles: obs.draw(self.surface, debug=self._debug)
        self.player.draw(self.surface, debug=self._debug)
        self._draw_hud()
        if self._telemetry_enabled:
            self._draw_telemetry_panel()
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

    def toggle_telemetry(self) -> None:
        """Flip in-game telemetry overlay on/off."""
        self._telemetry_enabled = not self._telemetry_enabled
    
    def _update_agent_inference(self) -> None:
        """Update agent prediction info for live inference display."""
        if self._agent_policy is None or not TORCH_AVAILABLE:
            return
        
        try:
            obs_norm = self.get_normalized_observation()
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=self._agent_policy.device)
            obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                logits = self._agent_policy.forward(obs_tensor)
                probs = torch.softmax(logits, dim=1)[0]  # Get first (only) batch element
                self._agent_action_probs = [probs[0].item(), probs[1].item()]
                self._agent_predicted_action = torch.argmax(probs).item()
                self._agent_confidence = probs[self._agent_predicted_action].item()
        except Exception:
            # Silently ignore errors in inference (e.g., if model not loaded properly)
            pass
    
    def get_agent_action(self) -> int:
        """Get action prediction from AI agent policy."""
        if self._agent_policy is None:
            return 0
        
        obs_norm = self.get_normalized_observation()
        return self._agent_policy.predict(obs_norm)

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
            "obstacles": obs_array,
        }

    def get_normalized_observation(self) -> list[float]:
        """
        Return observation as a normalized list of floats suitable for NN input.
        Includes: [player_y, player_vy, on_ground, obstacle features..., is_jump_possible]
        Total: 3 + (3 obstacles × 8 features) + 1 = 28 values.
        All normalized to roughly [0, 1] or [-1, 1].
        """
        obs = self._obs()
        normalized = []

        # Player state (normalized)
        normalized.append(obs["player_y"] / C.GROUND_Y)
        normalized.append(max(-1.0, min(1.0, obs["player_vy"] / C.MAX_FALL_SPEED)))
        normalized.append(obs["on_ground"])

        # Obstacle features (reduced to 3 obstacles, 8 features each)
        MAX_OBSTACLES = 3
        VISION_LIMIT_PX = 784.0  # Exactly 7 blocks of vision

        raw_upcoming = sorted([o for o in self.obstacles if o.x + o.w >= C.PLAYER_X], key=lambda o: o.x)
        
        # Merge adjacent obstacles of the same type to create macro-obstacles for the MLP
        upcoming = []
        for o in raw_upcoming:
            if not upcoming:
                upcoming.append({"kind": o.kind, "x": o.x, "y": o._y, "w": o.w, "h": o.h})
                continue
            
            last = upcoming[-1]
            if o.kind == last["kind"] and o.x <= last["x"] + last["w"] + 1.0:
                last["w"] = (o.x + o.w) - last["x"]
                last["y"] = min(last["y"], o._y)
                last["h"] = max(last["h"], o.h)
            else:
                upcoming.append({"kind": o.kind, "x": o.x, "y": o._y, "w": o.w, "h": o.h})

        for i in range(MAX_OBSTACLES):
            if i < len(upcoming):
                o = upcoming[i]
                rel_x = o["x"] - C.PLAYER_X
                
                # --- THE DISTANCE MASK ---
                if rel_x > VISION_LIMIT_PX:
                    # Obstacle is outside the 7-block actionable zone; blind the agent to it
                    normalized.extend([0.0] * 8)
                    continue
                # -------------------------

                otype = 0.0 if o["kind"] == "spike" else 1.0
                rel_y = o["y"] - self.player.y
                time_to_reach = rel_x / C.GAME_SPEED if C.GAME_SPEED > 0 else 0.0
                
                gap_top = max(0.0, o["y"] - (self.player.y + C.PLAYER_SIZE))
                gap_bot = max(0.0, (self.player.y - C.PLAYER_SIZE) - (o["y"] + o["h"])) if o["kind"] == "block" else 0.0
                
                normalized.extend([
                    otype,                                        
                    max(0.0, min(1.0, rel_x / VISION_LIMIT_PX)), # Normalize against vision limit so it scales 0.0 to 1.0
                    max(-1.0, min(1.0, rel_y / C.GROUND_Y)),     
                    o["w"] / C.BLOCK_SIZE / 5.0,                 
                    o["h"] / C.BLOCK_SIZE / 5.0,                 
                    max(0.0, min(1.0, time_to_reach / 6.0)),    
                    max(0.0, min(1.0, gap_top / C.GROUND_Y)),    
                    max(0.0, min(1.0, gap_bot / C.GROUND_Y)),    
                ])
            else:
                # Pad with zeros if fewer than 3 obstacles exist
                normalized.extend([0.0] * 8)

        # is_jump_possible_now (same as on_ground)
        normalized.append(obs["on_ground"])

        return normalized

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
            f"telem: {'ON ' if self._telemetry_enabled else 'OFF'}  (T to toggle)",
        ]
        
        # Add AI agent indicator if in agent mode
        if self._agent_enabled:
            lines.append(f"agent: AI MODE  (watching trained agent)")
        
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, C.HUD_COLOR)
            self.surface.blit(surf, (10, 10 + i * 20))

    def _draw_telemetry_panel(self) -> None:
        """Right-side in-game telemetry panel."""
        if self.surface is None or self.font is None:
            return

        # Adjust panel height based on whether agent mode is active
        panel_w = 700
        panel_h = 240 if self._agent_enabled else 160
        panel_x = C.SCREEN_W - panel_w - 12
        panel_y = 12
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((15, 15, 30, 200))
        pygame.draw.rect(panel, (90, 90, 120, 240), panel.get_rect(), 2)
        self.surface.blit(panel, (panel_x, panel_y))

        obs = self._telemetry_obs if self._telemetry_obs else self._obs()
        obstacles = obs.get("obstacles", [])
        if len(obstacles) >= 5 and (obstacles[1] != 0.0 or obstacles[3] != 0.0):
            otype = "spike" if obstacles[0] == 0.0 else "block"
            next_obs = f"{otype} x={obstacles[1]:.1f} y={obstacles[2]:.1f} w={obstacles[3]:.1f} h={obstacles[4]:.1f}"
        else:
            next_obs = "none"

        lines = [
            f"telemetry  |  reward={self._telemetry_reward:.3f}  done={self._telemetry_done}",
            f"player_y={obs.get('player_y', 0.0):.1f}  vy={obs.get('player_vy', 0.0):.1f}  on_ground={obs.get('on_ground', 0.0):.0f}",
            f"distance={int(self._scroll_x)} px  steps={self._step_n}",
            f"next: {next_obs}",
            (
                "spike_hitbox: "
                f"w={C.SPIKE_HITBOX_WIDTH_FRAC:.3f}, "
                f"top={C.SPIKE_HITBOX_TOP_INSET_FRAC:.3f}, "
                f"bot={C.SPIKE_HITBOX_BOTTOM_INSET_FRAC:.3f}"
            ),
        ]
        
        # Add AI agent inference info if enabled
        if self._agent_enabled:
            lines.append("-" * 80)
            lines.append(
                f"AI AGENT INFERENCE  |  "
                f"action={'JUMP' if self._agent_predicted_action == 1 else 'WAIT':>4}  "
                f"confidence={self._agent_confidence:.1%}"
            )
            lines.append(
                f"probs: no-jump={self._agent_action_probs[0]:.3f}  jump={self._agent_action_probs[1]:.3f}"
            )

        for i, line in enumerate(lines):
            surf = self.font.render(line, True, C.HUD_COLOR)
            self.surface.blit(surf, (panel_x + 10, panel_y + 10 + i * 28))


# =============================================================================
# Human play entry point
# =============================================================================

def main() -> None:
    """
    Run the game in human-playable mode for testing.

    CLI flags:
        --debug         start with hitbox overlay enabled
        --seed INT      fixed random seed (same seed = same level every run)
        --agent         watch trained AI agent play
        --weights PATH  path to trained model weights (use with --agent)
    """
    parser = argparse.ArgumentParser(description="Geometry Dash RL — human play or AI agent mode")
    parser.add_argument("--debug", action="store_true", help="Show hitbox outlines (yellow rectangles)")
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Fixed RNG seed for reproducible level layout (default: 42 in agent mode, random in human mode)"
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=None,
        help="Level difficulty 1-5 (1=easy, 5=extreme). Agent mode defaults to 1. "
             "If not specified, uses random procedural obstacles instead of LevelGenerator."
    )
    parser.add_argument(
        "--length",
        type=int,
        default=6000,
        help="Level length in pixels when using --difficulty (default: 6000px ≈ 20 seconds)"
    )
    parser.add_argument("--telemetry", action="store_true", help="Start with in-game telemetry panel enabled")
    
    # ── AI Agent Mode ─────────────────────────────────────────────────────────
    # These flags control INFERENCE (watching trained agent), NOT training.
    # Training is done with train.py, this just loads weights and watches.
    parser.add_argument(
        "--agent", 
        action="store_true", 
        help="INFERENCE MODE: Watch trained AI agent play (loads model weights, no training)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="logs/checkpoints/policy_final.pth",
        help="Path to trained model checkpoint (.pth file). Use with --agent flag. "
             "Example: --weights logs/checkpoints/policy_ep500.pth"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        choices=["cpu", "cuda"], 
        help="Device for inference (cpu or cuda). Only matters with --agent flag."
    )
    args = parser.parse_args()
    
    # Load AI agent if requested
    agent_policy = None
    if args.agent:
        if not TORCH_AVAILABLE:
            print("ERROR: PyTorch not available. Cannot use --agent mode.")
            print("Install with: pip install torch")
            sys.exit(1)
        
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"ERROR: Weights file not found: {weights_path}")
            print("Train a model first with: python train.py")
            sys.exit(1)
        
        print(f"Loading trained agent from {weights_path}...")
        agent_policy = SimplePolicyNetwork(device=args.device)
        agent_policy.load(str(weights_path))
        agent_policy.eval()  # Set to evaluation mode
        print(f"✓ Agent loaded successfully ({agent_policy.parameter_count:,} parameters)")
        print("  Watching AI agent play. Press T to see live inference.\n")
    
    # In agent mode, set sensible defaults for comparison:
    # - Fixed seed (same level every time)
    # - Difficulty 1 (easiest level)
    if args.agent:
        if args.seed is None:
            args.seed = 42  # Fixed seed so you watch agent on SAME level repeatedly
        if args.difficulty is None:
            args.difficulty = 1  # Start with easiest level
        print(f"Agent mode: Using difficulty {args.difficulty}, seed={args.seed}")
        print(f"  (Same obstacles every attempt so you can see if agent improves)\n")
    
    # Generate level using LevelGenerator if difficulty specified
    level_obstacles = None
    if args.difficulty is not None:
        print(f"Generating level: difficulty={args.difficulty}, seed={args.seed}, length={args.length}px")
        level_gen = LevelGenerator(
            difficulty=args.difficulty,
            seed=args.seed,
            progressive=False
        )
        level_obstacles = level_gen.generate(length=args.length)
        print(f"  Generated {len(level_obstacles)} obstacles\n")

    game = Game(render=True, seed=args.seed, debug=args.debug, agent_policy=agent_policy)
    
    # Load the generated level if we made one
    if level_obstacles is not None:
        game.load_level(level_obstacles)
    if args.telemetry or args.agent:
        game.toggle_telemetry()  # Auto-enable telemetry in agent mode
    attempts = 0
    best_px  = 0

    print("=" * 70)
    if args.agent:
        print("  GEOMETRY DASH RL — AI Agent Mode")
    else:
        print("  GEOMETRY DASH RL — Human Play Mode")
    print("=" * 70)
    if level_obstacles:
        print(f"  Level: Difficulty {args.difficulty} | {len(level_obstacles)} obstacles | seed={args.seed}")
    else:
        print(f"  Level: Random procedural obstacles | seed={args.seed if args.seed else 'random'}")
    print("=" * 70)
    if args.agent:
        print("  AI is playing automatically")
        print("  T           →  toggle inference telemetry")
    else:
        print("  SPACE / UP  →  jump")
    print("  R           →  restart (same level)")
    print("  H           →  toggle hitbox debug overlay")
    print("  T           →  toggle telemetry panel")
    print("  Q / ESC     →  quit")
    print("=" * 70)

    running = True
    while running:
        dt     = game.tick()
        
        # In agent mode, AI chooses actions. In human mode, player controls.
        if args.agent:
            action = game.get_agent_action()
        else:
            action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                # Only allow manual jumps in human mode
                if not args.agent and event.key in (pygame.K_SPACE, pygame.K_UP): 
                    action = 1
                if event.key == pygame.K_r:
                    # Reload the same level (if using LevelGenerator)
                    if level_obstacles is not None:
                        game.load_level(level_obstacles)
                    else:
                        game.reset()
                    attempts += 1
                elif event.key == pygame.K_h: game.toggle_debug()
                elif event.key == pygame.K_t: game.toggle_telemetry()
                elif event.key in (pygame.K_q, pygame.K_ESCAPE): running = False

        # Check held keys for responsive jumping (human mode only)
        if not args.agent:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] or keys[pygame.K_UP]: action = 1

        obs, reward, done = game.step(action, dt)
        game.render()

        best_px = max(best_px, int(game._scroll_x))

        if done:
            attempts += 1
            print(f"  Attempt {attempts:>3}  |  dist = {int(game._scroll_x):>6} px  |  best = {best_px:>6} px")
            pygame.time.wait(300)
            # Reload the same level (if using LevelGenerator)
            if level_obstacles is not None:
                game.load_level(level_obstacles)
            else:
                game.reset()

    game.close()
    print(f"\nSession over. {attempts} attempts. Best distance: {best_px} px.")


if __name__ == "__main__":
    main()