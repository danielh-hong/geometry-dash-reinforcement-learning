"""
=============================================================================
level_generator.py
=============================================================================
Procedural Geometry Dash level generator. Produces realistic obstacle layouts
based on patterns found in real GD official levels (Stereo Madness through
Cycles). Used to train and evaluate the RL agent with varied environments.

─────────────────────────────────────────────────────────────────────────────
PROJECT OVERVIEW (where this file fits)
─────────────────────────────────────────────────────────────────────────────
This is one of 3 core files in the project:

  constants.py        — all tunable numbers (physics, sizes, rewards, colours)
  game.py             — the Geometry Dash simulation engine (physics, rendering,
                        collision, step/reset API for RL)
  level_generator.py  — THIS FILE — procedurally generates levels for the
                        agent to train and be evaluated on

Future files (not yet built):
  gym_wrapper.py      — wraps game.py in the Gymnasium interface for SB3/PPO
  train.py            — PPO training loop using stable-baselines3
  yolo_detector.py    — YOLO v8 obstacle detector (replaces perfect game state)

─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES
─────────────────────────────────────────────────────────────────────────────
Generates a flat list of obstacle dicts that game.py can load via
Game.load_level(). Each dict describes one obstacle:

    { "type": "spike"|"block", "x": float, "y": float,
      "w": float, "h": float }

Levels are built by picking "chunks" — small pre-designed obstacle patterns
(like real GD levels which reuse a set of patterns throughout). Chunks are
separated by random gaps. Difficulty controls which chunk patterns are chosen
and how tight the gaps between them are.

─────────────────────────────────────────────────────────────────────────────
KEY PHYSICS CONSTRAINT (why blocks need 3-block minimum width)
─────────────────────────────────────────────────────────────────────────────
  Player cube size    = 1 block = 30px wide
  Normal speed        = 300px/s
  Jump airtime        ≈ 0.67s  (up + down)
  Horizontal per jump ≈ 200px  (300 × 0.67)
  Minimum landing zone = 3 blocks (90px)
    → Player needs 90px to touch down AND get airborne again
      before the next obstacle's hitbox hits them.
    → This is why staircase steps are 3 blocks wide.
    → This is why alternating spike gaps are 3 blocks minimum.
    → 1-block or 2-block landing zones are physically impossible to survive.

─────────────────────────────────────────────────────────────────────────────
AVAILABLE CHUNK PATTERNS
─────────────────────────────────────────────────────────────────────────────
  SIMPLE SPIKES
    chunk_single_spike          1 spike (30px wide)
                                Basic jump — bread and butter of early levels
    chunk_double_spike          2 spikes side by side (60px)
                                Needs earlier jump timing than single
    chunk_triple_spike          3 spikes side by side (90px)
                                Iconic hard obstacle — Stereo Madness 90%,
                                "9 triple spikes" in Time Machine

  BLOCK OBSTACLES
    chunk_single_block_wall     3-wide × 1-tall block wall
                                Player jumps over or lands on top
    chunk_double_block_wall     2-wide × 2-tall block wall
                                Must jump OVER — landing on top too risky
    chunk_block_platform        5-wide × 1-tall flat platform
                                Player runs along the top or vaults over
    chunk_spike_on_block        Spike sitting on top of a 3-wide block
                                Forces higher arc than a ground spike
    chunk_platform_with_spike_ends  [spike][block×3][spike]
                                Must land precisely in the middle section

  STAIRCASE PATTERNS
    chunk_staircase_up          3-step ascending staircase (3 blocks per step)
                                Stereo Madness ~25% death spot
                                Each step is 3 blocks wide — minimum safe width
    chunk_staircase_down        3-step descending staircase (3 blocks per step)
                                Mid-difficulty levels, hop down step by step

  RHYTHM / COMBO PATTERNS
    chunk_alternating_spikes    3 spikes with 3-block gaps between each
                                Forces rhythmic timed hops
                                (1-block gaps would be physically impossible)
    chunk_spike_then_platform   Spike → 3-block gap → 3-wide platform
                                Tests: jump spike, land on platform, jump off
    chunk_spike_cluster         [spike][spike] gap [spike][spike]
                                Jump into the middle gap — hard timing

─────────────────────────────────────────────────────────────────────────────
DIFFICULTY LEVELS
─────────────────────────────────────────────────────────────────────────────
  1 — Stereo Madness / Back On Track
        Patterns : mostly single spikes, block walls, occasional stairs
        Gaps     : 360–540px between chunks (1.2–1.8s breathing room)
        Real GD  : 1-star Easy difficulty

  2 — Polargeist / Dry Out
        Patterns : double spikes, spike-on-block, alternating spikes
        Gaps     : 270–420px (0.9–1.4s)
        Real GD  : 2–3 star Normal difficulty

  3 — Base After Base / Can't Let Go
        Patterns : triple spikes, platform+spike combos, spike_then_platform
        Gaps     : 180–300px (0.6–1.0s)
        Real GD  : 4–5 star Hard difficulty

  4 — Jumper / Time Machine
        Patterns : triple spikes dominant, spike clusters, staircase down
        Gaps     : 150–240px (0.5–0.8s) — getting tight
        Real GD  : 6–7 star Harder difficulty

  5 — Cycles and beyond
        Patterns : maximum variety, tightest gaps, spike clusters everywhere
        Gaps     : 120–180px (0.4–0.6s) — very tight
        Real GD  : 8+ star Insane/Demon difficulty

─────────────────────────────────────────────────────────────────────────────
PROGRESSIVE MODE (for RL curriculum training)
─────────────────────────────────────────────────────────────────────────────
  LevelGenerator(difficulty=4, progressive=True)

  Ramps from difficulty 1 → 4 over the first 70% of the level.
  The last 30% stays at difficulty 4.

  Why this matters for RL:
  If you train on difficulty 4 from the start, the agent never sees easy
  obstacles first and has a very hard time learning anything useful.
  Progressive difficulty is like curriculum learning — start simple,
  add complexity gradually. This is a well-known trick for hard RL problems.

─────────────────────────────────────────────────────────────────────────────
HOW TO RUN (command line preview)
─────────────────────────────────────────────────────────────────────────────
  Make sure venv is active first:
    source venv/bin/activate        (Mac/Linux)
    venv\\Scripts\\activate           (Windows)

  Then:
    python level_generator.py                          # difficulty 1, default
    python level_generator.py --diff 2                 # medium difficulty
    python level_generator.py --diff 3 --seed 99       # fixed layout
    python level_generator.py --diff 4 --progressive   # ramping difficulty
    python level_generator.py --diff 2 --debug         # show hitboxes (yellow)
    python level_generator.py --diff 5 --length 18000  # full 60s level

  Controls in the preview window:
    SPACE / UP  →  jump
    R           →  restart (same level)
    H           →  toggle hitbox debug overlay
    Q / ESC     →  quit

─────────────────────────────────────────────────────────────────────────────
HOW TO USE FROM PYTHON (for RL training)
─────────────────────────────────────────────────────────────────────────────
  from level_generator import LevelGenerator
  from game import Game

  # ── Basic usage ────────────────────────────────────────────────────────────
  gen = LevelGenerator(difficulty=2, seed=42)
  g   = Game(render=False)                    # headless = fast training
  obs = g.load_level(gen.generate(length=6000))

  while True:
      action = agent.predict(obs)             # 0 = do nothing, 1 = jump
      obs, reward, done = g.step(action)
      if done:
          obs = g.reset()                     # reloads the SAME level

  # ── Different level each episode (for generalisation) ──────────────────────
  for episode in range(1000):
      gen = LevelGenerator(difficulty=2, seed=episode)
      obs = g.load_level(gen.generate())
      ...

  # ── Progressive curriculum ─────────────────────────────────────────────────
  gen = LevelGenerator(difficulty=4, seed=42, progressive=True)
  obs = g.load_level(gen.generate(length=12000))

─────────────────────────────────────────────────────────────────────────────
OBSTACLE DICT FORMAT (what generate() returns)
─────────────────────────────────────────────────────────────────────────────
  Each obstacle is a plain Python dict:

    {
      "type" : "spike" | "block"   obstacle type
      "x"    : float               left edge x position in screen pixels
      "y"    : float               top  edge y position in screen pixels
      "w"    : float               width  in pixels (always 30 = 1 block)
      "h"    : float               height in pixels (30 per block level)
    }

  This format is intentionally identical to what YOLO will output later.
  When we swap in YOLO detections instead of perfect game state, the rest
  of the pipeline doesn't need to change at all — same dict format.

─────────────────────────────────────────────────────────────────────────────
ADDING NEW CHUNK PATTERNS
─────────────────────────────────────────────────────────────────────────────
  1. Write a function with signature:
       def chunk_my_pattern(x: float) -> tuple[list[dict], float]:
           ...
           return obstacle_list, total_width_consumed

  2. Add it to the relevant POOL_N lists with a weight.

  3. Test it:
       python level_generator.py --diff N --debug
       (yellow boxes show hitboxes — verify nothing is impossibly tight)

  Rules for new chunks:
    - Landing zones must be >= 3 blocks (90px) wide
    - Never place spikes < 3 blocks apart (player can't fit between them)
    - Stacked blocks > 3 high are unjumpable — don't use them
    - Always return the correct width so the gap between chunks is right
=============================================================================
"""


from __future__ import annotations

import random
from typing import Optional

import constants as C

B = C.BLOCK_SIZE   # 30px shorthand — everything is multiples of this


# =============================================================================
# Low-level object helpers
# =============================================================================

def _spike(x: float) -> dict:
    """One spike sitting on the ground."""
    return {
        "type": "spike",
        "x":    float(x),
        "y":    float(C.GROUND_Y - C.SPIKE_H),
        "w":    float(C.SPIKE_W),
        "h":    float(C.SPIKE_H),
    }

def _block(x: float, stack: int = 1) -> list[dict]:
    """
    One column of `stack` blocks stacked vertically.
    stack=1 → single block sitting on ground (30px tall)
    stack=2 → two blocks tall (60px) — player must jump over, can't walk through
    stack=3 → three blocks tall (90px) — maximum staircase height

    Returns a list of dicts (one per block in the column).
    Each block is a separate object with its own y position.
    """
    return [
        {
            "type": "block",
            "x":    float(x),
            "y":    float(C.GROUND_Y - C.BLOCK_H * (i + 1)),
            "w":    float(C.BLOCK_W),
            "h":    float(C.BLOCK_H),
        }
        for i in range(stack)
    ]


# =============================================================================
# Chunk functions
# =============================================================================
# Each function takes a starting x (screen pixels) and returns:
#   (list_of_obstacle_dicts, width_consumed_in_pixels)
#
# The generator calls these sequentially, adding random gaps between them.
# Width is how much horizontal space the chunk takes up — next chunk starts
# at x + width + gap.

# ── Simple spikes ─────────────────────────────────────────────────────────────

def chunk_single_spike(x: float) -> tuple[list[dict], float]:
    """One spike. The most basic GD obstacle. Bread and butter of early levels."""
    return [_spike(x)], float(B)

def chunk_double_spike(x: float) -> tuple[list[dict], float]:
    """
    Two spikes side by side (60px wide).
    Requires jumping slightly earlier than a single spike.
    Appears from Back On Track onward.
    """
    return [_spike(x), _spike(x + B)], float(2 * B)

def chunk_triple_spike(x: float) -> tuple[list[dict], float]:
    """
    Three spikes side by side (90px wide).
    The iconic hard obstacle — famous from Stereo Madness ~90%,
    common in Time Machine ("9 triple spikes"), Cycles, etc.
    Requires maximum jump arc — jump at the right moment to clear all three.
    """
    return [_spike(x), _spike(x + B), _spike(x + 2 * B)], float(3 * B)

# ── Block obstacles ───────────────────────────────────────────────────────────

def chunk_single_block_wall(x: float) -> tuple[list[dict], float]:
    """
    A 1-block-tall wall, 3 blocks wide.
    Player jumps over it — can land on top or just clear it.
    3 blocks wide = safe landing zone if agent wants to jump on top.
    """
    objs = []
    for col in range(3):
        objs.extend(_block(x + col * B, stack=1))
    return objs, float(3 * B)

def chunk_double_block_wall(x: float) -> tuple[list[dict], float]:
    """
    A 2-block-tall wall, 2 blocks wide.
    Player must jump OVER it (can't walk through, can't land on top — too risky).
    Jump height ≈ 3 blocks so player can just barely clear a 2-block wall.
    """
    objs = []
    for col in range(2):
        objs.extend(_block(x + col * B, stack=2))
    return objs, float(2 * B)

def chunk_block_platform(x: float, length: int = 5) -> tuple[list[dict], float]:
    """
    Flat platform of blocks — 5 wide by default (150px).
    Player runs along the top or jumps over the whole thing.
    5 blocks wide matches real GD early-level platform widths.
    From Stereo Madness: "four adjacent narrow platforms" and similar sections.
    """
    objs = []
    for i in range(length):
        objs.extend(_block(x + i * B, stack=1))
    return objs, float(length * B)

def chunk_spike_on_block(x: float) -> tuple[list[dict], float]:
    """
    One spike sitting on top of a 3-wide block platform.
    Can't land on the block — must jump OVER.
    Forces a taller arc than a plain ground spike.
    Common pattern throughout GD levels.

    [spike on top]
    [block][block][block]
    """
    objs = []
    for col in range(3):
        objs.extend(_block(x + col * B, stack=1))
    # Spike sits on top of middle block
    objs.append({
        "type": "spike",
        "x":    float(x + B),          # centred on the 3-wide platform
        "y":    float(C.GROUND_Y - C.BLOCK_H - C.SPIKE_H),
        "w":    float(C.SPIKE_W),
        "h":    float(C.SPIKE_H),
    })
    return objs, float(3 * B)

def chunk_platform_with_spike_ends(x: float) -> tuple[list[dict], float]:
    """
    Platform with spikes guarding both ends — must land in the middle.
    Tests precision: jump onto the platform cleanly, then jump off cleanly.
    Pattern: [spike] [block×3] [spike]
    """
    objs = [_spike(x)]
    for col in range(3):
        objs.extend(_block(x + B + col * B, stack=1))
    objs.append(_spike(x + 4 * B))
    return objs, float(5 * B)

# ── Staircase patterns ────────────────────────────────────────────────────────

def chunk_staircase_up(x: float, steps: int = 3) -> tuple[list[dict], float]:
    """
    Ascending staircase. Classic Stereo Madness ~25% pattern.

          [■■■]        step 3: 3 blocks tall × 3 wide
       [■■■]           step 2: 2 blocks tall × 3 wide
    [■■■]              step 1: 1 block tall  × 3 wide

    STEP WIDTH = 3 BLOCKS (90px) — this is critical.
    At normal speed the player needs 90px of landing zone to touch down
    and get airborne again before the edge of the next (taller) step kills them.
    Guides confirm: "the staircase around 25% catches players who jump too early
    — wait for your cube to fully land on each platform before jumping."
    """
    objs: list[dict] = []
    step_w = 3  # columns per step
    for i in range(steps):
        sx = x + i * step_w * B
        for col in range(step_w):
            objs.extend(_block(sx + col * B, stack=i + 1))
    return objs, float(steps * step_w * B)

def chunk_staircase_down(x: float, steps: int = 3) -> tuple[list[dict], float]:
    """
    Descending staircase. Same 3-block-wide step rule.
    Appears in mid-difficulty levels — player must hop down step by step.
    """
    objs: list[dict] = []
    step_w = 3
    for i in range(steps):
        height = steps - i       # tallest first
        sx = x + i * step_w * B
        for col in range(step_w):
            objs.extend(_block(sx + col * B, stack=height))
    return objs, float(steps * step_w * B)

# ── Rhythm patterns ───────────────────────────────────────────────────────────

def chunk_alternating_spikes(x: float) -> tuple[list[dict], float]:
    """
    Three spikes with gaps between — forces rhythmic hops.

    [spike] ... [spike] ... [spike]
         ^gap^        ^gap^
         3 blocks     3 blocks

    Gap = 3 blocks (90px) minimum — player needs that much horizontal space
    to land safely and take off again before hitting the next spike.
    """
    return [_spike(x), _spike(x + 4 * B), _spike(x + 8 * B)], float(9 * B)

def chunk_spike_then_platform(x: float) -> tuple[list[dict], float]:
    """
    [spike] [3-block gap] [block platform × 3]
    Jump over spike, land on block platform, jump off.
    Tests sequential obstacle awareness.
    """
    objs = [_spike(x)]
    for col in range(3):
        objs.extend(_block(x + 4 * B + col * B, stack=1))
    return objs, float(7 * B)

def chunk_spike_cluster(x: float) -> tuple[list[dict], float]:
    """
    Dense spike cluster with one 3-block gap in the middle.
    Pattern: [spike][spike][gap×3][spike][spike]
    Player must time jump to land in the gap.
    From harder GD sections.
    """
    objs = [_spike(x), _spike(x + B), _spike(x + 5 * B), _spike(x + 6 * B)]
    return objs, float(7 * B)


# =============================================================================
# Difficulty pools
# =============================================================================
# Each entry: (chunk_function, relative_weight)
# Higher weight = chosen more often.

POOL_1 = [   # Stereo Madness / Back On Track — lots of breathing room
    (chunk_single_spike,           40),
    (chunk_double_spike,           20),
    (chunk_single_block_wall,      20),
    (chunk_block_platform,         15),
    (chunk_staircase_up,            5),
]

POOL_2 = [   # Polargeist / Dry Out — double spikes, spike+block combos
    (chunk_single_spike,           15),
    (chunk_double_spike,           30),
    (chunk_single_block_wall,      15),
    (chunk_spike_on_block,         20),
    (chunk_staircase_up,           10),
    (chunk_alternating_spikes,     10),
]

POOL_3 = [   # Base After Base / Can't Let Go — triple spikes, combos
    (chunk_double_spike,           15),
    (chunk_triple_spike,           25),
    (chunk_spike_on_block,         15),
    (chunk_platform_with_spike_ends, 15),
    (chunk_spike_then_platform,    15),
    (chunk_alternating_spikes,     10),
    (chunk_double_block_wall,       5),
]

POOL_4 = [   # Jumper / Time Machine — dense, everything
    (chunk_triple_spike,           30),
    (chunk_double_spike,           10),
    (chunk_spike_on_block,         15),
    (chunk_spike_cluster,          20),
    (chunk_platform_with_spike_ends, 10),
    (chunk_staircase_down,         10),
    (chunk_double_block_wall,       5),
]

POOL_5 = [   # Cycles and beyond — maximum density
    (chunk_triple_spike,           25),
    (chunk_spike_cluster,          25),
    (chunk_spike_on_block,         15),
    (chunk_platform_with_spike_ends, 15),
    (chunk_alternating_spikes,     10),
    (chunk_staircase_down,         10),
]

POOLS = {1: POOL_1, 2: POOL_2, 3: POOL_3, 4: POOL_4, 5: POOL_5}

# Gap BETWEEN chunks at each difficulty (in pixels).
# This is the breathing room between obstacle groups.
# At 300px/s: 180px = 0.6s, 360px = 1.2s, 540px = 1.8s
GAPS = {
    1: (360, 540),    # lots of space — 1.2s to 1.8s
    2: (270, 420),    # comfortable   — 0.9s to 1.4s
    3: (180, 300),    # tighter       — 0.6s to 1.0s
    4: (150, 240),    # tight         — 0.5s to 0.8s
    5: (120, 180),    # very tight    — 0.4s to 0.6s
}


# =============================================================================
# LevelGenerator
# =============================================================================

class LevelGenerator:
    """
    Generates a procedural Geometry Dash-style level as a list of obstacle dicts.

    Parameters
    ----------
    difficulty : int  1–5
        1 = Stereo Madness (easy), 5 = Cycles/extreme.
    seed : int | None
        Fixed seed → same level every call. None = random.
    progressive : bool
        Ramp difficulty from 1 → `difficulty` over the first 70% of the level.
        Good for curriculum RL training.
    """

    def __init__(
        self,
        difficulty  : int           = 1,
        seed        : Optional[int] = None,
        progressive : bool          = False,
    ) -> None:
        assert 1 <= difficulty <= 5, "difficulty must be 1–5"
        self.difficulty  = difficulty
        self.progressive = progressive
        self._rng        = random.Random(seed)

    def generate(self, length: int = 6000) -> list[dict]:
        """
        Generate a level of approximately `length` pixels.

        At 300px/s:
            3000px  ≈ 10s of gameplay
            6000px  ≈ 20s (one section of a real GD level)
            18000px ≈ 60s (full level length)

        Returns
        -------
        list[dict] — obstacle dicts sorted by x, ready for Game.load_level()
        """
        obstacles: list[dict] = []

        # Start well past the right edge so first obstacle scrolls in naturally.
        # 1.5s of run-up at 300px/s = 450px buffer.
        x = float(C.SCREEN_W + C.GAME_SPEED * 1.5)

        while x < length + C.SCREEN_W:
            # Determine current difficulty
            if self.progressive:
                progress = min((x - C.SCREEN_W) / length, 1.0)
                ramp     = min(progress / 0.7, 1.0)      # ramp over first 70%
                cur_diff = max(1, round(1 + ramp * (self.difficulty - 1)))
            else:
                cur_diff = self.difficulty

            # Pick and place a chunk
            chunk_fn          = self._weighted_choice(POOLS[cur_diff])
            chunk_objs, width = chunk_fn(x)
            obstacles.extend(chunk_objs)
            x += width

            # Add gap before next chunk
            gap_min, gap_max = GAPS[cur_diff]
            x += self._rng.randint(gap_min, gap_max)

        obstacles.sort(key=lambda o: o["x"])
        return obstacles

    def _weighted_choice(self, pool: list[tuple]) -> object:
        fns, weights = zip(*pool)
        total = sum(weights)
        r = self._rng.randint(1, total)
        cumulative = 0
        for fn, w in zip(fns, weights):
            cumulative += w
            if r <= cumulative:
                return fn
        return fns[-1]


# =============================================================================
# Preview entry point
# =============================================================================

def main() -> None:
    """
    Visual preview of a generated level.

    Usage:
        python level_generator.py                        # diff 1
        python level_generator.py --diff 3               # diff 3
        python level_generator.py --diff 2 --seed 99     # fixed seed
        python level_generator.py --diff 4 --progressive # ramping difficulty
        python level_generator.py --diff 2 --debug       # show hitboxes
    """
    import argparse
    import pygame
    from game import Game

    parser = argparse.ArgumentParser(description="Preview a generated GD level")
    parser.add_argument("--diff",       type=int,  default=1)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--length",     type=int,  default=9000)
    parser.add_argument("--progressive", action="store_true")
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--telemetry",  action="store_true",
                        help="Start with in-game telemetry panel enabled")
    args = parser.parse_args()

    gen       = LevelGenerator(difficulty=args.diff, seed=args.seed,
                               progressive=args.progressive)
    obstacles = gen.generate(length=args.length)

    # Print summary
    types: dict[str, int] = {}
    for o in obstacles:
        types[o["type"]] = types.get(o["type"], 0) + 1
    print(f"\nDifficulty {args.diff} | seed={args.seed} | "
          f"progressive={args.progressive} | length={args.length}px "
          f"({args.length / C.GAME_SPEED:.0f}s)")
    print(f"Generated {len(obstacles)} objects: {types}")
    print("SPACE/UP=jump  R=restart  H=hitboxes  T=telemetry  Q=quit\n")

    game = Game(render=True, debug=args.debug)
    if args.telemetry:
        game.toggle_telemetry()
    game.load_level(obstacles)

    attempts = 0
    best     = 0
    running  = True

    while running:
        dt     = game.tick()
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    action = 1
                elif event.key == pygame.K_r:
                    game.load_level(obstacles)
                    attempts += 1
                elif event.key == pygame.K_h:
                    game.toggle_debug()
                elif event.key == pygame.K_t:
                    game.toggle_telemetry()
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        # Also respond to held key for smoother feel
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = 1

        obs, reward, done = game.step(action, dt)
        game.render()
        best = max(best, int(game._scroll_x))

        if done:
            attempts += 1
            print(f"  #{attempts:>3}  dist={int(game._scroll_x):>6}px  best={best:>6}px")
            pygame.time.wait(250)
            game.load_level(obstacles)

    game.close()


if __name__ == "__main__":
    main()