# =============================================================================
# constants.py
# =============================================================================
# ALL game configuration lives here. Never hardcode values in game.py.
# Change things here and they update everywhere automatically.
#
# PHYSICS SOURCES:
#   Block size / hitboxes : gdcreatorschool.com advanced-hitboxes guide
#                           → Player hitbox is exactly 30 units square in GD
#   Speed                 : GD forum community research
#                           → Normal speed = 10 blocks/sec
#                           → 10 blocks × 30 px = 300 px/s
#   Jump height           : ~2.5–3 blocks at normal speed ≈ 75–90 px peak
#                           → Physics: peak = v² / (2 × g)
#                           → JUMP_VEL=570, GRAVITY=1710 → peak ≈ 95 px (3.2 blocks)
#   Spike hitbox          : Real GD spike kill zone ≈ inner 40% of triangle width
#                           (the pointy tip region); outer edges are forgiving
# =============================================================================


# ── Display ───────────────────────────────────────────────────────────────────

SCREEN_W     = 800          # window width  in pixels
SCREEN_H     = 400          # window height in pixels
FPS          = 60           # frames per second — real GD runs at 60 fps
WINDOW_TITLE = "Geometry Dash RL"


# ── Colours  (R, G, B) ────────────────────────────────────────────────────────

BG_COLOR     = (20,  20,  40)    # dark navy background
GROUND_COLOR = (50,  50,  70)    # slightly lighter ground panel
GRID_COLOR   = (35,  35,  55)    # subtle horizontal scanlines on background
PLAYER_COLOR = (80,  180, 255)   # bright blue cube
SPIKE_COLOR  = (255, 80,  80)    # red spikes
BLOCK_COLOR  = (100, 220, 140)   # green solid blocks
HUD_COLOR    = (200, 200, 220)   # light grey HUD text
WHITE        = (255, 255, 255)
DEBUG_HITBOX_COLOR = (255, 255, 0, 120)  # yellow, semi-transparent hitbox overlay


# ── World geometry ────────────────────────────────────────────────────────────

# One "block" in real GD = 30 pixels in our simulation.
# Everything is derived from BLOCK_SIZE so the game stays proportional.
BLOCK_SIZE   = 30           # px per block (real GD unit)

# Y-coordinate of the ground surface (top edge of the ground panel).
# The player cube sits with its bottom edge touching this line.
GROUND_Y     = SCREEN_H - 80   # leaves an 80 px tall ground panel at the bottom


# ── Game speed ────────────────────────────────────────────────────────────────

# Real GD normal (blue portal) speed = 10 blocks / second.
# 10 blocks × 30 px/block = 300 px/s horizontal scroll speed.
#
# All speed portal multipliers for reference (not yet implemented):
#   Slow    (yellow): 0.7×  →  210 px/s
#   Normal  (blue)  : 1.0×  →  300 px/s   ← this is what we use
#   Fast    (green) : 1.1×  →  330 px/s
#   Faster  (pink)  : 1.3×  →  390 px/s
#   Fastest (red)   : 1.6×  →  480 px/s
GAME_SPEED   = 300.0        # px/s  — normal (1×) speed


# ── Player physics ────────────────────────────────────────────────────────────

# Cube is exactly 1 block = 30 px square, matching real GD's documented hitbox.
PLAYER_SIZE  = BLOCK_SIZE   # px — side length of the player cube (= 30)

# Fixed screen x-position of the player. The world scrolls left; player stays put.
PLAYER_X     = 130          # px from left edge of screen

# Downward acceleration in px/s².
# Real GD gravity ≈ 0.958 GD-units per frame² at 60 fps.
# Converting: 0.958 × 30 px/block × 60² frames/s² ≈ 103,000 — way too high
# at our small scale, so we use an empirically tuned value that feels authentic.
# At GRAVITY=1710 and JUMP_VEL=570:
#   Time to peak   = 570 / 1710 ≈ 0.33 s  (≈ 20 frames) — matches real GD feel
#   Peak height    = 570² / (2 × 1710) ≈ 95 px ≈ 3.2 blocks
#   Full jump time ≈ 0.67 s (up + down) — feels snappy like real GD
GRAVITY      = 1710.0       # px/s²

# Upward velocity applied on jump (negative because pygame y increases downward).
# See GRAVITY comment above for the peak-height calculation.
JUMP_VEL     = -570.0       # px/s  (negative = upward)

# Inward shrink of the player's kill hitbox on all four sides.
# Real GD uses a hitbox slightly smaller than the visible cube so that
# near-misses feel fair. 3 px gives ~10% forgiveness on each edge.
HITBOX_MARGIN = 3           # px inward on all 4 sides of the cube


# ── Obstacles ─────────────────────────────────────────────────────────────────

# All obstacles snap to the same 30 px tile grid as the player.
SPIKE_W      = BLOCK_SIZE   # px — base width of a spike  (= 30)
SPIKE_H      = BLOCK_SIZE   # px — height of a spike      (= 30)

BLOCK_W      = BLOCK_SIZE   # px — width  of a solid block (= 30)
BLOCK_H      = BLOCK_SIZE   # px — height of a solid block (= 30)

# Spike hitbox is NARROWER than the visible triangle.
# In real GD the dangerous zone is roughly the inner 40% of spike width —
# the outer edges of the triangle silhouette won't actually kill you.
# We trim 30% off each side → kill zone = middle 40% of the triangle.
# This matches "I can barely squeeze past" GD spike feel.
SPIKE_HITBOX_MARGIN = 0.30  # fraction of SPIKE_W trimmed from each side
                             # 0.30 left + 0.30 right = 0.40 kill zone width

# Solid blocks use the full rectangle — touching any face is fatal.
BLOCK_HITBOX_MARGIN = 0     # px (0 = no forgiveness, full rectangle kills)


# ── Obstacle spawning ─────────────────────────────────────────────────────────

# Where new obstacles are created — just off the right edge so they scroll in.
SPAWN_X      = SCREEN_W + 60   # px

# Random gap between consecutive obstacles (pixels of horizontal scroll).
# At 300 px/s:
#   GAP_MIN = 280 px  →  ~0.93 s reaction time  (medium difficulty)
#   GAP_MAX = 520 px  →  ~1.73 s reaction time  (easy / breathing room)
# Tighten GAP_MIN later to increase difficulty.
GAP_MIN      = 280          # px — tightest gap
GAP_MAX      = 520          # px — widest gap


# ── Rewards  (for the RL gym wrapper — ignored in human play mode) ────────────

# Tiny reward each step the agent survives.
# Encourages the agent to stay alive rather than doing nothing and dying fast.
REWARD_ALIVE  =  0.05       # per step

# Large penalty on death.
# Should outweigh any alive-reward benefit of risky behaviour.
REWARD_DEATH  = -10.0       # on collision