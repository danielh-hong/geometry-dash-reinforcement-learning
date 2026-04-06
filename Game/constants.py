# =============================================================================
# constants.py
# =============================================================================
# ALL game configuration lives here. Never hardcode values in game.py.
# Change things here and they update everywhere automatically.
#
# PHYSICS SOURCES:
#   Block size / hitboxes : Measured from real GD. Roughly 17 blocks fit over 1920 px width, so 1 block ≈ 112 px. Player cube is exactly 1 block = 112 px square.
#   Speed                 : https://www.reddit.com/r/geometrydash/comments/1gtsoma/i_measured_the_exact_player_movement_speed_might/
#   Jump height           : https://gdforum.freeforums.net/thread/48749/p1kachu-presents-physics-geometry-dash
#   Spike hitbox          : Real GD spike kill zone ≈ inner 40% of triangle width
#                           (the pointy tip region); outer edges are forgiving
# =============================================================================


# ── Display ───────────────────────────────────────────────────────────────────

SCREEN_W     = 1920         # window width  in pixels
SCREEN_H     = 1200         # window height in pixels
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

# One "block" in real GD = 112 pixels in our simulation.
# Everything is derived from BLOCK_SIZE so the game stays proportional.
BLOCK_SIZE = 112

# Ground occupies 3 blocks at the bottom of the screen.
GROUND_HEIGHT_BLOCKS = 3

# Y-coordinate of the ground surface (top edge of the ground panel).
# The player cube sits with its bottom edge touching this line.
GROUND_Y     = SCREEN_H - (GROUND_HEIGHT_BLOCKS * BLOCK_SIZE)


# ── Game speed ────────────────────────────────────────────────────────────────

# In-game speed table (blocks / second).
SPEED_SLOW_BPS    = 8.37193493
SPEED_NORMAL_BPS  = 10.38592002
SPEED_FAST_BPS    = 12.91390686
SPEED_FASTER_BPS  = 15.59989392
SPEED_FASTEST_BPS = 19.19986483

# Default running speed for this simulation (normal speed).
GAME_SPEED   = SPEED_NORMAL_BPS * BLOCK_SIZE


# ── Player physics ────────────────────────────────────────────────────────────

# Cube is exactly 1 block = 112 px square, matching real GD's documented hitbox.
PLAYER_SIZE  = BLOCK_SIZE   # px — side length of the player cube (= 112)

# Player starts at block coordinate (6.5, 4) with the cube resting on ground.
# X is measured from the left edge in blocks.
PLAYER_START_BLOCK_X = 6.5

# Fixed screen x-position of the player. The world scrolls left; player stays put.
PLAYER_X     = PLAYER_START_BLOCK_X * BLOCK_SIZE

# Vertical physics values from reference data (in block-time units):
#   jump y_speed set to 1.94
#   gravity is -0.876 while y_speed > -2.6
#
# Unit definition:
#   1 block-time = time to move 1 block horizontally at reference speed.
#   So if reference speed is S blocks/s:
#     velocity (blocks/block-time)   × S     = blocks/s
#     acceleration (blocks/block-time²) × S² = blocks/s²
#
# Engine convention uses px/s with negative = upward.
PHYSICS_REF_SPEED_BPS = SPEED_NORMAL_BPS
JUMP_VEL_BLOCKTIME_UNITS       = 1.94
GRAVITY_BLOCKTIME_UNITS_MAG    = 0.876
MAX_FALL_SPEED_BLOCKTIME_UNITS = 2.6

# Converted to engine units.
GRAVITY      = GRAVITY_BLOCKTIME_UNITS_MAG * (PHYSICS_REF_SPEED_BPS ** 2) * BLOCK_SIZE  # px/s² (downward)
JUMP_VEL     = -JUMP_VEL_BLOCKTIME_UNITS * PHYSICS_REF_SPEED_BPS * BLOCK_SIZE             # px/s (upward)
MAX_FALL_SPEED = MAX_FALL_SPEED_BLOCKTIME_UNITS * PHYSICS_REF_SPEED_BPS * BLOCK_SIZE      # px/s (downward)

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

# Spike hitbox tuning.
# Horizontal: hitbox stays centered on the spike and uses this fraction of width.
# Vertical: top and bottom insets are controlled independently so the kill zone
# can be moved up/down and resized without changing spike art.
#
# Effective rectangle:
#   x = spike_x + (SPIKE_W * (1 - SPIKE_HITBOX_WIDTH_FRAC) / 2)
#   w = SPIKE_W * SPIKE_HITBOX_WIDTH_FRAC
#   y = spike_y + (SPIKE_H * SPIKE_HITBOX_TOP_INSET_FRAC)
#   h = SPIKE_H * (1 - top_inset - bottom_inset)
SPIKE_HITBOX_WIDTH_FRAC = 20 / 95
SPIKE_HITBOX_TOP_INSET_FRAC = 84/284
SPIKE_HITBOX_BOTTOM_INSET_FRAC = 84/284

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
GAP_MIN      = 450          # px — tightest gap (even less dense for human play)
GAP_MAX      = 850          # px — widest gap (even less dense for human play)


# ── Rewards  (for the RL gym wrapper — ignored in human play mode) ────────────

# No dense reward for just existing, to prevent stalling/pointless jumping exploits.
REWARD_ALIVE  =  0.0        # per step

# Reward strictly for passing an obstacle successfully.
REWARD_CLEAR  =  1.0        # per cleared obstacle

# Standardized death penalty.
REWARD_DEATH  = -1.0        # on collision