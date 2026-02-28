# ── Display ──────────────────────────────────────────────────────────────────
SCREEN_W      = 800
SCREEN_H      = 400
FPS           = 60
WINDOW_TITLE  = "Geometry Dash RL"

# ── Colours ───────────────────────────────────────────────────────────────────
BG_COLOR      = (20,  20,  40)
GROUND_COLOR  = (50,  50,  70)
GRID_COLOR    = (35,  35,  55)
PLAYER_COLOR  = (80,  180, 255)
SPIKE_COLOR   = (255, 80,  80)
BLOCK_COLOR   = (100, 220, 140)
HUD_COLOR     = (200, 200, 220)
WHITE         = (255, 255, 255)

# ── World ─────────────────────────────────────────────────────────────────────
GROUND_Y      = SCREEN_H - 70    # y-pixel of ground surface
GAME_SPEED    = 280.0            # horizontal scroll speed  (px / s)

# ── Player ────────────────────────────────────────────────────────────────────
PLAYER_X      = 130              # fixed screen x of the cube
PLAYER_SIZE   = 34               # square side (px)
GRAVITY       = 1750.0           # px / s²
JUMP_VEL      = -600.0           # initial upward velocity on jump  (negative = up)
HITBOX_MARGIN = 4                # shrink hitbox inward for fairness

# ── Obstacles ─────────────────────────────────────────────────────────────────
SPIKE_W       = 34
SPIKE_H       = 34
BLOCK_W       = 34
BLOCK_H       = 34

# Gap between consecutive obstacles (world px)
GAP_MIN       = 280
GAP_MAX       = 520

# Spawn x in screen coords (just off right edge)
SPAWN_X       = SCREEN_W + 60

# ── Scoring / reward ──────────────────────────────────────────────────────────
REWARD_ALIVE   =  0.05
REWARD_DEATH   = -10.0