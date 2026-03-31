from __future__ import annotations

# pyright: reportMissingImports=false

import random
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

# Ensure project modules can be imported when launched from repo root.
ROOT_DIR = Path(__file__).resolve().parents[1]
GAME_DIR = ROOT_DIR / "Game"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))

from Game import constants as C
from Game.game import Game
from Game.level_generator import LevelGenerator


@dataclass
class RunConfig:
    mode: str
    difficulty: int
    level_length: int
    seed: int
    max_steps: int
    threshold_px: int
    count_single: int
    count_double: int
    count_triple: int
    count_staircase: int


@dataclass
class ModelEntry:
    key: str
    label: str
    kind: str
    path: Path | None = None


MODE_HELP = {
    "procedural": "Mixed procedural chunks from LevelGenerator (best generalization view).",
    "triple_only": "Only triple-spike style patterns (hard timing specialization).",
    "staircase_only": "Stair-heavy sections to stress vertical platform control.",
    "rhythm_only": "Timing-heavy rhythm sections (spike gates, alternating spike motifs, limited verticality).",
    "spike_only": "Spike motifs only, no stairs/blocks; pure jump-timing stress test.",
    "custom_builder": "You choose how many single/double/triple/stair chunks to include.",
}

MODEL_HELP = {
    "baseline_fixed": "Heuristic baseline. Jumps when nearest obstacle is within a distance threshold.",
    "baseline_wait": "Control baseline. Never jumps, useful to show task difficulty.",
    "ppo_general": "Main PPO policy trained on broader procedural distribution.",
    "ppo_triple": "PPO finetuned for triple-spike motifs.",
    "ppo_saved_1218": "Experimental PPO checkpoint from saved_parameters.",
    "ppo_saved_1225": "Experimental PPO checkpoint from saved_parameters.",
    "ppo_saved_1323": "Experimental PPO checkpoint from saved_parameters.",
    "ppo_saved_9949": "Experimental PPO checkpoint from saved_parameters.",
}


def _model_registry() -> list[ModelEntry]:
    entries = [
        ModelEntry("baseline_fixed", "Baseline: Fixed-Distance Jump", "baseline"),
        ModelEntry("baseline_wait", "Baseline: Never Jump", "baseline"),
        ModelEntry("ppo_general", "PPO General (logs_custom final)", "ppo", GAME_DIR / "logs_custom" / "checkpoints" / "ppo_policy_final.zip"),
        ModelEntry("ppo_triple", "PPO Triple Finetune (final)", "ppo", GAME_DIR / "logs_triple_finetune" / "checkpoints" / "ppo_policy_final.zip"),
    ]

    # Optional experimental checkpoints (only shown if present on disk).
    optional_models = [
        ModelEntry("ppo_saved_1218", "PPO Experimental A (2026-03-19)", "ppo", GAME_DIR / "saved_parameters" / "2026_03_22_12_18_PPO.zip"),
        ModelEntry("ppo_saved_1225", "PPO Experimental B (2026-03-20)", "ppo", GAME_DIR / "saved_parameters" / "2026_03_22_12_25_PPO.zip"),
        ModelEntry("ppo_saved_1323", "PPO Experimental C (2026-03-21)", "ppo", GAME_DIR / "saved_parameters" / "2026_03_22_13_23_PPO.zip"),
        ModelEntry("ppo_saved_9949", "PPO Experimental D (2026-03-22)", "ppo", GAME_DIR / "saved_parameters" / "2026_03_22_99_49_PPO.zip"),
    ]

    entries.extend([entry for entry in optional_models if entry.path is not None and entry.path.exists()])
    return entries


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .hero {
            padding: 0.9rem 1.1rem;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(23,31,58,0.95), rgba(8,65,72,0.95));
            border: 1px solid rgba(130,180,255,0.35);
            margin-bottom: 0.8rem;
        }
        .hero-title {
            font-size: 1.28rem;
            font-weight: 700;
            color: #e8f3ff;
            margin: 0;
        }
        .hero-sub {
            color: #bcd6ff;
            margin-top: 0.2rem;
            font-size: 0.95rem;
        }
        .chip {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            margin-right: 0.45rem;
            margin-top: 0.45rem;
            background: rgba(50, 100, 180, 0.7);
            border: 2px solid rgba(100, 180, 255, 0.8);
            color: #ffffff;
            font-size: 0.88rem;
            font-weight: 600;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_info_panel(selected_entries: list[ModelEntry], cfg: RunConfig) -> None:
    st.markdown(
        """
        <div class="hero">
            <p class="hero-title">Geometry Dash RL Playground</p>
            <p class="hero-sub">Watch agents play, compare policy behavior, and understand each level mode quickly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chips = [
        f"<span class='chip'>mode: {cfg.mode}</span>",
        f"<span class='chip'>difficulty: {cfg.difficulty}</span>",
        f"<span class='chip'>seed: {cfg.seed}</span>",
        f"<span class='chip'>length: {cfg.level_length}px</span>",
    ]
    st.markdown("".join(chips), unsafe_allow_html=True)

    with st.expander("What each thing means", expanded=False):
        st.markdown("### Level Modes")
        for mode_key, mode_desc in MODE_HELP.items():
            st.write(f"- **{mode_key}**: {mode_desc}")

        st.markdown("### Models")
        for entry in selected_entries:
            st.write(f"- **{entry.label}**: {MODEL_HELP.get(entry.key, 'No description yet.')}")

        st.markdown("### Key Settings Explained")
        st.write("""
        **Level length (3000-18000 px)**: How far horizontally the level extends. 
        - **Longer = more obstacles total** because the generator packs obstacles to fill that distance. 
        - 9000px is typical; 3000px is short/hard, 18000px is very long/hard (more obstacles to navigate).
        
        **Max steps (500-4000)**: How many game frames the AI is allowed to play before we stop and score it.
        - This is a **time limit**, not distance. The AI still needs to reach as far as possible within this limit.
        - **Completed**: True if AI uses all max_steps without dying (rare = very good).
        
        **Difficulty (1-6)**: Controls game speed and chunk complexity.
        - Higher = faster gameplay + harder patterns. This affects how quickly obstacles approach.
        
        **Seed**: Controls the random obstacle layout. Same seed = identical level every time.
        """)

        st.markdown("### Metrics")
        st.write("""
        - **DistancePx**: Horizontal pixels reached before death or step timeout.
        - **Reward**: Cumulative environment reward (combo of distance bonuses, death penalties).
        - **Steps**: Number of game frames used before death/timeout.
        - **Jumps**: How many times the AI pressed "jump".
        - **Completed**: Did the AI survive all max_steps? (Very hard to achieve - usually dies before limit.)
        - **ObstacleCount**: Total obstacles in this generated level.
        """)


@st.cache_resource(show_spinner=False)
def _load_ppo_model(model_path: str):
    from stable_baselines3 import PPO

    return PPO.load(model_path, device="cpu")


def _adapt_obs_for_ppo(model, obs_norm):
    expected_shape = getattr(model.observation_space, "shape", None)
    expected_dim = int(expected_shape[0]) if expected_shape else len(obs_norm)
    if len(obs_norm) == expected_dim:
        return obs_norm
    if len(obs_norm) < expected_dim:
        return obs_norm + [0.0] * (expected_dim - len(obs_norm))
    return obs_norm[:expected_dim]


def _make_policy(entry: ModelEntry, threshold_px: int) -> Callable[[list[float], dict], int]:
    if entry.kind == "baseline":
        if entry.key == "baseline_wait":
            return lambda obs_norm, state: 0

        def baseline_fixed(obs_norm: list[float], _state: dict) -> int:
            rel_x_norm = obs_norm[4] if len(obs_norm) > 4 else 1.0
            rel_x_px = rel_x_norm * 784.0
            return 1 if 0.0 < rel_x_px <= float(threshold_px) else 0

        return baseline_fixed

    if entry.path is None or not entry.path.exists():
        raise FileNotFoundError(f"Model file not found: {entry.path}")

    if entry.kind == "ppo":
        model = _load_ppo_model(str(entry.path))

        def ppo_policy(obs_norm: list[float], _state: dict) -> int:
            adapted = _adapt_obs_for_ppo(model, obs_norm)
            action, _ = model.predict(adapted, deterministic=True)
            return int(action)

        return ppo_policy

    raise ValueError(f"Unknown model kind: {entry.kind}")


def _build_custom_obstacles(cfg: RunConfig) -> list[dict]:
    rng = random.Random(cfg.seed)

    chunks: list[str] = []
    chunks.extend(["single"] * cfg.count_single)
    chunks.extend(["double"] * cfg.count_double)
    chunks.extend(["triple"] * cfg.count_triple)
    chunks.extend(["stair"] * cfg.count_staircase)

    if not chunks:
        chunks = ["single", "double", "triple", "stair"]

    rng.shuffle(chunks)

    x = float(C.SCREEN_W + C.GAME_SPEED * 1.5)
    obstacles: list[dict] = []

    def add_spike_run(count: int):
        nonlocal x
        for i in range(count):
            obstacles.append(
                {
                    "type": "spike",
                    "x": float(x + i * C.BLOCK_SIZE),
                    "y": float(C.GROUND_Y - C.SPIKE_H),
                    "w": float(C.SPIKE_W),
                    "h": float(C.SPIKE_H),
                }
            )
        x += float(count * C.BLOCK_SIZE)

    def add_staircase():
        nonlocal x
        # 3-step staircase with 3-block landings (safe geometry)
        for step in range(1, 4):
            for wcol in range(3):
                bx = x + (step - 1) * 3 * C.BLOCK_SIZE + wcol * C.BLOCK_SIZE
                for hrow in range(step):
                    by = C.GROUND_Y - (hrow + 1) * C.BLOCK_SIZE
                    obstacles.append(
                        {
                            "type": "block",
                            "x": float(bx),
                            "y": float(by),
                            "w": float(C.BLOCK_W),
                            "h": float(C.BLOCK_H),
                        }
                    )
        x += float(9 * C.BLOCK_SIZE)

    for chunk in chunks:
        if chunk == "single":
            add_spike_run(1)
        elif chunk == "double":
            add_spike_run(2)
        elif chunk == "triple":
            add_spike_run(3)
        else:
            add_staircase()

        x += float(rng.randint(int(0.4 * C.GAME_SPEED), int(0.9 * C.GAME_SPEED)))
        if x > cfg.level_length + C.SCREEN_W:
            break

    obstacles.sort(key=lambda o: o["x"])
    return obstacles


def _build_level(cfg: RunConfig) -> list[dict]:
    gen = LevelGenerator(difficulty=cfg.difficulty, seed=cfg.seed, progressive=True)
    if cfg.mode == "triple_only":
        return gen.generate_triple_only(length=cfg.level_length)
    if cfg.mode == "staircase_only":
        return gen.generate_staircase_only(length=cfg.level_length)
    if cfg.mode == "rhythm_only":
        return gen.generate_rhythm_only(length=cfg.level_length)
    if cfg.mode == "spike_only":
        return gen.generate_spike_only(length=cfg.level_length)
    if cfg.mode == "custom_builder":
        return _build_custom_obstacles(cfg)
    return gen.generate(length=cfg.level_length)


def _run_episode(entry: ModelEntry, cfg: RunConfig, prebuilt_obstacles: list[dict] | None = None) -> dict:
    policy = _make_policy(entry, cfg.threshold_px)
    obstacles = prebuilt_obstacles if prebuilt_obstacles is not None else _build_level(cfg)

    game = Game(render=False, seed=cfg.seed, debug=False, agent_policy=None)
    game.load_level(obstacles)

    done = False
    total_reward = 0.0
    jumps = 0
    steps = 0

    while not done and steps < cfg.max_steps:
        obs_norm = game.get_normalized_observation()
        action = int(policy(obs_norm, {"step": steps}))
        jumps += int(action == 1)
        _, reward, done = game.step(action)
        total_reward += float(reward)
        steps += 1

    distance_px = int(game._scroll_x)
    completion = (not done) and (steps >= cfg.max_steps)
    game.close()

    return {
        "Model": entry.label,
        "Mode": cfg.mode,
        "Seed": cfg.seed,
        "DistancePx": distance_px,
        "Reward": round(total_reward, 3),
        "Steps": steps,
        "Jumps": jumps,
        "Completed": completion,
        "ObstacleCount": len(obstacles),
    }


def _render_frame(game: Game, scale: float = 0.35) -> Image.Image:
    width = max(1, int(C.SCREEN_W * scale))
    height = max(1, int(C.SCREEN_H * scale))
    ground_y = int(C.GROUND_Y * scale)

    img = Image.new("RGB", (width, height), C.BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Subtle top-to-bottom sky gradient for better readability.
    for y in range(0, ground_y):
        t = y / max(1, ground_y)
        col = (
            int(22 + 16 * t),
            int(24 + 22 * t),
            int(42 + 26 * t),
        )
        draw.line([(0, y), (width, y)], fill=col)

    # Light grid lines for motion cues.
    for gy in range(0, ground_y, max(10, int(60 * scale))):
        draw.line([(0, gy), (width, gy)], fill=(46, 54, 76))

    # Ground panel
    draw.rectangle([(0, ground_y), (width, height)], fill=C.GROUND_COLOR)

    # Obstacles
    for obs in game.obstacles:
        x1 = int(obs.x * scale)
        y1 = int(obs._y * scale)
        x2 = int((obs.x + obs.w) * scale)
        y2 = int((obs._y + obs.h) * scale)

        if getattr(obs, "kind", "") == "spike":
            tip_x = int((obs.x + obs.w * 0.5) * scale)
            tip_y = y1
            draw.polygon([(x1, y2), (x2, y2), (tip_x, tip_y)], fill=C.SPIKE_COLOR)
        else:
            draw.rectangle([(x1, y1), (x2, y2)], fill=C.BLOCK_COLOR)

    # Player
    px1 = int(game.player.x * scale)
    py1 = int(game.player.y * scale)
    px2 = int((game.player.x + C.PLAYER_SIZE) * scale)
    py2 = int((game.player.y + C.PLAYER_SIZE) * scale)
    draw.rectangle([(px1, py1), (px2, py2)], fill=C.PLAYER_COLOR)

    # HUD text with dark background for readability
    hud_text = f"Distance: {int(game._scroll_x)} px"
    bbox = draw.textbbox((10, 10), hud_text)
    pad = 8
    bg_box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(bg_box, fill=(20, 20, 25, 200))  # Semi-transparent dark background
    draw.text((10, 10), hud_text, fill=(255, 255, 255))  # Bright white text
    return img


def _run_visual_episode(
    entry: ModelEntry,
    cfg: RunConfig,
    fps: int,
    max_visual_steps: int,
    render_scale: float,
    sim_steps_per_frame: int,
    render_every_n: int,
    show_visual_feedback: bool,
) -> dict:
    policy = _make_policy(entry, cfg.threshold_px)
    obstacles = _build_level(cfg)

    game = Game(render=False, seed=cfg.seed, debug=False, agent_policy=None)
    game.load_level(obstacles)

    frame_slot = st.empty() if show_visual_feedback else None
    stats_slot = st.empty() if show_visual_feedback else None
    progress = st.progress(0) if show_visual_feedback else None
    done = False
    total_reward = 0.0
    steps = 0
    visual_ticks = 0
    last_action = 0

    # Watch mode has its own dedicated step cap independent from compare mode.
    target_steps = max_visual_steps
    while not done and steps < target_steps:
        for _ in range(max(1, sim_steps_per_frame)):
            if done or steps >= target_steps:
                break
            obs_norm = game.get_normalized_observation()
            action = int(policy(obs_norm, {"step": steps}))
            last_action = action
            _, reward, done = game.step(action)
            total_reward += float(reward)
            steps += 1

        visual_ticks += 1
        if show_visual_feedback and (visual_ticks % max(1, render_every_n) == 0 or done or steps >= target_steps):
            frame = _render_frame(game, scale=render_scale)
            frame_slot.image(
                frame,
                caption=f"{entry.label} | step {steps} | action={'JUMP' if last_action == 1 else 'WAIT'}",
                use_container_width=True,
            )
            stats_slot.write(
                {
                    "model": entry.label,
                    "distance_px": int(game._scroll_x),
                    "steps": steps,
                    "reward": round(total_reward, 3),
                    "done": done,
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                }
            )

        if show_visual_feedback:
            progress.progress(min(100, int((steps / max(1, target_steps)) * 100)))

        if show_visual_feedback and fps > 0:
            time.sleep(1.0 / float(fps))

    final = {
        "model": entry.label,
        "distance_px": int(game._scroll_x),
        "steps": steps,
        "reward": round(total_reward, 3),
        "done": done,
    }
    game.close()
    return final


def _sidebar_controls() -> tuple[list[ModelEntry], RunConfig]:
    st.sidebar.header("⚙️ Simulation Setup")
    st.sidebar.caption("These options control WHAT and HOW the game is played")

    registry = _model_registry()
    selected_entries = list(registry)

    st.sidebar.markdown("### 🤖 Model Set")
    st.sidebar.caption(f"All available models are auto-included ({len(selected_entries)} total).")

    st.sidebar.divider()
    st.sidebar.markdown("### Level Design")

    mode = st.sidebar.selectbox(
        "Obstacle pattern",
        options=["procedural", "triple_only", "staircase_only", "rhythm_only", "spike_only", "custom_builder"],
        index=0,
        help="**procedural**: Mixed obstacles (best for general testing) | **triple_only**: Only hard triple-spike patterns | **staircase_only**: Only vertical platforms | **custom_builder**: You control obstacle counts.",
    )
    difficulty = st.sidebar.slider(
        "Difficulty (1-6)",
        min_value=1,
        max_value=6,
        value=3,
        help="Higher difficulty = faster game speed + more complex chunk patterns. Affects AI decision speed.",
    )
    level_length = st.sidebar.slider(
        "Level length (pixels)",
        min_value=3000,
        max_value=18000,
        value=9000,
        step=500,
        help="Horizontal distance the level must span. Longer = more obstacles. Doesn't change obstacle DENSITY, just total COUNT.",
    )

    st.sidebar.divider()
    st.sidebar.markdown("### Game Rules")

    random_seed = st.sidebar.checkbox(
        "Random seed (new level each time)",
        value=True,
        help="If unchecked, uses the same scene layout—great for comparing models on identical levels.",
    )
    seed = random.randint(1, 10_000_000) if random_seed else st.sidebar.number_input(
        "Fixed seed (if not random)", min_value=1, max_value=999_999_999, value=42,
        help="When random seed OFF: seed determines exact obstacle layout. Same seed = same level.",
    )
    max_steps = st.sidebar.slider(
        "Max simulation steps",
        min_value=500,
        max_value=4000,
        value=2500,
        step=100,
        help="How many game frames to allow before stopping (even if alive). Affects 'Completed' metric.",
    )
    threshold_px = st.sidebar.slider(
        "Baseline jump distance (px)",
        min_value=80,
        max_value=450,
        value=220,
        step=10,
        help="Used by Baseline: Fixed-Distance Jump. Increase = jumps earlier, decrease = jumps later.",
    )

    count_single = 10
    count_double = 8
    count_triple = 8
    count_staircase = 4

    if mode == "custom_builder":
        st.sidebar.divider()
        st.sidebar.markdown("### Custom Build")
        st.sidebar.caption("Choose how many of each obstacle type to use")
        count_single = st.sidebar.slider("Single spikes", min_value=0, max_value=30, value=10, help="1-spike patterns")
        count_double = st.sidebar.slider("Double spikes", min_value=0, max_value=30, value=8, help="2-spike patterns")
        count_triple = st.sidebar.slider("Triple spikes", min_value=0, max_value=30, value=8, help="3-spike patterns")
        count_staircase = st.sidebar.slider("Staircases", min_value=0, max_value=20, value=4, help="Vertical platform sections")

    cfg = RunConfig(
        mode=mode,
        difficulty=int(difficulty),
        level_length=int(level_length),
        seed=int(seed),
        max_steps=int(max_steps),
        threshold_px=int(threshold_px),
        count_single=int(count_single),
        count_double=int(count_double),
        count_triple=int(count_triple),
        count_staircase=int(count_staircase),
    )

    return selected_entries, cfg


def main() -> None:
    st.set_page_config(page_title="Geometry Dash RL Playground", layout="wide")
    _inject_styles()

    selected_entries, cfg = _sidebar_controls()
    _render_info_panel(selected_entries, cfg)

    if not selected_entries:
        st.warning("Select at least one model to run.")
        return

    all_labels = [entry.label for entry in selected_entries]
    ppo_general_label = next((label for label in all_labels if "PPO General" in label), None)
    baseline_label = next((label for label in all_labels if "Baseline: Fixed-Distance Jump" in label), None)

    compare_default_labels: list[str] = []
    if ppo_general_label is not None:
        compare_default_labels.append(ppo_general_label)
    if baseline_label is not None:
        compare_default_labels.append(baseline_label)
    if not compare_default_labels:
        compare_default_labels = all_labels[: min(2, len(all_labels))]

    watch_default_index = all_labels.index(ppo_general_label) if ppo_general_label in all_labels else 0

    watch_tab, run_tab = st.tabs(["Watch Live Playback", "Compare Models"])

    with run_tab:
        compare_labels = st.multiselect(
            "Models for this comparison",
            options=all_labels,
            default=compare_default_labels,
            help="Pick the PPO models you want in this specific comparison run.",
        )
        compare_entries = [entry for entry in selected_entries if entry.label in compare_labels]

        if st.button("Run Episode Comparison", type="primary"):
            if not compare_entries:
                st.warning("Select at least one model in 'Models for this comparison'.")
                return

            rows = []
            errors = []
            compare_progress = st.progress(0)
            compare_status = st.empty()
            with st.spinner("Running selected models..."):
                # Build one deterministic level and reuse it across models for fair,
                # faster comparisons.
                shared_obstacles = _build_level(cfg)
                total_models = max(1, len(compare_entries))
                for idx, entry in enumerate(compare_entries, start=1):
                    compare_status.caption(f"Running {entry.label} ({idx}/{total_models})")
                    try:
                        rows.append(_run_episode(entry, cfg, prebuilt_obstacles=shared_obstacles))
                    except Exception as exc:
                        errors.append(f"{entry.label}: {exc}")
                    compare_progress.progress(int((idx / total_models) * 100))

            compare_status.empty()
            compare_progress.empty()

            if rows:
                df = pd.DataFrame(rows).sort_values(by="DistancePx", ascending=False)
                st.subheader("Results")

                c1, c2, c3 = st.columns(3)
                c1.metric("Top Distance", f"{int(df.iloc[0]['DistancePx'])} px")
                c2.metric("Top Model", str(df.iloc[0]["Model"]))
                c3.metric("Avg Distance", f"{int(df['DistancePx'].mean())} px")

                st.dataframe(df, use_container_width=True)

                winner = df.iloc[0]
                st.success(f"Top model: {winner['Model']} with distance {winner['DistancePx']} px")

            if errors:
                st.subheader("Model Load/Run Issues")
                for err in errors:
                    st.error(err)

    with watch_tab:
        st.subheader("👀 Live Visual Playback")
        st.caption("**These settings control DISPLAY speed, not game speed.** Adjust to balance smoothness vs fast simulation.")

        col_info = st.container()
        with col_info.expander("What do these settings do?", expanded=False):
            st.markdown("""
            **Playback FPS**: Sleep time between drawing frames. 0 = no delays (renders as fast as possible).
            
            **Sim steps per visual tick**: How many game frames to run before drawing. Higher = faster but more jittery.
            
            **Render every N ticks**: Skip N frames between redraws. Combine with FPS to control visual smoothness.
            
            **Example presets:**
            - **Smooth (24 FPS, every 2, sim 5)**: Buttery smooth playback, slower simulation
            - **Balanced (12 FPS, every 4, sim 8)**: Good mix
            - **Turbo (0 FPS, every 12, sim 15)**: Fastest sim, looks choppy
            """)

        visual_label = st.selectbox(
            "🎮 Model to watch",
            options=all_labels,
            index=watch_default_index,
            help="Select which AI to visualize playing the level.",
        )

        st.markdown("### Display Speed Controls")
        show_visual_feedback = st.checkbox(
            "Show visual feedback (frames + live stats)",
            value=True,
            help="Turn this off to run a fast simulation-only watch run without rendering frames.",
        )

        c1, c2, c3 = st.columns(3)
        fps = c1.slider(
            "Playback FPS",
            min_value=0,
            max_value=60,
            value=24,
            help="Frames per second. 0 = turbo (no sleep delays). Higher = delayed display.",
        )
        render_every_n = c2.slider(
            "Render every N",
            min_value=1,
            max_value=20,
            value=2,
            help="Skip frames: 1=every frame, 2=every 2nd, etc. Saves rendering work.",
        )
        sim_steps_per_frame = c3.slider(
            "Sim steps/tick",
            min_value=1,
            max_value=30,
            value=5,
            help="Run game this many times before showing. Higher = faster but choppier animation.",
        )

        st.markdown("### Simulation Length")
        c4, c5 = st.columns(2)
        max_visual_steps = c4.slider(
            "Max steps to show",
            min_value=100,
            max_value=20000,
            value=2500,
            step=100,
            help="Stop playback after this many game steps (independent of FPS settings).",
        )
        render_scale = c5.slider(
            "Display size",
            min_value=0.15,
            max_value=0.60,
            value=0.30,
            step=0.05,
            help="Scale game view (pixel-dropped rendering). Smaller = faster to draw.",
        )

        if st.button("▶️ Run Live Visual Playback", type="primary"):
            try:
                visual_entry = next((entry for entry in selected_entries if entry.label == visual_label), None)
                if visual_entry is None:
                    st.error(f"Model '{visual_label}' not found. Please reselect from sidebar.")
                else:
                    with st.spinner(f"Loading {visual_entry.label} and starting playback..."):
                        final = _run_visual_episode(
                            visual_entry,
                            cfg,
                            fps=fps,
                            max_visual_steps=max_visual_steps,
                            render_scale=render_scale,
                            sim_steps_per_frame=sim_steps_per_frame,
                            render_every_n=render_every_n,
                            show_visual_feedback=show_visual_feedback,
                        )
                    if not show_visual_feedback:
                        st.success(
                            f"Fast run complete: {final['model']} | distance={final['distance_px']} px | "
                            f"steps={final['steps']} | reward={final['reward']}"
                        )
            except Exception as exc:
                st.error(f"Playback error: {exc}")


if __name__ == "__main__":
    main()
