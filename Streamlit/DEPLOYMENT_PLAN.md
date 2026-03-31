# Geometry Dash RL Streamlit Deployment Plan

## Goal
Deploy a public Streamlit app with a shareable link where users can:
- Run multiple agent policies on procedurally generated levels.
- Compare baseline heuristic vs trained models.
- Switch obstacle presets (single spike, double spike, triple spike, staircase).
- Optionally build custom obstacle sequences.

## Current Codebase Capabilities (already present)
- Headless game stepping and normalized observations: Game/game.py
- Procedural generation with difficulty and special modes: Game/level_generator.py
- PPO model loading and inference path: Game/watch_ppo.py + Game/evaluate_ppo.py
- PPO model artifacts already present:
  - Game/logs_custom/checkpoints/ppo_policy_final.zip
  - Game/logs_triple_finetune/checkpoints/ppo_policy_final.zip
- REINFORCE model artifacts:
  - Game/logs/checkpoints/policy_final.pth

## Critical Fixes / Risks Identified
1. gym_env.py had syntax break in reward-shaping branch. Fixed in this session.
2. level_generator.py contains duplicate generate_triple_only definition. Keep only one to reduce confusion.
3. gym_env.py currently prints debug logs every step. Must gate behind a debug flag before production deployment.
4. Game/game.py uses fullscreen display when render=True. Streamlit path should run render=False and use frame rendering in app code.

## Deployment Architecture
1. Keep simulation logic in existing Game modules.
2. Add Streamlit-specific orchestration layer under Streamlit/:
   - app.py (UI and controls)
   - model_registry.py (discover and load available models)
   - policies.py (baseline policies + wrappers for PPO and REINFORCE)
   - episode_runner.py (headless step loop, metrics, optional frame snapshots)
   - obstacle_presets.py (single/double/triple/staircase/custom)
3. Run one episode per user action and stream metrics/preview frames.

## Public Modes for Users
1. Baseline policies
   - WaitOnly: never jump.
   - FixedDistanceJump: jump if nearest obstacle is within threshold band.
   - RhythmJump: periodic jump cadence baseline.
2. Trained policies
   - PPO-General: logs_custom/checkpoints/ppo_policy_final.zip
   - PPO-TripleFinetune: logs_triple_finetune/checkpoints/ppo_policy_final.zip
   - REINFORCE-Final: logs/checkpoints/policy_final.pth
3. Level presets
   - Procedural difficulty 1-5
   - Triple only
   - Staircase only
   - Manual obstacle builder (ordered list of chunk primitives)

## UI Layout (Streamlit)
1. Sidebar
   - Policy selector
   - Device selector (cpu)
   - Seed / random seed toggle
   - Difficulty / level length / action repeat
   - Preset selector (procedural/triple/staircase/custom)
   - Custom obstacle editor controls
2. Main area
   - Run episode button
   - Live status + progress bar
   - Episode KPIs (distance, reward, survived steps, completion)
   - Comparison table across selected models
   - Optional trajectory chart

## Hosting Strategy
Primary: Streamlit Community Cloud
- Repo includes Streamlit/ app files.
- Entry point: Streamlit/app.py
- Requirements from root requirements.txt plus streamlit and pillow.
- Expected public URL pattern after deploy: https://<your-app-name>.streamlit.app

Fallbacks (if pygame wheel/system deps fail)
- Hugging Face Spaces (Streamlit runtime)
- Render (Docker + Streamlit)

## Implementation Milestones
1. Phase 1 - App skeleton and model registry
2. Phase 2 - Baseline policies and level preset routing
3. Phase 3 - Episode runner + metrics table + charts
4. Phase 4 - Custom obstacle builder UI
5. Phase 5 - Harden logs, remove debug spam, add tests
6. Phase 6 - Deploy to Streamlit Cloud and publish link

## Validation Checklist
- App boots in a clean environment.
- All listed models load without crash.
- Procedural generation works across difficulty presets.
- Baselines and trained policies produce different outcomes.
- One-click reproducibility with fixed seed.
- Public deployment URL reachable.

## First Recommended Default Release
- Policies exposed: FixedDistanceJump, PPO-General, PPO-TripleFinetune.
- Presets exposed: difficulty 1-3, triple only, staircase only.
- Custom builder as beta feature.
