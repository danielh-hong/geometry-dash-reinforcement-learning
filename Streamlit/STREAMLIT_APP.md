# Geometry Dash RL Playground
# streamlit run Streamlit/app.py

An interactive Streamlit application to visualize, compare, and analyze reinforcement learning agents playing Geometry Dash.

## What It Does

This app lets you:
- **Compare multiple AI models side-by-side** on the same level
- **Watch live playback** of agents navigating procedurally generated levels
- **Adjust game parameters** (difficulty, level layout, obstacles)
- **Test different strategies** (baselines vs. trained policies)
- **Analyze metrics** (distance, reward, jumps, completion rate)

## Quick Start

```powershell
# From workspace root
.\.venv\Scripts\Activate.ps1
streamlit run Streamlit/app.py
```

Opens: `http://localhost:8501`

## Available Models

### Baselines (Heuristic)
- **Baseline: Fixed-Distance Jump** → Jumps when nearest obstacle is within threshold distance
- **Baseline: Never Jump** → Control baseline (shows task difficulty)

### Trained Policies (PPO)
- **PPO General** (`logs_custom/checkpoints/ppo_policy_final.zip`)  
  - Trained on mixed procedural obstacles for generalization
- **PPO Triple Finetune** (`logs_triple_finetune/checkpoints/ppo_policy_final.zip`)  
  - Finetuned on triple-spike patterns for specialization

### Legacy
- **REINFORCE Final** (`logs/checkpoints/policy_final.pth`)  
  - Older policy for historical comparison

## Features

### Watch Live Playback Tab (Primary)
- Visualize a single agent playing in real-time (main feature)
- Control display speed independently of game speed
- Adjust FPS, frame skip, and rendering resolution
- Watch agent navigate obstacles step-by-step with live stats
- Smooth playback with configurable rendering

### Compare Models Tab
- Select multiple models to run on identical levels
- Sorts results by distance achieved
- Shows metrics: distance (px), reward, steps, jumps, completion
- Compare policies on same seed for direct comparison
- Batch evaluate multiple models at once

### Level Customization
- **Procedural** – Mixed obstacles (default, best for generalization)
- **Triple-Only** – Hard triple-spike patterns
- **Staircase-Only** – Vertical platform challenges
- **Custom Builder** – Choose your own obstacle mix

### Game Settings
- **Difficulty** (1–6) – Controls speed and pattern complexity
- **Level Length** (3000–18000 px) – Total obstacles to navigate
- **Max Steps** (500–4000) – Time limit before stopping
- **Seed** – Fixed or random for reproducibility
- **Baseline Jump Distance** – Tuning parameter for heuristic models

## Observation Space

The normalized observation vector fed to each model contains **28 floats**:

```
[Player State (3)]
  - player_y: vertical position [0, 1]
  - player_vy: velocity [-1, 1]
  - on_ground: binary jump availability

[Upcoming Obstacles (24 = 3 obstacles × 8 features)]
  Per obstacle:
    - type: 0=spike, 1=block
    - rel_x: horizontal distance [0, 1]
    - rel_y: vertical offset [-1, 1]
    - width: normalized size
    - height: normalized size
    - time_to_reach: seconds [0, 1]
    - gap_top: clearance above [0, 1]
    - gap_bottom: clearance below [0, 1]

[Derived (1)]
  - is_jump_possible: duplicate of on_ground
```

## Performance Notes

- **First model load**: ~2–5 seconds (PPO models from disk)
- **Baseline models**: Instant (no file I/O)
- **Visual playback**: Configurable (1–60 FPS)
- **Caching**: Models cached after first load per session

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PPO model not found | Verify `logs_custom/checkpoints/ppo_policy_final.zip` exists |
| REINFORCE crashes | Check `logs/checkpoints/policy_final.pth` is readable |
| App is slow | Lower FPS, increase frame skip, reduce render size |
| Observation mismatch error | Observation must be exactly 28 values; check game.py |
| Watch tab shows no models | Select at least one model in the sidebar first |
| "Model not found" error on playback | Reselect model from watches dropdown or check sidebar |

## Verification Checklist

✅ **Sidebar control** – All controls (difficulty, level, seed, etc.) work and flow correctly  
✅ **Model registry** – All 5 models load without import errors  
✅ **Watch Live Playback** – Renders correctly, updates stats in real-time, handles model changes safely  
✅ **Compare Models** – Runs episodes, sorts by distance, displays results table  
✅ **Error handling** – Model load failures caught, playback errors caught, user-friendly messages  
✅ **100% Functional** – App is production-ready with defensive error handling

## File Structure

```
Streamlit/
├── app.py                 # Main Streamlit app
├── STREAMLIT_APP.md      # This file
└── README.md             # Additional info
```

## Architecture

The app uses a **policy factory pattern**:

1. **ModelEntry registry** – Metadata about each model (path, type)
2. **_make_policy()** – Factory function that creates model-specific prediction functions
3. **_run_episode()** – Executes one game without rendering
4. **_run_visual_episode()** – Executes with frame-by-frame rendering to st.empty() slots

### Model Types

- `baseline` → Simple heuristic rules (distance threshold)
- `ppo` → Stable-Baselines3 PPO (loads .zip)
- `reinforce` → Custom torch-based REINFORCE (loads .pth)

### Error Handling

- **Model loading failures** → Caught and displayed per-model in Compare tab
- **Model selection mismatches** → Safe `next()` with fallback in Watch tab
- **Playback exceptions** → Caught and displayed to user with context

## Development

To add a new model:

1. Add entry to `_model_registry()` in `app.py`
2. Ensure model file exists at the specified path
3. Set `kind` to `baseline`, `ppo`, or `reinforce`
4. Test with "Run Episode Comparison"

### Adding Custom Model Type

If you want a new model type (e.g., DQN):

```python
ModelEntry("dqn_model", "DQN Trained", "dqn", path_to_model)
```

Then add a handler in `_make_policy()`:
```python
if entry.kind == "dqn":
    model = _load_dqn_model(str(entry.path))
    def dqn_policy(obs_norm, _state):
        return int(model.predict(obs_norm))
    return dqn_policy
```

## Example Workflows

### Compare Training Progress
- Select "PPO General" at different checkpoints
- Fixed seed to compare on identical level
- Observe distance/reward improvement over time

### Baseline Tuning
- Select "Baseline: Fixed-Distance"
- Adjust threshold slider (80–400 px)
- Run episodes and find optimal threshold

### Agent Specialization
- Compare "PPO General" vs "PPO Triple Finetune"
- Set level to "triple_only"
- Observe how specialists outperform generalists

## References

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Streamlit**: https://docs.streamlit.io/
- **Gymnasium**: https://gymnasium.farama.org/
