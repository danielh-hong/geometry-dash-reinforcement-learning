# Streamlit App

This folder contains a deployable Streamlit app for comparing Geometry Dash RL policies.

## Local run

From repository root:

1. Activate the project venv.
2. Install dependencies from requirements.txt.
3. Start Streamlit:

streamlit run Streamlit/app.py

## What the app supports

- Procedural level generation (difficulty 1-6)
- Triple-only and staircase-only generation
- Custom obstacle builder mode (single/double/triple/stair counts)
- Policy comparison in one click:
  - Baseline fixed-distance jumper
  - Baseline never-jump
  - PPO general model (logs_custom)
  - PPO triple-finetune model
  - REINFORCE final model

## Deploy to Streamlit Community Cloud

- Push this repo to GitHub.
- In Streamlit Community Cloud, create new app.
- Set main file path to: Streamlit/app.py
- Use Python 3.12.
- Add any missing dependencies if prompted.

After deployment you get a public app URL.
