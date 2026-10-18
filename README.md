# CPSC 4420 Assignment 2 Taxi Q Learning

This project trains a tabular Q-learning agent in Gymnasium's Taxi-v4
environment. The program runs the baseline, learning-rate experiment,
discount-factor experiment, starting-epsilon experiment, final evaluation,
and video recording.

## What the program creates

After the program finishes, a new `results` folder will contain:

- `baseline_training.png`
- `learning_rate_comparison.png`
- `discount_factor_comparison.png`
- `epsilon_comparison.png`
- `experiment_summary.csv`
- `final_summary.json`
- `best_q_table.npy`
- `best_agent.mp4`

## Windows setup

Open the extracted project folder in File Explorer. Click the address bar,
type `powershell`, and press Enter. Then run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python taxi_q_learning.py
```

If PowerShell blocks activation, run this once in the same window and then try
the activation command again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The experiment can take several minutes. Keep the terminal open until it says
`Finished. Results are in the results folder.`

## Q-learning method

The Q-table starts at zero. The agent uses an epsilon-greedy policy and updates
the selected value using:

`Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(next_state,a') - Q(s,a)]`

For terminal transitions, the future Q-value is zero. Final evaluation uses
100 episodes with exploration disabled.

## GitHub

After running the program, upload the code files and generated `results` folder
to a public GitHub repository. Do not upload the `.venv` folder.
