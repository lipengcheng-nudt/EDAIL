import subprocess
import sys
import re

def run_wandb_sweep(conf):
    result = subprocess.run(['python', '-m', 'wandb', 'sweep', conf], capture_output=True, text=True)
    sweep = re.search(r'wandb agent (\S+/\S+/\S+)',result.stderr)
    print(sweep)
    if sweep:
        sweep_id = sweep.group(1)
        print(f"wandb agent {sweep_id}")
        subprocess.run(['python', '-m', 'wandb', 'agent', sweep_id])
    else:
        print("Failed to extract sweep ID from wandb sweep output")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    run_wandb_sweep(config_file)