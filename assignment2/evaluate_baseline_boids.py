import torch
import wandb
import os
import glob

from modules.boids_model_baseline import BoidsBaselineModel, BoidsBaselineDataset
from modules.evaluation import Evaluator

from types import SimpleNamespace
from pathlib import Path

# Change to script directory to ensure relative paths work
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

args = {
    "evaluation_step_sizes": [5],
    "problem": "boids",
    "prior_conditioning": False,  # Not used for baseline, but required by Evaluator
    "euler_steps": 0,  # Not used for baseline
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sigma": 0,  # Not used for baseline
    "use_tqdm": True,
    "wandb_api_key": "a638f884eba75c05c95730b04b2f27f7260503bb",  # Update with your key or use wandb.login file
}
args = SimpleNamespace(**args)

# Initialize datasets
train_files = sorted(glob.glob('../data/boids/raw/boids_0*.npy'))[:15]
val_files = sorted(glob.glob('../data/boids/raw/boids_*.npy'))[16:25]

train_dataset = BoidsBaselineDataset(train_files, timesample=1)
val_dataset = BoidsBaselineDataset(val_files, timesample=1)

# Initialize model
device = torch.device(args.device)
# Initialize with parameters matching the trained model (64, 64, 64, 4)
model = BoidsBaselineModel(h_dim=64, m_dim=64, hidden_dim=64, layers=4).to(device)

# Load model weights
model_path = Path(__file__).resolve().parent / 'models' / 'best_baseline_boids.pt'
if model_path.exists():
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {model_path}")
else:
    print(f"WARNING: No model found at {model_path}")
    print("Please provide a trained boids model or adjust the model_path variable.")
    print("Proceeding with untrained model for demonstration purposes.")

# Initialize Weights and Biases
if os.path.exists('wandb.login'):
    with open('wandb.login', 'r') as f:
        wandb.login(key=f.read().strip())
elif args.wandb_api_key is not None:
    wandb.login(key=args.wandb_api_key)
else:
    print("No Weights and Biases API key provided.")

wandb.init(
    entity="atai-apple-juice", 
    project=f"{args.problem}_baseline", 
    config=args,
    tags=[f'{args.problem}_baseline']
)

# Evaluate the baseline model
evaluator = Evaluator(
    model,
    train_dataset,
    val_dataset,
    args,
    baseline=True
)

print("Evaluating at different step sizes...")
for step_size in args.evaluation_step_sizes:
    print(f"Evaluating at step size {step_size}...")
    evaluator.evaluate_step_size(step_size)

print("Evaluating full trajectories...")
evaluator.evaluate_trajectories()

# Finish Weights and Biases
wandb.finish()
print("Evaluation complete!")

