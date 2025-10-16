import torch
import wandb
import os

from modules.cfd_model_baseline import CFDBaselineModel, CFDBaselineDataset
from modules.evaluation import Evaluator

from types import SimpleNamespace
from pathlib import Path

# Change to script directory to ensure relative paths work
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

args = {
    "evaluation_step_sizes": [5, 20, 40],
    "problem": "cfd",
    "prior_conditioning": True,
    "euler_steps": 0,
    "device": "cuda",
    "sigma": 0,
    "use_tqdm": False,
    "wandb_api_key": "a638f884eba75c05c95730b04b2f27f7260503bb",
}
args = SimpleNamespace(**args)

# Initialize datasets
datafolder_path = Path(__file__).resolve().parents[1] / 'data' / 'CFD' / 'grid' / 'concat'
train_files = list(sorted(datafolder_path.glob('uvp_grid_Re*.npy')))
val_files = list(sorted(datafolder_path.glob('uvp_grid_Re*.npy')))
train_dataset = CFDBaselineDataset(train_files, flip_augmentation=False, timesample=20)
val_dataset = CFDBaselineDataset(val_files, flip_augmentation=False, timesample=20)


# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CFDBaselineModel(in_channels=4, out_channels=3, base_channels=64, mult=[1, 2, 4, 8]).to(device)
model_path = Path(__file__).resolve().parent / 'models' / 'model3_1248_cyclic_4step_50single.pth'
state_dict = torch.load(str(model_path), map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Initialize Weights and Biases
if os.path.exists('wandb.login'):
    with open('wandb.login', 'r') as f:
        wandb.login(key=f.read())
elif args.wandb_api_key is not None:
    wandb.login(key=args.wandb_api_key)
else:
    print("No Weights and Biases API key provided.")
wandb.init(entity="atai-apple-juice", project=f"{args.problem}_baseline", config=args, tags=[f'{args.problem}_baseline'])

# Evaluate the baseline model
evaluator = Evaluator(
    model,
    train_dataset,
    val_dataset,
    args,
    baseline=True
)
for step_size in args.evaluation_step_sizes:
    evaluator.evaluate_step_size(step_size)
evaluator.evaluate_trajectories()

# Finish Weights and Biases
wandb.finish()