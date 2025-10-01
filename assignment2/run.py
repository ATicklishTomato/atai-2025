import os
from argparse import ArgumentParser
import logging
from datetime import datetime
import torch

import wandb

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description='Train and test a neural network on either a cfd or a boids dataset')
    parser.add_argument('--problem',
                        type=str,
                        choices=['cfd', 'boids'],
                        default='cfd',
                        help='Type of problem to train and test the respective model on. Default is cfd.')
    parser.add_argument('--sweep',
                        action='store_true',
                        help='Run a hyperparameter sweep. Default is False. Note: This will override ' +
                        'any arguments passed related to sweep parameters')
    parser.add_argument('--sweep_runs',
                        type=int,
                        default=25,
                        help='Number of random runs to perform in the hyperparameter sweep. Default is 25')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to train for. Default is 100')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='Batch size for training. Default is 8')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='Learning rate for training. Default is 1e-4')
    parser.add_argument('--bundle_size',
                        type=int,
                        default=20,
                        help='Number of frames in each training bundle (when using time bundling). Default is 20')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=64,
                        help='Base hidden size for the model. Can be multiplied for deeper layers. Default is 64')
    parser.add_argument('--sigma',
                        type=float,
                        default=0.1,
                        help='Noise level for flow matching model. Default is 0.1')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='PyTorch device to train on. Default is cuda')
    parser.add_argument('--use_tqdm',
                        action='store_true',
                        help='Use tqdm progress bars during training. Default is False')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show any animations or graphs that are generated. Default is False')
    parser.add_argument('--verbose',
                        type=int,
                        default=logging.INFO,
                        choices=[logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR],
                        help='Verbosity level for logging. Options are for DEBUG, INFO, WARNING, and ERROR, ' +
                             'respectively. Default is INFO')
    parser.add_argument('--save',
                        action='store_true',
                        help='Save the model and optimizer state_dicts (if applicable) after training. ' +
                             'Default is False')
    parser.add_argument('--load',
                        action='store_true',
                        help='Load the stored model and optimizer state_dicts (if applicable) ' +
                             'before training and skip training. Default is False')
    parser.add_argument('--skip_train',
                        action='store_true',
                        help='Skip training and only evaluate the model. Default is False')
    parser.add_argument('--skip_test',
                        action='store_true',
                        help='Skip testing and only train the model. Default is False')
    parser.add_argument('--no_save_figures',
                        action='store_false',
                        help='Save any figures that are generated during testing. Default is True')
    parser.add_argument('--wandb_api_key',
                        type=str,
                        default=None,
                        help='Your personal API key for Weights and Biases. Default is None. Alternatively, you can ' +
                             'leave this empty and store the key in a file in the root of this script called "wandb.login". ' +
                             'This file will be ignored by git. ' +
                             'NOTE: Make sure to keep this key private and secure. Do not share it or upload it to ' +
                             'a public repository.')
    return parser.parse_args()

def get_model(args):
    if args.problem == 'cfd':
        from modules.cfd_model import CFDModel
        model = CFDModel(base_ch=args.hidden_size)
    elif args.problem == 'boids':
        from modules.boids_model import BoidsModel
        model = BoidsModel(args)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")
    return model

def get_dataloaders(args):
    if args.problem == 'cfd':
        from modules.cfd_dataloaders import get_cfd_dataloaders
        return get_cfd_dataloaders(bundle=args.bundle_size, batch_size=args.batch_size)
    elif args.problem == 'boids':
        from modules.boids_dataloaders import get_boids_dataloaders
        return get_boids_dataloaders(args)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

def get_trainer(args, model, train_dataloader, val_dataloader):
    if args.problem == 'cfd':
        from modules.cfd_trainer import CFDTrainer
        return CFDTrainer(args, model, train_dataloader, val_dataloader)
    elif args.problem == 'boids':
        from modules.boids_trainer import BoidsTrainer
        return BoidsTrainer(args, model, train_dataloader, val_dataloader)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

def do_test(args, model, val_dataloader):
    if args.problem == 'cfd':
        from modules.cfd_tester import compute_rollout, animate_rollout
        device = args.device
        model = model.to(device)
        trajectory = compute_rollout(model, num_steps=200, bundle_size=args.bundle_size, device=device)
        animate_rollout(trajectory, save_path=f"models/output/cfd_rollout.gif", show=args.show, save=(not args.no_save_figures))
    elif args.problem == 'boids':
        raise NotImplementedError("Testing not yet implemented for boids problem")
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

def sweep_train():
    wandb.init(config=wandb.config)
    sweep_args = parse_args()
    sweep_args.lr = wandb.config.lr
    sweep_args.sigma = wandb.config.sigma
    sweep_args.batch_size = wandb.config.batch_size
    sweep_args.bundle_size = wandb.config.bundle_size
    sweep_model = get_model(sweep_args)
    sweep_train_dataloader, sweep_val_dataloader = get_dataloaders(sweep_args)
    trainer = get_trainer(sweep_args, sweep_model, sweep_train_dataloader, sweep_val_dataloader)
    trainer.train()

def main():
    args = parse_args()
    logging.basicConfig(
        filename=f'run-{args.problem}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log',
        level=args.verbose,
        format="%(levelname)s %(asctime)s (%(filename)s, %(funcName)s) - %(message)s"
    )

    wandb_config = {
        "problem": args.problem,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sigma": args.sigma,
        "bundle_size": args.bundle_size,
        "use_tqdm": args.use_tqdm,
        "device": args.device,
        "verbose": args.verbose,
        "save": args.save,
        "load": args.load,
        "skip_train": args.skip_train,
        "skip_test": args.skip_test,
        "no_save_figures": args.no_save_figures,
    }

    if args.wandb_api_key is not None:
        wandb.login(key=args.wandb_api_key)
    elif os.path.exists('wandb.login'):
        with open('wandb.login', 'r') as f:
            wandb.login(key=f.read())
    else:
        logger.warning("No Weights and Biases API key provided.")

    if args.sweep:
        sweep_config = {
            'method': 'random',
            'metric': {'name': 'val_avg_loss', 'goal': 'minimize'},
            'parameters': {
                'lr': {'values': [1e-5, 1e-4, 1e-3]},
                'sigma': {'min': 0.05, 'max': 0.2},
                'batch_size': {'values': [4, 8, 16]},
                'bundle_size': {'values': [10, 20, 30]},
                'layer_size': {'values': [64, 128, 256]}
            }
        }
        sweep_id = wandb.sweep(sweep_config, project=args.problem)
        wandb.agent(sweep_id, function=sweep_train, count=args.sweep_runs)
        return

    wandb.init(project=args.problem, config=wandb_config)
    logger.info("Weights and Biases initialized")
    model = get_model(args)
    train_dataloader, val_dataloader = get_dataloaders(args)

    if args.load:
        if os.path.exists(f'models/{args.problem}_model.pth'):
            model.load_state_dict(torch.load(f'models/{args.problem}_model.pth', map_location=args.device))
            logger.info(f"Loaded model state_dict from models/{args.problem}_model.pth")
        else:
            logger.warning(f"No saved model found at models/{args.problem}_model.pth. Starting from scratch.")
    if args.skip_train and not args.load:
        logger.error("Cannot skip training without loading a model. Exiting.")
        return
    if not args.skip_train:
        trainer = get_trainer(args, model, train_dataloader, val_dataloader)
        trainer.train()
    if not args.skip_test:
        do_test(args, model, val_dataloader)
    wandb.finish()
    logger.info("Run complete. Logs saved in run.log")


if __name__ == '__main__':
    main()
