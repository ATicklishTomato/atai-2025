import os
from argparse import ArgumentParser
import logging
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
                        default=1001,
                        help='Number of epochs to train for. Default is 1001')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Batch size for training. Default is 1')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate for training. Default is 1e-3')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='PyTorch device to train on. Default is cuda')
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
        from assignment2.modules.cfd_model import CFDModel
        model = CFDModel(args)
    elif args.problem == 'boids':
        from assignment2.modules.boids_model import BoidsModel
        model = BoidsModel(args)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")
    return model

def get_dataloaders(args):
    if args.problem == 'cfd':
        from assignment2.modules.cfd_dataloaders import get_cfd_dataloaders
        return get_cfd_dataloaders(args)
    elif args.problem == 'boids':
        from assignment2.modules.boids_dataloaders import get_boids_dataloaders
        return get_boids_dataloaders(args)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

def get_trainer(args, model, train_dataloader, val_dataloader):
    if args.problem == 'cfd':
        from assignment2.modules.cfd_trainer import CFDTrainer
        return CFDTrainer(args, model, train_dataloader, val_dataloader)
    elif args.problem == 'boids':
        from assignment2.modules.boids_trainer import BoidsTrainer
        return BoidsTrainer(args, model, train_dataloader, val_dataloader)
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

def main():
    args = parse_args()
    logging.basicConfig(
        filename='run.log',
        level=args.verbose,
        format="%(levelname)s %(asctime)s (%(filename)s, %(funcName)s) - %(message)s"
    )

    wandb_config = {
        "problem": args.problem,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "verbose": args.verbose,
        "save": args.save,
        "load": args.load
    }

    if args.wandb_api_key is not None:
        wandb.login(key=args.wandb_api_key)
    elif os.path.exists('wandb.login'):
        with open('wandb.login', 'r') as f:
            wandb.login(key=f.read())
    else:
        logger.warning("No Weights and Biases API key provided.")

    wandb.init(project=args.problem, config=wandb_config)
    logger.info("Weights and Biases initialized")

    model = get_model(args)
    train_dataloader, val_dataloader = get_dataloaders(args)
    trainer = get_trainer(args, model, train_dataloader, val_dataloader)

    trainer.train()
    wandb.finish()
    logger.info("Run complete. Logs saved in run.log")


if __name__ == '__main__':
    main()
