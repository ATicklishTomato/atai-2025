import torch
import wandb
import matplotlib.pyplot as plt

from typing import Tuple, List
from torch import Tensor

class Evaluator:
    def __init__(self, model, train_dataset, val_dataset, args):
        """
        Provide the model, datasets, and problem type to evaluate various metrics.
        Results are logged to wandb.

        The datasets need to provide the following methods:
        - `get_step_data_points(steps: int)` -> List[Tuple[Tensor, Tensor]]: To get the all pairs of data points within the same sequence that
            have an initial frame and a target frame separated by `steps` frames. This is used
            to evaluate the prediction of `steps` steps. When this method is called with the maximum step size, 
            it should return one data point per trajectory (the initial state).
        - `get_maximum_step_size() -> int`: To get the maximum `step` size, such that there is one data point
            per sequence.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.problem_type = args.problem
        self.prior_conditioning = args.prior_conditioning
        self.euler_steps = args.euler_steps
        self.device = args.device
        self.sigma = args.sigma  # Noise level for flow matching generation

        # The training dataset and validation set have the same maximum step size.
        self.maximum_step_size = self.val_dataset.get_maximum_step_size() 
   

    def predict_steps(self, steps: int, include_training_data: bool = False):
        """
        Auto-regressively predict a sequence of steps given an initial frame.
        Returns the predicted sequence and the ground truth sequence for comparison.

        Predicts results on the validation set only unless full_data is True.
        Making use of the full_data is useful for density based metrics as
        they are evaluated on full trajectories only, which yields few prediction points.
        
        Args:
            steps: Number of steps to predict
            full_data: Whether to include training data in evaluation
            include_training_data: Whether to include training data in evaluation

        Returns:
            predictions: List of predicted final states (shape is dependent on the problem)
            targets: List of ground truth final states (shape is dependent on the problem)
        """
        assert steps > 0, "Number of steps must be positive."
        assert steps <= self.maximum_step_size, "Number of steps must be less than or equal to the maximum step size."

        # Collect the data points to evaluate on
        data_points: List[Tuple[Tensor, Tensor]] = self.val_dataset.get_step_data_points(steps)
        if include_training_data:
            data_points += self.train_dataset.get_step_data_points(steps)

        # Unpack the data points into input and target tensors
        # Input shape for CFD: [1, C+1, frame_history, 128, 64]
        # Target shape for CFD: [1, C+1, 1, 128, 64]
        # Input shape for Boids: [1, 25, C]
        # Target shape for Boids: [1, 25, C]
        inputs, targets = zip(*data_points)
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        # Create model predictions using rollout
        predictions = []
        for input in inputs:
            # The prediction shape should match the target shape.
            # Prediction shape for CFD: [1, C+1, 1, 128, 64]
            # Prediction shape for Boids: [1, 25, C]
            prediction = self._rollout(input, steps)
            predictions.append(prediction)

        return predictions, targets
    
    def _rollout(self, input_state: Tensor, steps: int):
        """
        Make an auto-regressive rollout of the model for a given number of steps.
        Return the final predicted state after the rollout.
        
        This method delegates to _make_flow_matching_prediction which handles
        both CFD (flow matching) and boids (simple autoregressive) models.
        
        Args:
            input_state: The input state to start from
            steps: Number of steps to predict (used for boids)
        """
        assert steps > 0, "Number of steps must be positive."
        
        # Input shape for CFD: [1, C+1, frame_history, 128, 64]
        # Input shape for Boids: [1, 25, C]
        output_state = input_state
        for _ in range(steps):
            if self.problem_type == 'cfd':
                # Augment the output state with the prediction
                # We remove the first frame of the input and append the first frame of the prediction
                output_state = torch.cat([output_state[:, :, 1:, :, :], self._make_flow_matching_prediction(output_state)[:, :, 0:1, :, :]], dim=2)
            if self.problem_type == 'boids':
                # Replace the output state with the prediction
                output_state = self._make_flow_matching_prediction(output_state)
        
        if self.problem_type == 'cfd':
            # For CFD: Return only the last frame
            return output_state[:, :, -1:, :, :]
        if self.problem_type == 'boids':
            # For boids: Return the full prediction directly
            return output_state
    
    def _make_flow_matching_prediction(
        self, 
        input: Tensor
    ):
        """
        Generate a single prediction using flow matching for either CFD or boids problem.
        
        Both CFD and boids use flow matching generation:
        - CFD uses time bundling (multiple frames per state)
        - Boids uses frame-by-frame prediction (single frame per state)
        
        Prior conditioning behavior:
        - If prior_conditioning is True: Start from input + noise (conditional generation)
        - If prior_conditioning is False: Start from pure noise (unconditional generation)
            
        Returns:
            Generated prediction with same shape as input
        """
        self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            # Shape for CFD: [1, C+1, frame_history, 128, 64]
            input = input.to(self.device)
            
            # Define initial state x based on prior_conditioning
            if self.prior_conditioning:
                # Conditional generation: start from noisy input
                x = input + self.sigma * torch.randn_like(input).to(self.device)
            else:
                # Unconditional generation: start from pure noise
                x = torch.randn_like(input).to(self.device)
            
            # Handle padding for time bundling (CFD only)
            if self.problem_type == 'cfd':
                noise_padding = torch.randn_like(input).to(self.device)
                padding_needed = noise_padding.shape[2] - x.shape[2]
                
                if padding_needed > 0:
                    # Prepend random noise to x to match target's frame dimension
                    x = torch.cat([noise_padding[:, :, :padding_needed], x], dim=2)
                elif padding_needed < 0:
                    # Trim x to match target's frame dimension
                    x = x[:, :, -padding_needed:]
            
            # Generate using flow matching with x the data point along the path from x_0 to x_1 and input as the initial state
            # of the dynamical system to condition on.
            output = self.model.generation(
                x=x,
                x_hist=input, 
                n_euler_steps=self.euler_steps
            )
        
        return output

    def evaluate_step_size(self, steps: int):
        """
        Evaluate the model's performance at a given step size.

        Log the performance metrics in wandb.
        """
        for include_training_data in [True, False]:
            predictions, targets = self.predict_steps(steps=steps, include_training_data=include_training_data)

            assert len(predictions) == len(targets), "Predictions and targets must have the same length."

            mean_error = self._evaluate_mean_error(predictions, targets)
            mean_euclidean_distance = self._evaluate_mean_euclidean_distance(predictions, targets)

            wandb.log({
                "step_size": steps,
                "evaluation_set_size": len(predictions),
                "include_training_data": include_training_data,
                "mean_error": mean_error,
                "mean_euclidean_distance": mean_euclidean_distance
            })

        return mean_error, mean_euclidean_distance

    def evaluate_trajectories(self):
        """
        Evaluate the model's performance on a trajectory.

        Log the performance metrics in wandb.
        """
        for include_training_data in [True, False]:
            predictions, targets = self.predict_steps(steps=self.maximum_step_size, include_training_data=include_training_data)

            assert len(predictions) == len(targets), "Predictions and targets must have the same length."

            if self.problem_type == 'cfd':
                kl_divergence_velocity_densities, velocity_density_predictions, velocity_density_targets = self._evaluate_velocity_density(predictions, targets)
                kl_divergence_pressure_densities, pressure_density_predictions, pressure_density_targets = self._evaluate_pressure_density(predictions, targets)

                wandb.log({
                    "step_size": self.maximum_step_size,
                    "evaluation_set_size": len(predictions),
                    "include_training_data": include_training_data,
                    "kl_divergence_velocity_densities": kl_divergence_velocity_densities,
                    "velocity_density_predictions": velocity_density_predictions,
                    "velocity_density_targets": velocity_density_targets,
                    "kl_divergence_pressure_densities": kl_divergence_pressure_densities,
                    "pressure_density_predictions": pressure_density_predictions,
                    "pressure_density_targets": pressure_density_targets,
                })

                return kl_divergence_velocity_densities, kl_divergence_pressure_densities

            if self.problem_type == 'boids':
                kl_divergence_velocity_densities, velocity_density_predictions, velocity_density_targets = self._evaluate_velocity_density(predictions, targets)
                kl_divergence_cluster_densities, cluster_density_predictions, cluster_density_targets = self._evaluate_cluster_density(predictions, targets)

                wandb.log({
                    "step_size": self.maximum_step_size,
                    "evaluation_set_size": len(predictions),
                    "include_training_data": include_training_data,
                    "kl_divergence_velocity_densities": kl_divergence_velocity_densities,
                    "velocity_density_predictions": velocity_density_predictions,
                    "velocity_density_targets": velocity_density_targets,
                    "kl_divergence_cluster_densities": kl_divergence_cluster_densities,
                    "cluster_density_predictions": cluster_density_predictions,
                    "cluster_density_targets": cluster_density_targets,
                })

                return kl_divergence_velocity_densities, kl_divergence_cluster_densities

    def _evaluate_mean_error(self, predictions, targets):
        """
        Evaluate the mean error of the predictions.
        """
        return torch.mean(predictions - targets)

    def _evaluate_mean_euclidean_distance(self, predictions, targets):
        """
        Evaluate the mean euclidean distance between the predictions and targets.
        """
        # Predictions and targets shape for CFD are: (N, 64, 128, 2) with N the number of data points for this step size.
        # Predictions and targets shape for Boids are: (N, 25, 2) with N the number of data points for this step size.
        # We compute the mean euclidean distance over the last dimension (feature dimension)
        return torch.mean(torch.norm(predictions - targets, dim=-1))

    def _evaluate_velocity_density(self, predictions, targets):
        """
        Evaluate the velocity density of the predictions.

        This density is computed for both problems, hence the velocity features 
        are extracted separately for each problem.

        For the cfd problem we 

        For the boids problem 
        """
        # TODO: Extract the velocity features for each problem.
        if self.problem_type == 'cfd':
            # The state space contains 64x128 cells each of which contain a velocity (2d) and pressure (1d).
            prediction_velocity_features = predictions
            target_velocity_features = targets
            # Resulting shape: (T, 64, 128, 2) with T the number of trajectories
            # Shape we need: (X, 2) with X the number of velocity features
        elif self.problem_type == 'boids':
            # The state space contains 25 boids each of which contain a position (2d) and velocity (2d).
            prediction_velocity_features = predictions
            target_velocity_features = targets
            # Resulting shape: (T, 25, 2) with T the number of trajectories
            # Shape we need: (X, 2) with X the number of velocity features

        # Compute the velocity density of the predictions.
        velocity_density_predictions = torch.histc(prediction_velocity_features, bins=10, min=0, max=1)

        # Compute the velocity density of the targets.
        velocity_density_targets = torch.histc(target_velocity_features, bins=10, min=0, max=1)

        # Make density plot
        plt.figure()
        plt.hist(prediction_velocity_features.cpu().numpy(), bins=10, alpha=0.5, label='Predictions')
        plt.hist(target_velocity_features.cpu().numpy(), bins=10, alpha=0.5, label='Targets')
        plt.legend()
        plt.title('Velocity Density')
        plt.xlabel('Velocity Magnitude')
        plt.ylabel('Frequency')
        plt.savefig(f'./models/output/{self.problem_type}_velocity_density.png')
        plt.close()
        wandb.log({f"{self.problem_type}_velocity_density_plot":
                       wandb.Image(f'./models/output/{self.problem_type}_velocity_density.png')})

        # Compute the kullback-leibler divergence between of the predicted velocity densities under the target velocity densities.
        # TODO: Check whether we need to use reduction here.
        kl_divergence = torch.nn.functional.kl_div(velocity_density_predictions, velocity_density_targets, reduction='sum')

        return kl_divergence, velocity_density_predictions, velocity_density_targets

    def _evaluate_pressure_density(self, predictions, targets):
        """
        Evaluate the pressure density of the predictions.

        This density is computed for both problems, hence the pressure features 
        are extracted separately for each problem.
        """
        # TODO: Extract the pressure features
        # The state space contains 64x128 cells each of which contain a pressure (2d) and pressure (1d).
        prediction_pressure_features = predictions
        target_pressure_features = targets
        # Resulting shape: (T, 64, 128, 2) with T the number of trajectories
        # Shape we need: (X, 2) with X the number of pressure features

        # Compute the pressure density of the predictions.
        pressure_density_predictions = torch.histc(prediction_pressure_features, bins=10, min=0, max=1)

        # Compute the pressure density of the targets.
        pressure_density_targets = torch.histc(target_pressure_features, bins=10, min=0, max=1)

        # Make density plot
        plt.figure()
        plt.hist(prediction_pressure_features.cpu().numpy(), bins=10, alpha=0.5, label='Predictions')
        plt.hist(target_pressure_features.cpu().numpy(), bins=10, alpha=0.5, label='Targets')
        plt.legend()
        plt.title('Pressure Density')
        plt.xlabel('Pressure')
        plt.ylabel('Frequency')
        plt.savefig(f'./models/output/{self.problem_type}_pressure_density.png')
        plt.close()
        wandb.log({f"{self.problem_type}_pressure_density_plot":
                       wandb.Image(f'./models/output/{self.problem_type}_pressure_density.png')})

        # Compute the kullback-leibler divergence between of the predicted pressure densities under the target pressure densities.
        # TODO: Check whether we need to use reduction here.
        kl_divergence = torch.nn.functional.kl_div(pressure_density_predictions, pressure_density_targets, reduction='sum')

        return kl_divergence, pressure_density_predictions, pressure_density_targets

    def _evaluate_cluster_density(self, predictions, targets):
        """
        Evaluate the cluster density of the predictions.
        """
        # TODO: Extract the position features for the boids problem.
        position_features_predictions = predictions
        position_features_targets = targets

        cluster_features_predictions = self._compute_cluster_features(position_features_predictions)
        cluster_features_targets = self._compute_cluster_features(position_features_targets)

        assert len(cluster_features_predictions) == len(cluster_features_targets), "Cluster features and cluster features targets must have the same length."

        # Compute the cluster density of the predictions.
        cluster_density_predictions = torch.histc(cluster_features_predictions, bins=10, min=0, max=1)

        # Compute the cluster density of the targets.
        cluster_density_targets = torch.histc(cluster_features_targets, bins=10, min=0, max=1)

        # Make density plot
        plt.figure()
        plt.hist(cluster_features_predictions.cpu().numpy(), bins=10, alpha=0.5, label='Predictions')
        plt.hist(cluster_features_targets.cpu().numpy(), bins=10, alpha=0.5, label='Targets')
        plt.legend()
        plt.title('Cluster Density')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Frequency')
        plt.savefig(f'./models/output/{self.problem_type}_cluster_density.png')
        plt.close()
        wandb.log({f"{self.problem_type}_cluster_density_plot":
                       wandb.Image(f'./models/output/{self.problem_type}_cluster_density.png')})


        # Compute the kullback-leibler divergence between of the predicted cluster densities under the target cluster densities.
        # TODO: Check whether we need to use reduction here.
        kl_divergence = torch.nn.functional.kl_div(cluster_density_predictions, cluster_density_targets, reduction='sum')

        return kl_divergence, cluster_density_predictions, cluster_density_targets

    def _compute_cluster_features(self, position_features):
        """
        Compute the number of clusters found in the position features for the boids problem.

        The input shape should be (N, 25, 2) with N the number of states.
        The output shape should be (N, 1) with N the number of states.
        """
        # TODO: Run a clustering algorithm on each of the position_features to determine the number of clusters
        cluster_features = None
        return cluster_features
