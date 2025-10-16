import torch
import wandb
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

from typing import Tuple, List
from torch import Tensor

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model, train_dataset, val_dataset, args, baseline: bool = False):
        """
        Provide the model, datasets, baseline, and problem type to evaluate various metrics.
        Results are logged to wandb.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.baseline = baseline

        self.problem_type = args.problem
        self.prior_conditioning = args.prior_conditioning
        self.euler_steps = args.euler_steps
        self.device = args.device
        self.sigma = args.sigma  # Noise level for flow matching generation
        self.use_tqdm = args.use_tqdm

        if self.problem_type == 'cfd' and not self.baseline:
            self.predict_frames = self.val_dataset.predict_frames

        # The training dataset and validation set have the same maximum step size.
        self.maximum_step_size = self.val_dataset.get_maximum_step_size()
        logger.info("Evaluator initialized.")
   

    def predict_steps(self, steps: int, include_training_data: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
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

        logger.info(f"Evaluating {len(data_points)} data points for step size {steps} (include_training_data={include_training_data}).")

        # Unpack the data points into input and target tensors
        # Input shape for CFD: [1, C+1, frame_history, 128, 64]
        # Target shape for CFD: [1, C+1, 1, 128, 64]
        # Input shape for Boids: [1, 25, C]
        # Target shape for Boids: [1, 25, C]
        inputs, targets = zip(*data_points)
        inputs = torch.stack(inputs)

        logger.debug(f"Input shape: {inputs.shape}, Target shape: {targets[0].shape}")

        # Create model predictions using rollout
        predictions = []
        if self.use_tqdm:
            inputs = tqdm(inputs, desc=f"Predicting for step size {steps}", leave=False)
        for input in inputs:
            # The prediction shape should match the target shape.
            # Prediction shape for CFD: [1, C+1, 1, 128, 64]
            # Prediction shape for Boids: [1, 25, C]
            prediction = self._rollout(input, steps)
            predictions.append(prediction)

        logger.info(f"Made predictions for {len(predictions)} data points.")
        # Keep targets as list for consistency with predictions
        targets = list(targets)

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
        output_state = input_state.to(self.device)
        steps_done = 0
        for _ in range(steps):
            if self.baseline:
                # Replace the output state with the prediction
                output_state = self._make_baseline_prediction(output_state)
            if self.problem_type == 'cfd' and not self.baseline:
                # Replace the output state with the prediction
                target_shape = (output_state.shape[0], output_state.shape[1], self.predict_frames, output_state.shape[3], output_state.shape[4])
                output_state = self._make_flow_matching_prediction(output_state, target_shape)
                steps_done += self.predict_frames
                if steps_done >= steps:
                    excess = steps_done - steps
                    if excess > 0:
                        # Trim the excess frames to match the exact number of steps
                        output_state = output_state[:, :, -excess:, :, :]
                    break
            if self.problem_type == 'boids' and not self.baseline:
                # Replace the output state with the prediction
                output_state = self._make_flow_matching_prediction(output_state)

        if self.problem_type == 'cfd' and not self.baseline:
            # For CFD: Return only the last frame
            return output_state[:, :, -1:, :, :]
        else:
            # For boids: Return the full prediction directly
            # For baseline: Return the full prediction directly
            return output_state
    
    def _make_baseline_prediction(
        self, 
        input: Tensor,
    ):
        """
        Generate a single prediction using the baseline model for either CFD or boids problem.
        """
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Generating baseline prediction for {input.shape}.")

        with torch.no_grad():
            input = input.to(self.device)
            return self.model(input)

    def _make_flow_matching_prediction(
        self, 
        input: Tensor,
        target_shape = None
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

        logger.info(f"Generating flow matching prediction for {input.shape}.")
        
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
            if self.problem_type == 'cfd' and target_shape is not None:
                # Noise like target shape
                noise_padding = torch.randn(target_shape).to(self.device)
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

        logger.info(f"Flow matching prediction generated with shape {output.shape}.")
        
        return output

    def evaluate_step_size(self, steps: int):
        """
        Evaluate the model's performance at a given step size.
        This will result in multiple data points to evaluate on, per trajectory.

        Log the performance metrics in wandb.
        """
        logger.info(f"Evaluating model performance at step size {steps}.")
        # For CFD: predictions and targets are lists of Tensors of shape [1, C+1, 1, 128, 64].
        # For boids: ...
        predictions, targets = self.predict_steps(steps=steps, include_training_data=False)

        assert len(predictions) == len(targets), "Predictions and targets must have the same length."

        mean_error = self._evaluate_mean_error(predictions, targets)
        mean_euclidean_distance = self._evaluate_mean_euclidean_distance(predictions, targets)

        logger.info("Logging step_size, evaluation_set_size, include_training_data, " +
                    "mean_error, mean_euclidean_distance to wandb.")
        wandb.log({
            "step_size": steps,
            "evaluation_set_size": len(predictions),
            "mean_error": mean_error,
            "mean_euclidean_distance": mean_euclidean_distance
        })

        return mean_error, mean_euclidean_distance

    def evaluate_trajectories(self):
        """
        Evaluate the model's performance on a trajectory.
        This will use the maximum step size such that we have one data point per trajectory to compare.

        Log the performance metrics in wandb.
        """
        logger.info(f"Evaluating model performance on full trajectories.")
        # For CFD: predictions and targets are lists of Tensors of shape [1, C+1, 1, 128, 64].
        # For boids: ...
        predictions, targets = self.predict_steps(steps=self.maximum_step_size, include_training_data=False)

        assert len(predictions) == len(targets), "Predictions and targets must have the same length."

        if self.problem_type == 'cfd':
            kl_divergence_velocity_densities, velocity_density_predictions, velocity_density_targets = self._evaluate_velocity_density(predictions, targets)
            kl_divergence_pressure_densities, pressure_density_predictions, pressure_density_targets = self._evaluate_pressure_density(predictions, targets)

            logger.info("Logging trajectory evaluation metrics of cfd to wandb.")
            wandb.log({
                "step_size": self.maximum_step_size,
                "evaluation_set_size": len(predictions),
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

            logger.info("Logging trajectory evaluation metrics of boids to wandb.")
            wandb.log({
                "step_size": self.maximum_step_size,
                "evaluation_set_size": len(predictions),
                "kl_divergence_velocity_densities": kl_divergence_velocity_densities,
                "velocity_density_predictions": velocity_density_predictions,
                "velocity_density_targets": velocity_density_targets,
                "kl_divergence_cluster_densities": kl_divergence_cluster_densities,
                "cluster_density_predictions": cluster_density_predictions,
                "cluster_density_targets": cluster_density_targets,
            })

            return kl_divergence_velocity_densities, kl_divergence_cluster_densities

    def _evaluate_mean_error(self, predictions: List[Tensor], targets: List[Tensor]):
        """
        Evaluate the mean error of the predictions.
        """
        # For CFD: predictions and targets are lists of Tensors of shape [1, C+1, 1, 128, 64].
        # For boids: predictions and targets are lists of Tensors of shape [1, 25, C].
        # We compute the mean error over channels [0:3] (vx, vy, pressure) for CFD; over all features for boids.

        # Stack lists into batched tensors
        preds = torch.cat(predictions, dim=0).to(self.device)
        targs = torch.cat(targets, dim=0).to(self.device)

        if self.problem_type == 'cfd':
            # Select vx, vy, pressure channels
            preds = preds[:, 0:3, ...]
            targs = targs[:, 0:3, ...]

        return torch.mean(preds - targs)

    def _evaluate_mean_euclidean_distance(self, predictions: List[Tensor], targets: List[Tensor]):
        """
        Evaluate the mean euclidean distance between the predictions and targets.
        """
        # For CFD: predictions and targets are lists of Tensors of shape [1, C+1, 1, 128, 64].
        # For boids: predictions and targets are lists of Tensors of shape [1, 25, C].
        # We compute the mean euclidean distance over the first 3 channels (vx, vy, pressure) for CFD.
        
        # Stack lists into batched tensors
        preds = torch.cat(predictions, dim=0).to(self.device)
        targs = torch.cat(targets, dim=0).to(self.device)
        
        if self.problem_type == 'cfd':
            # Select vx, vy, pressure channels
            preds = preds[:, 0:3, ...]
            targs = targs[:, 0:3, ...]
        
        return torch.mean(torch.norm(preds - targs, dim=-1))

    def _evaluate_velocity_density(self, predictions: List[Tensor], targets: List[Tensor]):
        """
        Evaluate the velocity density of the predictions.

        This density is computed for both problems, hence the velocity features 
        are extracted separately for each problem.

        For the cfd the tensors have shape [1, C+1, 1, 128, 64] we need to extract the velocity features from the last frame.
        In this case C=3 and we add one dimension for the mask, resulting in 4 indices for the 2nd dimension.
        The velocity features are on the first 2 indices of the 2nd dimension.

        For the boids problem 
        """
        # Extract the velocity features for each problem.
        if self.problem_type == 'cfd':
            # The state space contains 64x128 cells each of which contain a velocity (2d) and pressure (1d).
            # Input shape: [1, 4, 1, 128, 64] where dimension 1 has [vx, vy, pressure, mask]
            # Extract velocity_x and velocity_y from indices 0 and 1 of dimension 1
            
            prediction_velocity_magnitudes = []
            target_velocity_magnitudes = []
            
            for pred_tensor in predictions:
                # Extract velocity components: shape [1, 1, 128, 64] each
                vx = pred_tensor[:, 0, :, :, :]  # velocity_x
                vy = pred_tensor[:, 1, :, :, :]  # velocity_y
                
                # Compute velocity magnitude: sqrt(vx^2 + vy^2)
                velocity_magnitude = torch.sqrt(vx**2 + vy**2)
                
                # Flatten to get all velocity magnitudes for this tensor (128*64 values)
                prediction_velocity_magnitudes.append(velocity_magnitude.flatten())
            
            for target_tensor in targets:
                # Extract velocity components
                vx = target_tensor[:, 0, :, :, :]
                vy = target_tensor[:, 1, :, :, :]
                
                # Compute velocity magnitude
                velocity_magnitude = torch.sqrt(vx**2 + vy**2)
                
                # Flatten to get all velocity magnitudes
                target_velocity_magnitudes.append(velocity_magnitude.flatten())
            
            # Concatenate all velocity magnitudes into a single tensor
            prediction_velocity_features = torch.cat(prediction_velocity_magnitudes)
            target_velocity_features = torch.cat(target_velocity_magnitudes)
            
        elif self.problem_type == 'boids':
            # Extract velocity magnitudes from boids states: [pos_x, pos_y, vel_x, vel_y]
            prediction_velocity_magnitudes = []
            target_velocity_magnitudes = []

            for pred_tensor in predictions:
                # pred_tensor shape expected: [1, 25, C] with C >= 4
                vx = pred_tensor[..., 2]
                vy = pred_tensor[..., 3]
                velocity_magnitude = torch.sqrt(vx**2 + vy**2)
                prediction_velocity_magnitudes.append(velocity_magnitude.flatten())

            for target_tensor in targets:
                vx = target_tensor[..., 2]
                vy = target_tensor[..., 3]
                velocity_magnitude = torch.sqrt(vx**2 + vy**2)
                target_velocity_magnitudes.append(velocity_magnitude.flatten())

            prediction_velocity_features = torch.cat(prediction_velocity_magnitudes)
            target_velocity_features = torch.cat(target_velocity_magnitudes)

        # Build shared-bin probability histograms for proper KL comparison
        pred = prediction_velocity_features.detach().float().cpu()
        targ = target_velocity_features.detach().float().cpu()

        combined = torch.cat([pred, targ], dim=0)
        vmin = torch.min(combined)
        vmax = torch.max(combined)
        if not torch.isfinite(vmin) or not torch.isfinite(vmax):
            vmin = torch.tensor(0.0)
            vmax = torch.tensor(1.0)
        if vmax <= vmin:
            vmax = vmin + 1e-6

        num_bins = 50
        counts_pred = torch.histc(pred, bins=num_bins, min=float(vmin), max=float(vmax))
        counts_targ = torch.histc(targ, bins=num_bins, min=float(vmin), max=float(vmax))

        # Convert counts to probability distributions with epsilon smoothing
        eps = 1e-8
        velocity_density_predictions = (counts_pred + eps) / (counts_pred.sum() + eps * num_bins)
        velocity_density_targets = (counts_targ + eps) / (counts_targ.sum() + eps * num_bins)

        logger.info("Making velocity density plot and logging to wandb.")
        # Make density plot
        plt.figure()
        bin_edges = torch.linspace(float(vmin), float(vmax), num_bins + 1)
        plt.hist(pred.numpy(), bins=bin_edges.numpy(), alpha=0.5, label='Predictions', density=True)
        plt.hist(targ.numpy(), bins=bin_edges.numpy(), alpha=0.5, label='Targets', density=True)
        plt.legend()
        plt.title('Velocity Density')
        plt.xlabel('Velocity Magnitude')
        plt.ylabel('Frequency')
        plt.savefig(f'./models/output/{self.problem_type}_velocity_density.png')
        plt.close()
        wandb.log({f"{self.problem_type}_velocity_density_plot":
                       wandb.Image(f'./models/output/{self.problem_type}_velocity_density.png')})

        # Compute KL divergence KL(P || Q) with P=pred, Q=target
        # kl_div expects log-probabilities as input and probabilities as target
        # To compute KL(pred || target): input=log(target), target=pred
        log_target = torch.log(velocity_density_targets)
        kl_divergence = torch.nn.functional.kl_div(log_target, velocity_density_predictions, reduction='sum')

        return kl_divergence, velocity_density_predictions, velocity_density_targets

    def _evaluate_pressure_density(self, predictions, targets):
        """
        Evaluate the pressure density of the predictions.

        This density is computed for both problems, hence the pressure features 
        are extracted separately for each problem.
        """
        assert self.problem_type == 'cfd', "_evaluate_pressure_density is only defined for CFD."

        # Extract pressure channel (index 2 of the 2nd dimension) and flatten
        prediction_pressure_list = []
        target_pressure_list = []

        for pred_tensor in predictions:
            pressure = pred_tensor[:, 2, :, :, :]  # shape [1, 1, 128, 64]
            prediction_pressure_list.append(pressure.flatten())
        for target_tensor in targets:
            pressure = target_tensor[:, 2, :, :, :]
            target_pressure_list.append(pressure.flatten())

        prediction_pressure_features = torch.cat(prediction_pressure_list)
        target_pressure_features = torch.cat(target_pressure_list)

        # Build shared-bin probability histograms
        pred = prediction_pressure_features.detach().float().cpu()
        targ = target_pressure_features.detach().float().cpu()

        combined = torch.cat([pred, targ], dim=0)
        vmin = torch.min(combined)
        vmax = torch.max(combined)
        if not torch.isfinite(vmin) or not torch.isfinite(vmax):
            vmin = torch.tensor(0.0)
            vmax = torch.tensor(1.0)
        if vmax <= vmin:
            vmax = vmin + 1e-6

        num_bins = 50
        counts_pred = torch.histc(pred, bins=num_bins, min=float(vmin), max=float(vmax))
        counts_targ = torch.histc(targ, bins=num_bins, min=float(vmin), max=float(vmax))

        eps = 1e-8
        pressure_density_predictions = (counts_pred + eps) / (counts_pred.sum() + eps * num_bins)
        pressure_density_targets = (counts_targ + eps) / (counts_targ.sum() + eps * num_bins)

        # Plot densities with shared bins
        logger.info("Making pressure density plot and logging to wandb.")
        # Make density plot
        plt.figure()
        bin_edges = torch.linspace(float(vmin), float(vmax), num_bins + 1)
        plt.hist(pred.numpy(), bins=bin_edges.numpy(), alpha=0.5, label='Predictions', density=True)
        plt.hist(targ.numpy(), bins=bin_edges.numpy(), alpha=0.5, label='Targets', density=True)
        plt.legend()
        plt.title('Pressure Density')
        plt.xlabel('Pressure')
        plt.ylabel('Density')
        plt.savefig(f'./models/output/{self.problem_type}_pressure_density.png')
        plt.close()
        wandb.log({f"{self.problem_type}_pressure_density_plot":
                       wandb.Image(f'./models/output/{self.problem_type}_pressure_density.png')})

        # Compute KL divergence KL(P || Q) with P=pred, Q=target
        # kl_div expects log-probabilities as input and probabilities as target
        # To compute KL(pred || target): input=log(target), target=pred
        log_target = torch.log(pressure_density_targets)
        kl_divergence = torch.nn.functional.kl_div(log_target, pressure_density_predictions, reduction='sum')

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

        logger.info("Making cluster density plot and logging to wandb.")
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
