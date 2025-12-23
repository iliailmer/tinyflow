import os
from typing import Any

import mlflow
from loguru import logger
from matplotlib import pyplot as plt


class MLflowLogger:
    """Wrapper for MLflow experiment tracking."""

    def __init__(
        self,
        experiment_name: str = "flow_matching",
        tracking_uri: str | None = None,
        enabled: bool = True,
    ):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking (default: local mlruns directory)
            enabled: Whether to enable MLflow logging
        """
        self.enabled = enabled
        self._run = None

        if self.enabled:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")
            if tracking_uri:
                logger.info(f"Tracking URI: {tracking_uri}")

    def start_run(self, run_name: str | None = None, tags: dict[str, Any] | None = None):
        """Start a new MLflow run."""
        if self.enabled:
            mlflow.end_run()
            self._run = mlflow.start_run(run_name=run_name)
            if tags:
                mlflow.set_tags(tags)
            logger.info(f"Started MLflow run: {self._run.info.run_id}")

    def end_run(self):
        """End the current MLflow run."""
        if self.enabled and self._run:
            mlflow.end_run()
            logger.info("Ended MLflow run")
            self._run = None

    def log_params(self, params: dict[str, Any]):
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameter names and values
        """
        if self.enabled and mlflow.active_run():
            # Flatten nested dictionaries for MLflow
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)

    def log_param(self, key: str, value: Any):
        """
        Log a single parameter to MLflow.

        Args:
            key: Parameter name
            value: Parameter value
        """
        if self.enabled and mlflow.active_run():
            mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int | None = None):
        """
        Log a single metric to MLflow.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number (e.g., epoch or iteration)
        """
        if self.enabled and mlflow.active_run():
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """
        Log multiple metrics to MLflow.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (e.g., epoch or iteration)
        """
        if self.enabled and mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        """
        Log an artifact (file) to MLflow.

        Args:
            local_path: Path to the local file
            artifact_path: Optional subdirectory within the run's artifact directory
        """
        if self.enabled and mlflow.active_run():
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path)
            else:
                logger.warning(f"Artifact not found: {local_path}")

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None):
        """
        Log a directory of artifacts to MLflow.

        Args:
            local_dir: Path to the local directory
            artifact_path: Optional subdirectory within the run's artifact directory
        """
        if self.enabled and mlflow.active_run():
            if os.path.exists(local_dir):
                mlflow.log_artifacts(local_dir, artifact_path)
            else:
                logger.warning(f"Artifact directory not found: {local_dir}")

    def log_figure(self, figure: plt.Figure, artifact_file: str, save_path: str | None = None):
        """
        Log a matplotlib figure to MLflow.

        Args:
            figure: Matplotlib figure object
            artifact_file: Name for the artifact file in MLflow
            save_path: Optional local path to also save the figure
        """
        if self.enabled and mlflow.active_run():
            mlflow.log_figure(figure, artifact_file)

        # Optionally save locally
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            figure.savefig(save_path)

    def log_image(self, image_path: str, artifact_file: str | None = None):
        """
        Log an image file to MLflow.

        Args:
            image_path: Path to the image file
            artifact_file: Optional name for the artifact (default: use filename)
        """
        if self.enabled and mlflow.active_run():
            if os.path.exists(image_path):
                mlflow.log_artifact(image_path, artifact_file)
            else:
                logger.warning(f"Image not found: {image_path}")

    def log_model(self, model: Any, artifact_path: str, **kwargs):
        """
        Log a model to MLflow.

        Args:
            model: Model object to log
            artifact_path: Path within the run's artifact directory
            **kwargs: Additional arguments for mlflow.pytorch.log_model or similar
        """
        if self.enabled and mlflow.active_run():
            # For now, we'll save model state separately
            # This can be extended based on the model type
            logger.info(f"Model logging to {artifact_path} (extend as needed)")

    def set_tags(self, tags: dict[str, Any]):
        """
        Set tags for the current run.

        Args:
            tags: Dictionary of tag names and values
        """
        if self.enabled and mlflow.active_run():
            mlflow.set_tags(tags)

    def set_tag(self, key: str, value: Any):
        """
        Set a single tag for the current run.

        Args:
            key: Tag name
            value: Tag value
        """
        if self.enabled and mlflow.active_run():
            mlflow.set_tag(key, value)

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """
        Flatten a nested dictionary for MLflow logging.

        Args:
            d: Dictionary to flatten
            parent_key: Prefix for keys (used in recursion)
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @property
    def run_id(self) -> str | None:
        """Get the current run ID."""
        if self.enabled and self._run:
            return self._run.info.run_id
        return None

    @property
    def run_name(self) -> str | None:
        """Get the current run name."""
        if self.enabled and self._run:
            return self._run.info.run_name
        return None

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()
