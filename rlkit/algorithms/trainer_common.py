# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common utilities shared across training algorithms (SFT, GRPO, KD, RM).

This module provides reusable components for:
- Logger setup
- Checkpointing
- Dataloader creation
- Cluster initialization
- Validation scheduling
- Metrics aggregation
"""
import logging
import os
from pathlib import Path
from typing import Any, Callable, NotRequired, Optional, TypedDict, TypeVar, cast
import warnings

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from rlkit.config import CheckpointingConfig, ClusterConfig
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import Logger
from rlkit.utils.timer import Timer

# Avoid circular import by using TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlkit.models.policy.lm_policy import Policy
    from transformers import PreTrainedTokenizerBase

SaveStateT = TypeVar("SaveStateT", bound=TypedDict)


# ===============================================================================
# Save State Management
# ===============================================================================


class BaseSaveState(TypedDict):
    """Base save state with common fields across all trainers."""

    step: int
    consumed_samples: int


def update_save_state_for_checkpoint(
    save_state: dict[str, Any],
    step: int,
    consumed_samples: int,
    val_metrics: Optional[dict[str, Any]],
    master_config: dict[str, Any],
    epoch: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> None:
    """Update save state before checkpointing.

    Args:
        save_state: Save state dict to update in-place
        step: Current step within epoch
        consumed_samples: Total samples consumed
        val_metrics: Validation metrics (if available)
        master_config: Master configuration
        epoch: Current epoch (optional)
        total_steps: Total steps across all epochs (optional)
    """
    save_state["step"] = step
    save_state["consumed_samples"] = consumed_samples

    if epoch is not None:
        save_state["epoch"] = epoch
    if total_steps is not None:
        save_state["total_steps"] = total_steps

    # Update validation metrics if available
    if val_metrics is not None and "val_loss" in val_metrics:
        save_state["val_loss"] = val_metrics["val_loss"]
    elif "val_loss" in save_state:
        del save_state["val_loss"]

    # Validate metric-based checkpointing configuration
    metric_name = master_config["checkpointing"].get("metric_name")
    if metric_name is not None:
        if metric_name not in save_state:
            warnings.warn(
                f"You asked to save checkpoints based on {metric_name} but the metric "
                "is not found in the save state. Saving most recent k checkpoints instead."
            )
            master_config["checkpointing"]["metric_name"] = None


# ===============================================================================
# Logger Setup
# ===============================================================================


def setup_logger(logger_config: dict[str, Any], master_config: dict[str, Any]) -> Logger:
    """Setup logger and log hyperparameters.

    Args:
        logger_config: Logger configuration dict
        master_config: Full master configuration to log as hyperparameters

    Returns:
        Configured Logger instance
    """
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)
    return logger


# ===============================================================================
# Checkpointing
# ===============================================================================


def setup_checkpointing(
    checkpoint_config: CheckpointingConfig,
    default_state_fn: Callable[[], SaveStateT],
) -> tuple[CheckpointManager, SaveStateT, Optional[str]]:
    """Setup checkpointing and load previous state if resuming.

    Args:
        checkpoint_config: Checkpointing configuration
        default_state_fn: Function that returns default save state

    Returns:
        Tuple of (checkpointer, save_state, last_checkpoint_path)
    """
    checkpointer = CheckpointManager(checkpoint_config)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    save_state = cast(
        Optional[SaveStateT],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if save_state is None:
        save_state = default_state_fn()
    return checkpointer, save_state, last_checkpoint_path


def should_checkpoint(
    step: int,
    save_period: int,
    is_last_step: bool,
    timeout_checker: Any,  # TimeoutChecker
) -> tuple[bool, bool]:
    """Determine if checkpoint should be saved.

    Args:
        step: Current training step (1-indexed)
        save_period: Save checkpoint every N steps
        is_last_step: Whether this is the last step
        timeout_checker: TimeoutChecker instance

    Returns:
        Tuple of (should_save_by_step, should_save_by_timeout)
    """
    should_save_by_step = is_last_step or (step % save_period == 0)
    should_save_by_timeout = timeout_checker.check_save()
    return should_save_by_step, should_save_by_timeout


def save_training_checkpoint(
    checkpointer: CheckpointManager,
    policy: Any,  # PolicyInterface
    dataloader: StatefulDataLoader,
    save_state: dict[str, Any],
    master_config: dict[str, Any],
    step: int,
    policy_subdir: str = "policy",
    timer: Optional[Timer] = None,
) -> None:
    """Save training checkpoint.

    Args:
        checkpointer: CheckpointManager instance
        policy: Policy to save
        dataloader: Dataloader to save state
        save_state: Training state dict
        master_config: Master configuration
        step: Current step for checkpoint naming
        policy_subdir: Subdirectory name for policy (default: "policy")
        timer: Optional Timer for timing checkpoint saving
    """
    context = timer.time("checkpointing") if timer else None
    
    if context:
        context.__enter__()
    
    try:
        logging.info(f"Saving checkpoint for step {step}...")
        checkpoint_path = checkpointer.init_tmp_checkpoint(step, save_state, master_config)
        
        policy.save_checkpoint(
            weights_path=os.path.join(checkpoint_path, policy_subdir, "weights"),
            optimizer_path=os.path.join(checkpoint_path, policy_subdir, "optimizer"),
            tokenizer_path=os.path.join(checkpoint_path, policy_subdir, "tokenizer"),
        )
        
        torch.save(
            dataloader.state_dict(),
            os.path.join(checkpoint_path, "train_dataloader.pt"),
        )
        
        checkpointer.finalize_checkpoint(checkpoint_path)
        logging.info(f"  ✓ Checkpoint saved to {checkpoint_path}")
    finally:
        if context:
            context.__exit__(None, None, None)


# ===============================================================================
# Dataloader Setup
# ===============================================================================


def dict_list_collate_fn(batch: list[dict]) -> dict[str, list]:
    """Standard collate function for dict-based datasets.

    Converts a list of dicts into a dict of lists.

    Args:
        batch: List of sample dicts from the dataset

    Returns:
        Dict mapping keys to lists of values
    """
    return {k: [x[k] for x in batch] for k in batch[0]}


def setup_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    collate_fn: Optional[Callable] = None,
    last_checkpoint_path: Optional[str] = None,
    drop_last: bool = True,
) -> StatefulDataLoader:
    """Setup stateful dataloader with checkpoint restoration.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        collate_fn: Collate function (defaults to dict_list_collate_fn)
        last_checkpoint_path: Path to checkpoint for state restoration
        drop_last: Whether to drop last incomplete batch

    Returns:
        Configured StatefulDataLoader
    """
    if collate_fn is None:
        collate_fn = dict_list_collate_fn
    
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    
    if last_checkpoint_path is not None:
        dataloader_state_path = os.path.join(last_checkpoint_path, "train_dataloader.pt")
        if os.path.exists(dataloader_state_path):
            dataloader_state_dict = torch.load(dataloader_state_path)
            dataloader.load_state_dict(dataloader_state_dict)
    
    return dataloader


# ===============================================================================
# Policy Initialization
# ===============================================================================


def initialize_policy(
    cluster: RayVirtualCluster,
    policy_config: dict[str, Any],
    tokenizer: Any,  # PreTrainedTokenizerBase
    last_checkpoint_path: Optional[str],
    init_optimizer: bool = True,
    init_reference_model: bool = False,
    use_hf_checkpoint: bool = False,
    policy_subdir: str = "policy",
    use_cut_cross_entropy: bool = False,
) -> Any:  # Policy
    """Initialize a policy with common configuration.

    Args:
        cluster: RayVirtualCluster for the policy
        policy_config: Policy configuration dict
        tokenizer: Tokenizer instance
        last_checkpoint_path: Path to checkpoint (if resuming)
        init_optimizer: Whether to initialize optimizer
        init_reference_model: Whether to initialize reference model
        use_hf_checkpoint: Whether to use HuggingFace checkpoint format
        policy_subdir: Subdirectory name for policy checkpoint (default: "policy")
        use_cut_cross_entropy: Whether to use cut cross-entropy loss kernel

    Returns:
        Initialized Policy instance
    """
    # Import here to avoid circular dependency
    from rlkit.models.policy.lm_policy import Policy

    weights_path = None
    optimizer_path = None

    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / policy_subdir / "weights"
        optimizer_path = (
            Path(last_checkpoint_path) / policy_subdir / "optimizer"
            if init_optimizer
            else None
        )

    return Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=init_optimizer,
        init_reference_model=init_reference_model,
        use_hf_checkpoint=use_hf_checkpoint,
        use_cut_cross_entropy=use_cut_cross_entropy,
    )


# ===============================================================================
# Cluster Setup
# ===============================================================================


def create_cluster(
    name: str,
    cluster_config: ClusterConfig,
    max_colocated_worker_groups: int = 1,
) -> RayVirtualCluster:
    """Create a standard RayVirtualCluster.

    Args:
        name: Cluster name
        cluster_config: Cluster configuration with num_nodes and gpus_per_node
        max_colocated_worker_groups: Max colocated worker groups

    Returns:
        Initialized RayVirtualCluster
    """
    cluster = RayVirtualCluster(
        name=name,
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]] * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=max_colocated_worker_groups,
    )
    logging.info(
        f"  ✓ {name} initialized with {cluster_config['num_nodes']} nodes "
        f"× {cluster_config['gpus_per_node']} GPUs"
    )
    return cluster


# ===============================================================================
# Validation Scheduling
# ===============================================================================


def should_run_validation(
    step: int,
    val_period: int,
    val_at_start: bool,
) -> bool:
    """Determine if validation should run at this step.

    Args:
        step: Current step (0-indexed)
        val_period: Run validation every N steps (0 = disabled)
        val_at_start: Whether to run validation at step 0

    Returns:
        True if validation should run
    """
    if val_at_start and step == 0:
        return True
    return val_period > 0 and step % val_period == 0


# ===============================================================================
# Metrics Processing
# ===============================================================================


def to_numpy_scalar(value: Any) -> np.ndarray:
    """Convert a tensor or numeric value to numpy array.

    Args:
        value: Tensor or numeric value

    Returns:
        Numpy array
    """
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.array(value)


def prepare_training_metrics(
    train_results: dict[str, Any],
    additional_metrics: Optional[dict[str, Any]] = None,
    mean_keys: set[str] = {"lr", "wd", "global_valid_seqs", "global_valid_toks"},
) -> dict[str, float]:
    """Prepare and aggregate training metrics from train results.

    Args:
        train_results: Training results dict with loss, grad_norm, and all_mb_metrics
        additional_metrics: Additional metrics to include (e.g., rollout metrics)
        mean_keys: Keys to average (others will be summed)

    Returns:
        Aggregated metrics dict with scalar values
    """
    metrics = {
        "loss": to_numpy_scalar(train_results["loss"]),
        "grad_norm": to_numpy_scalar(train_results["grad_norm"]),
    }

    # Add microbatch metrics
    if "all_mb_metrics" in train_results:
        metrics.update(train_results["all_mb_metrics"])

    # Add any additional metrics (e.g., rollout metrics for GRPO)
    if additional_metrics:
        metrics.update(additional_metrics)

    # Aggregate
    return aggregate_training_metrics(metrics, mean_keys)


def aggregate_training_metrics(
    metrics_dict: dict[str, Any],
    mean_keys: set[str] = {"lr", "wd", "global_valid_seqs", "global_valid_toks"},
) -> dict[str, float]:
    """Aggregate metrics with mean for specific keys, sum for others.

    Args:
        metrics_dict: Dict of metric names to values (typically numpy arrays)
        mean_keys: Keys to average (others will be summed)

    Returns:
        Dict with aggregated scalar values
    """
    result = {}
    for k, v in metrics_dict.items():
        if k in mean_keys:
            result[k] = np.mean(v).item()
        else:
            result[k] = np.sum(v).item()
    return result


# ===============================================================================
# Logging Helpers
# ===============================================================================


def log_timing_metrics(
    timing_metrics: dict[str, float],
    step: int,
    logger: Logger,
    prefix: str = "timing/train",
) -> None:
    """Log timing metrics to console and tracking systems.

    Args:
        timing_metrics: Dict of timing measurements
        step: Current training step
        logger: Logger instance
        prefix: Prefix for logged metrics
    """
    total_time = timing_metrics.get("total_step_time", 0)

    logging.info("\n  ⏱️  Timing:")
    logging.info(f"  • Total step time: {total_time:.2f}s")

    for k, v in sorted(timing_metrics.items(), key=lambda item: item[1], reverse=True):
        if k != "total_step_time":
            percent = (v / total_time * 100) if total_time > 0 else 0
            logging.info(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

    logger.log_metrics(timing_metrics, step, prefix=prefix)


def log_validation_results(
    val_metrics: dict[str, float],
    timing_metrics: dict[str, float],
    step: int,
    logger: Logger,
    metric_names: Optional[list[str]] = None,
) -> None:
    """Log validation results to console and tracking systems.

    Args:
        val_metrics: Validation metrics
        timing_metrics: Validation timing metrics
        step: Current training step
        logger: Logger instance
        metric_names: Optional list of metric names to display (in order)
    """
    validation_time = timing_metrics.get("total_validation_time", 0)

    logging.info("\n📊 Validation Results:")

    # Log specified metrics or all metrics
    if metric_names:
        for name in metric_names:
            if name in val_metrics:
                logging.info(f"    • {name}: {val_metrics[name]:.4f}")
    else:
        for k, v in val_metrics.items():
            logging.info(f"    • {k}: {v:.4f}")

    logging.info("\n  ⏱️  Validation Timing:")
    logging.info(f"    • Total validation time: {validation_time:.2f}s")

    logger.log_metrics(val_metrics, step, prefix="validation")
    logger.log_metrics(timing_metrics, step, prefix="timing/validation")


def calculate_and_log_tflops(
    train_results: dict[str, Any],
    timing_metrics: dict[str, float],
) -> Optional[dict[str, float]]:
    """Calculate TFLOPS from training results and log to console.

    Args:
        train_results: Training results with total_flops, num_ranks, etc.
        timing_metrics: Timing metrics with policy_training time

    Returns:
        Dict with TFLOPS metrics if available, None otherwise
    """
    if "total_flops" not in train_results:
        return None

    total_tflops = (
        train_results["total_flops"]
        / timing_metrics["policy_training"]
        / 1e12
    )
    num_ranks = train_results["num_ranks"]

    logging.info(
        f"  • Training FLOPS: {total_tflops:.2f} TFLOPS "
        f"({total_tflops / num_ranks:.2f} TFLOPS per rank)"
    )

    metrics = {}

    if "theoretical_tflops" in train_results:
        theoretical_tflops = train_results["theoretical_tflops"]
        fp_utilization = total_tflops / theoretical_tflops
        logging.info(
            f"  • Training Model Floating Point Utilization: "
            f"{100 * fp_utilization:.2f}%"
        )
        metrics["train_fp_utilization"] = fp_utilization

    return metrics


def should_stop_training(
    current_step: int,
    max_steps: int,
) -> bool:
    """Check if training should stop based on max steps.

    Args:
        current_step: Current training step (0-indexed)
        max_steps: Maximum number of steps

    Returns:
        True if training should stop
    """
    return max_steps != -1 and current_step >= max_steps
