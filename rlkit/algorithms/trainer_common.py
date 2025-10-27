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
from typing import Any, Callable, Optional, TypedDict, TypeVar, cast

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from rlkit.config import CheckpointingConfig, ClusterConfig
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import Logger
from rlkit.utils.timer import Timer

SaveStateT = TypeVar("SaveStateT", bound=TypedDict)


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
