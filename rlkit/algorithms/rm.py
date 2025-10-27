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
import os
import warnings
from pathlib import Path
from typing import Optional, TypedDict, cast

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from rlkit.algorithms.loss_functions import (
    PreferenceLoss,
)
from rlkit.algorithms.utils import set_seed
from rlkit.algorithms import trainer_common
from rlkit.config import (
    ClusterConfig,
    CheckpointingConfig,
    DataConfig,
    LoggerConfig,
    PolicyConfig,
    RMConfig,
    RMMasterConfig as MasterConfig,
)
from rlkit.data.datasets import (
    AllTaskProcessedDataset,
    preference_collate_fn,
)
from rlkit.data.interfaces import TaskDataSpec
from rlkit.distributed.batched_data_dict import BatchedDataDict
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.models.policy.interfaces import PolicyInterface
from rlkit.models.policy.lm_policy import Policy
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import Logger
from rlkit.utils.nsys import maybe_gpu_profile_step
from rlkit.utils.timer import Timer
from rlkit.utils.timer import TimeoutChecker


class RMSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: float
    consumed_samples: int


def _default_rm_save_state() -> RMSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
    }


class RMValMetrics(TypedDict):
    val_loss: float
    accuracy: float
    rewards_chosen_mean: float
    rewards_rejected_mean: float
    num_valid_samples: float


# =======================================================
# Setup & Initialization
# =======================================================
def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: AllTaskProcessedDataset,
) -> tuple[
    Policy,
    RayVirtualCluster,
    StatefulDataLoader,
    StatefulDataLoader,
    PreferenceLoss,
    MasterConfig,
    Logger,
    TaskDataSpec,
    RMSaveState,
]:
    """Main entry point for running RM algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, master_config, logger
    """
    set_seed(master_config["rm"]["seed"])

    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    rm_config = master_config["rm"]

    # ==========================
    #         Logger
    # ==========================
    logger = trainer_common.setup_logger(logger_config, master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer, rm_save_state, last_checkpoint_path = trainer_common.setup_checkpointing(
        master_config["checkpointing"], _default_rm_save_state
    )

    # ==========================
    #           Data
    # ==========================
    train_dataloader = trainer_common.setup_dataloader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=data_config["shuffle"],
        collate_fn=preference_collate_fn,
        last_checkpoint_path=last_checkpoint_path,
        drop_last=True,
    )

    val_dataloader = trainer_common.setup_dataloader(
        val_dataset,
        batch_size=rm_config["val_global_batch_size"],
        shuffle=False,
        collate_fn=preference_collate_fn,
        last_checkpoint_path=None,
        drop_last=True,
    )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = trainer_common.create_cluster("rm_cluster", cluster_config)

    # ==========================
    #   Training
    # ==========================
    print("\n▶ Setting up model...")
    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=Path(last_checkpoint_path) / "policy" / "weights" if last_checkpoint_path else None,
        optimizer_path=Path(last_checkpoint_path) / "policy" / "optimizer" if last_checkpoint_path else None,
        init_optimizer=True,
        init_reference_model=False,
    )
    loss_fn = PreferenceLoss()
    print("  ✓ Model initialized")

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        rm_save_state,
        master_config,
    )


# =======================================================
# Training & Validation
# =======================================================
async def validate(
    policy: PolicyInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    rm_task_spec: TaskDataSpec,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
    logger,
):
    """Run validation on the validation dataset."""

    def process_batch_fn(raw_batch):
        """Process raw batch into model input."""
        return trainer_common.process_preference_batch(
            raw_batch,
            tokenizer=tokenizer,
            max_seq_len=master_config["policy"]["make_sequence_length_divisible_by"],
            roles_to_train_on=["assistant"],
        )

    # Track sample counts for weighted averaging
    sample_counts = []

    def accumulate_metrics_fn(val_metrics, val_results):
        """Accumulate validation metrics with sample-weighted averaging."""
        # Sum metrics across microbatches
        num_valid_samples = sum(val_results["all_mb_metrics"]["num_valid_samples"])
        sample_counts.append(num_valid_samples)

        val_metrics["val_loss"] += sum(val_results["all_mb_metrics"]["loss"]) * num_valid_samples
        val_metrics["accuracy"] += sum(val_results["all_mb_metrics"]["accuracy"]) * num_valid_samples
        val_metrics["rewards_chosen_mean"] += (
            sum(val_results["all_mb_metrics"]["rewards_chosen_mean"]) * num_valid_samples
        )
        val_metrics["rewards_rejected_mean"] += (
            sum(val_results["all_mb_metrics"]["rewards_rejected_mean"]) * num_valid_samples
        )
        val_metrics["num_valid_samples"] += num_valid_samples

    result = await trainer_common.run_validation_loop(
        val_dataloader=val_dataloader,
        policy=policy,
        loss_fn=loss_fn,
        step=step,
        logger=logger,
        max_val_batches=val_batches,
        # NOTE: we double the batch size here because each preference example corresponds to a pair of
        # examples, chosen and rejected, and the pair needs to be processed as part of the same microbatch.
        val_global_batch_size=val_batch_size * 2,
        val_micro_batch_size=val_mbs * 2,
        process_batch_fn=process_batch_fn,
        metric_names=["val_loss", "accuracy", "rewards_chosen_mean", "rewards_rejected_mean", "num_valid_samples"],
        accumulate_metrics_fn=accumulate_metrics_fn,
    )

    if result is None:
        return None

    val_metrics, timing_metrics = result

    # Perform weighted averaging (divide by total num_valid_samples)
    total_samples = val_metrics.get("num_valid_samples", 1)
    if total_samples > 0:
        val_metrics["val_loss"] /= total_samples
        val_metrics["accuracy"] /= total_samples
        val_metrics["rewards_chosen_mean"] /= total_samples
        val_metrics["rewards_rejected_mean"] /= total_samples

    return val_metrics, timing_metrics


async def rm_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    rm_task_spec,
    checkpointer,
    rm_save_state,
):
    # Run basic rm training
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    if rm_save_state is None:
        rm_save_state = _default_rm_save_state()
        current_epoch = 0
        current_step = 0
        total_steps = 0
    else:
        current_epoch = rm_save_state["epoch"]
        current_step = rm_save_state["step"]
        total_steps = rm_save_state["total_steps"]

    rm_config = master_config["rm"]
    # Validation configuration
    val_period = rm_config["val_period"]
    val_at_start = rm_config["val_at_start"]
    max_num_epochs = rm_config["max_num_epochs"]

    # Run validation at the start if configured
    if trainer_common.should_validate_now(total_steps, val_period, val_at_start):
        print("\n🔍 Running initial validation...")
        val_metrics, validation_timings = await validate(
            policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
            rm_task_spec=rm_task_spec,
            val_batches=rm_config["val_batches"],
            val_batch_size=rm_config["val_global_batch_size"],
            val_mbs=rm_config["val_micro_batch_size"],
            logger=logger,
        )

        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    policy.prepare_for_training()

    while current_epoch < max_num_epochs and (
        master_config["rm"]["max_num_steps"] == -1 or total_steps < master_config["rm"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in train_dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config['rm']['max_num_steps'] if master_config['rm']['max_num_steps'] != -1 else len(train_dataloader))} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch and generate responses
                print("▶ Preparing batch...")
                with timer.time("data_processing"):
                    train_data = trainer_common.process_preference_batch(
                        batch,
                        tokenizer=tokenizer,
                        max_seq_len=master_config["policy"]["make_sequence_length_divisible_by"],
                        roles_to_train_on=["assistant"],
                    )

                print("▶ Taking a training step...")

                train_results = await policy.train(
                    train_data,
                    loss_fn,
                    eval_mode=False,
                    ## NOTE: we double the batch size here because each preference example corresponds to a pair of
                    ## examples, chosen and rejected, and the pair needs to be processed as part of the same microbatch.
                    gbs=master_config["policy"]["train_global_batch_size"] * 2,
                    mbs=master_config["policy"]["train_micro_batch_size"] * 2,
                )

                is_last_step = (
                    master_config["rm"]["max_num_steps"] != -1
                    and total_steps + 1 >= master_config["rm"]["max_num_steps"]
                ) or (current_epoch + 1 == max_num_epochs and current_step + 1 == len(train_dataloader))

                # Run validation if it's a validation step
                if trainer_common.should_validate_now(total_steps + 1, val_period, val_at_start):
                    val_metrics, validation_timings = await validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        rm_task_spec=rm_task_spec,
                        val_batches=rm_config["val_batches"],
                        val_batch_size=rm_config["val_global_batch_size"],
                        val_mbs=rm_config["val_micro_batch_size"],
                        logger=logger,
                    )
                    logger.log_metrics(validation_timings, total_steps + 1, prefix="timing/validation")
                    logger.log_metrics(val_metrics, total_steps + 1, prefix="validation")

                ## Checkpointing
                rm_save_state["consumed_samples"] += master_config["policy"]["train_global_batch_size"]
                timeout.mark_iteration()
                should_save_by_step, should_save_by_timeout = trainer_common.should_checkpoint(
                    total_steps + 1,
                    master_config["checkpointing"]["save_period"],
                    is_last_step,
                    timeout,
                )

                if master_config["checkpointing"]["enabled"] and (should_save_by_step or should_save_by_timeout):
                    trainer_common.update_save_state_for_checkpoint(
                        rm_save_state,
                        step=(current_step + 1) % len(train_dataloader),
                        consumed_samples=rm_save_state["consumed_samples"],
                        val_metrics=val_metrics,
                        master_config=master_config,
                        epoch=current_epoch,
                        total_steps=total_steps + 1,
                    )

                    trainer_common.save_training_checkpoint(
                        checkpointer,
                        policy,
                        train_dataloader,
                        rm_save_state,
                        master_config,
                        total_steps + 1,
                        policy_subdir="policy",
                        timer=timer,
                    )

            losses = train_results["loss"]
            metrics = {
                "loss": train_results["loss"].numpy(),
                "grad_norm": train_results["grad_norm"].numpy(),
            }
            metrics.update(train_results["all_mb_metrics"])
            metrics = trainer_common.aggregate_training_metrics(metrics)
            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

            print("\n📊 Training Results:")
            print(f"  • Loss: {float(metrics['loss']):.4f}")
            print(f"  • Accuracy: {float(metrics['accuracy']):.4f}")
            print(f"  • Rewards chosen mean: {float(metrics['rewards_chosen_mean']):.4f}")
            print(f"  • Rewards rejected mean: {float(metrics['rewards_rejected_mean']):.4f}")
            print(f"  • Num valid samples: {float(metrics['num_valid_samples']):.0f}")

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            trainer_common.log_timing_metrics(timing_metrics, total_steps + 1, logger, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1

            if master_config["rm"]["max_num_steps"] != -1 and total_steps >= master_config["rm"]["max_num_steps"]:
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
