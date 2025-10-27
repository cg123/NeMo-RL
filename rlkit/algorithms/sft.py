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
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict, cast

from datasets import Dataset
import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rlkit.algorithms.loss_functions import (
    NLLLoss,
)
from rlkit.algorithms.utils import set_seed, _pad_tensor
from rlkit.algorithms import trainer_common
from rlkit.config import (
    ClusterConfig,
    CheckpointingConfig,
    DataConfig,
    LoggerConfig,
    PolicyConfig,
    SFTConfig,
    SFTMasterConfig as MasterConfig,
)
from rlkit.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from rlkit.distributed.batched_data_dict import BatchedDataDict
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.models.policy.interfaces import PolicyInterface
from rlkit.models.policy.lm_policy import Policy
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import Logger
from rlkit.utils.nsys import maybe_gpu_profile_step
from rlkit.utils.timer import TimeoutChecker, Timer


class SFTSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: float  # Optional field - may not be present during training
    consumed_samples: int


def _default_sft_save_state() -> SFTSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
    }


class SFTTrainer:
    """Encapsulates setup, validation, and training logic for SFT."""

    def __init__(
        self,
        master_config: MasterConfig,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
    ) -> None:
        self.master_config = master_config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        policy_config = self.master_config["policy"]
        cluster_config = self.master_config["cluster"]

        set_seed(master_config["sft"]["seed"])

        self.logger = trainer_common.setup_logger(master_config["logger"], self.master_config)

        (
            self.checkpointer,
            self.sft_save_state,
            last_checkpoint_path,
        ) = trainer_common.setup_checkpointing(master_config["checkpointing"], _default_sft_save_state)

        (
            self.train_dataloader,
            self.val_dataloader,
        ) = self._setup_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_config=master_config["data"],
            policy_config=master_config["policy"],
            sft_config=master_config["sft"],
            last_checkpoint_path=last_checkpoint_path,
        )

        logging.info("Setting up compute cluster...")

        self.cluster = trainer_common.create_cluster("sft_train_cluster", cluster_config)

        if last_checkpoint_path:
            weights_path = Path(last_checkpoint_path) / "policy" / "weights"
            optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
        else:
            weights_path = None
            optimizer_path = None

        self.use_hf_checkpoint = self.master_config["checkpointing"].get("hf_checkpoint", False)

        self.policy = self._initialize_policy(self.cluster, policy_config, self.tokenizer, weights_path, optimizer_path)

        self.loss_fn = NLLLoss()

    def _setup_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        data_config: DataConfig,
        policy_config: PolicyConfig,
        sft_config: SFTConfig,
        last_checkpoint_path: Optional[str],
    ) -> tuple[
        StatefulDataLoader,
        Optional[StatefulDataLoader],
    ]:
        train_dataloader = trainer_common.setup_dataloader(
            train_dataset,
            batch_size=policy_config["train_global_batch_size"],
            shuffle=data_config["shuffle"],
            collate_fn=trainer_common.dict_list_collate_fn,
            last_checkpoint_path=last_checkpoint_path,
            drop_last=True,
        )

        val_dataloader: Optional[StatefulDataLoader] = None
        if val_dataset is not None:
            val_dataloader = trainer_common.setup_dataloader(
                val_dataset,
                batch_size=sft_config["val_global_batch_size"],
                shuffle=False,
                collate_fn=trainer_common.dict_list_collate_fn,
                last_checkpoint_path=None,
                drop_last=False,
            )

        return train_dataloader, val_dataloader

    def _initialize_policy(
        self,
        train_cluster: RayVirtualCluster,
        policy_config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        weights_path: Optional[Path],
        optimizer_path: Optional[Path],
    ) -> Policy:
        use_cce = self.master_config["sft"].get("use_cut_cross_entropy", False)
        if use_cce:
            logging.info("Using cut cross-entropy loss kernel")

        return trainer_common.initialize_policy(
            cluster=train_cluster,
            policy_config=policy_config,
            tokenizer=tokenizer,
            last_checkpoint_path=str(weights_path.parent.parent) if weights_path else None,
            init_optimizer=True,
            init_reference_model=False,
            use_hf_checkpoint=self.use_hf_checkpoint,
            use_cut_cross_entropy=use_cce,
        )

    async def validate(self, step: int) -> Optional[tuple[dict[str, float], dict[str, float]]]:
        """Run validation on the validation dataset."""
        sft_config = self.master_config["sft"]

        def process_batch_fn(raw_batch):
            """Process raw batch into model input."""
            batch = BatchedDataDict(raw_batch)
            return trainer_common.process_supervised_batch(
                batch,
                max_seq_len=self.master_config["policy"]["max_total_sequence_length"],
                tokenizer_pad_token_id=self.tokenizer.pad_token_id,
            )

        def accumulate_metrics_fn(val_metrics, val_results):
            """Accumulate validation metrics."""
            val_metrics["val_loss"] += float(val_results["loss"])

        result = await trainer_common.run_validation_loop(
            val_dataloader=self.val_dataloader,
            policy=self.policy,
            loss_fn=self.loss_fn,
            step=step,
            logger=self.logger,
            max_val_batches=sft_config["val_batches"],
            val_global_batch_size=sft_config["val_global_batch_size"],
            val_micro_batch_size=sft_config["val_micro_batch_size"],
            process_batch_fn=process_batch_fn,
            metric_names=["val_loss"],
            accumulate_metrics_fn=accumulate_metrics_fn,
        )

        return result

    async def train(self) -> None:
        timer = Timer()
        timeout = TimeoutChecker(
            timeout=self.master_config["checkpointing"]["checkpoint_must_save_by"],
            fit_last_save_time=True,
        )
        timeout.start_iterations()

        current_epoch = self.sft_save_state.get("epoch", 0)
        current_step = self.sft_save_state.get("step", 0)
        total_steps = self.sft_save_state.get("total_steps", 0)

        sft_config = self.master_config["sft"]
        val_period = sft_config["val_period"]
        val_at_start = sft_config["val_at_start"]
        max_num_epochs = sft_config["max_num_epochs"]

        if trainer_common.should_validate_now(total_steps, val_period, val_at_start):
            print("\n🔍 Running initial validation...")
            validation_result = await self.validate(step=0)
            if validation_result is not None:
                val_metrics, validation_timings = validation_result
                self.logger.log_metrics(val_metrics, total_steps, prefix="validation")
                self.logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

        self.policy.prepare_for_training()

        while current_epoch < max_num_epochs and total_steps < self.master_config["sft"]["max_num_steps"]:
            logging.info(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

            for raw_batch in self.train_dataloader:
                logging.info(
                    f"\n{'=' * 25} Step {current_step + 1}/{min(len(self.train_dataloader), self.master_config['sft']['max_num_steps'])} {'=' * 25}"
                )

                batch = BatchedDataDict(raw_batch)

                maybe_gpu_profile_step(self.policy, total_steps + 1)
                val_metrics, validation_timings = None, None

                with timer.time("total_step_time"):
                    logging.info("Preparing batch...")
                    with timer.time("data_processing"):
                        train_data = trainer_common.process_supervised_batch(
                            batch,
                            max_seq_len=self.master_config["policy"]["max_total_sequence_length"],
                            tokenizer_pad_token_id=self.tokenizer.pad_token_id,
                            run_vram_torture_test=self.master_config["sft"].get("run_vram_torture_test", False),
                        )

                    logging.info("Taking a training step...")
                    with timer.time("policy_training"):
                        train_results = await self.policy.train(train_data, self.loss_fn)

                    is_last_step = total_steps + 1 >= self.master_config["sft"]["max_num_steps"] or (
                        current_epoch + 1 == max_num_epochs and current_step + 1 == len(self.train_dataloader)
                    )

                    if trainer_common.should_validate_now(total_steps + 1, val_period, val_at_start):
                        logging.info("Running validation...")
                        validation_result = await self.validate(step=total_steps + 1)
                        if validation_result is not None:
                            val_metrics, validation_timings = validation_result
                            self.logger.log_metrics(
                                validation_timings,
                                total_steps + 1,
                                prefix="timing/validation",
                            )
                            self.logger.log_metrics(val_metrics, total_steps + 1, prefix="validation")

                    self.sft_save_state["consumed_samples"] += self.master_config["policy"]["train_global_batch_size"]
                    timeout.mark_iteration()
                    should_save_by_step, should_save_by_timeout = trainer_common.should_checkpoint(
                        total_steps + 1,
                        self.master_config["checkpointing"]["save_period"],
                        is_last_step,
                        timeout,
                    )

                    if self.master_config["checkpointing"]["enabled"] and (
                        should_save_by_step or should_save_by_timeout
                    ):
                        trainer_common.update_save_state_for_checkpoint(
                            self.sft_save_state,
                            step=(current_step + 1) % len(self.train_dataloader),
                            consumed_samples=self.sft_save_state["consumed_samples"],
                            val_metrics=val_metrics,
                            master_config=self.master_config,
                            epoch=current_epoch,
                            total_steps=total_steps + 1,
                        )

                        trainer_common.save_training_checkpoint(
                            self.checkpointer,
                            self.policy,
                            self.train_dataloader,
                            self.sft_save_state,
                            self.master_config,
                            total_steps + 1,
                            policy_subdir="policy",
                            timer=timer,
                        )

                metrics = trainer_common.prepare_training_metrics(train_results)

                self._log_step(metrics, timer, train_results, total_steps)

                timer.reset()
                current_step += 1
                total_steps += 1

                if total_steps >= self.master_config["sft"]["max_num_steps"]:
                    return

            current_epoch += 1
            current_step = 0

    def _log_step(
        self,
        metrics: dict[str, Any],
        timer: Timer,
        train_results: dict[str, Any],
        total_steps: int,
    ) -> None:
        print("\n📊 Training Results:")
        print(f"  • Loss: {float(metrics['loss']):.4f}")

        timing_metrics = timer.get_timing_metrics(reduction_op="sum")

        if "total_flops" in train_results:
            tflops_metrics = trainer_common.calculate_and_log_tflops(train_results, timing_metrics)
            if tflops_metrics:
                metrics.update(tflops_metrics)

            total_valid_toks = train_results["all_mb_metrics"]["global_valid_toks"][0]
            print(f"  • Total valid tokens: {total_valid_toks}")
            print(
                f"  • Mean microbatch tokens: {total_valid_toks / len(train_results['all_mb_metrics']['global_valid_toks']):.0f}"
            )
            print(f"  • Estimated throughput: {total_valid_toks / timing_metrics['policy_training']:.2f} tok/s")

        self.logger.log_metrics(metrics, total_steps + 1, prefix="train")
        trainer_common.log_timing_metrics(timing_metrics, total_steps + 1, self.logger, prefix="timing/train")
