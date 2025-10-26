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

import asyncio
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

from datasets import Dataset
import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from rlkit.algorithms.loss_functions import NLLLoss, CombinedKDLoss, KnowledgeDistillationLoss
from rlkit.algorithms.utils import set_seed, _pad_tensor
from rlkit.config import (
    KDConfig,
    KDLoggerConfig,
    KDMasterConfig as MasterConfig,
    TeacherConfig,
    PolicyConfig,
    CheckpointingConfig,
    ClusterConfig,
    DataConfig,
    KD_DEFAULT_ALPHA,
    KD_DEFAULT_TEMPERATURE,
    KD_DEFAULT_VAL_PERIOD,
    KD_DEFAULT_VAL_AT_START,
)
from rlkit.distributed.batched_data_dict import BatchedDataDict
from rlkit.distributed.virtual_cluster import RayVirtualCluster
from rlkit.models.policy.lm_policy import Policy
from rlkit.utils.checkpoint import CheckpointManager
from rlkit.utils.logger import Logger
from rlkit.utils.timer import TimeoutChecker, Timer
from rlkit.utils.nsys import maybe_gpu_profile_step

import ray


# Default timeout values for KD operations
DEFAULT_TEACHER_INFERENCE_TIMEOUT = 300  # 5 minutes
DEFAULT_VALIDATION_TIMEOUT = 600  # 10 minutes
DEFAULT_CHECKPOINTING_TIMEOUT = 120  # 2 minutes

# Default validation settings
DEFAULT_VAL_PERIOD = 100  # Validate every 100 steps
DEFAULT_VAL_BATCHES = 10  # Number of validation batches


class KDValidationMetrics(TypedDict):
    """Validation metrics for knowledge distillation."""
    val_loss: float
    val_base_loss: float
    val_kd_loss: float


class KDTimingMetrics(TypedDict, total=False):
    """Timing metrics for KD training."""
    total_validation_time: float
    total_step_time: float
    data_processing: float
    teacher_inference: float
    student_training: float


class KDSaveState(TypedDict):
    """Training state for knowledge distillation."""
    epoch: int
    step: int
    total_steps: int
    val_loss: float
    consumed_samples: int


def _default_kd_save_state() -> KDSaveState:
    """Create default KD save state."""
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
    }


class KDTrainer:
    """Knowledge Distillation trainer.
    
    Trains a student model to mimic a frozen teacher model using
    combined supervised + distillation loss.
    
    The trainer manages two separate clusters:
    - Student cluster: For trainable student policy (forward + backward)
    - Teacher cluster: For frozen teacher inference (forward only)
    """
    
    def __init__(
        self,
        master_config: MasterConfig,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
    ) -> None:
        """Initialize KD trainer.
        
        Args:
            master_config: Complete KD configuration
            tokenizer: Tokenizer (must be compatible with both student and teacher)
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
        """
        self.master_config = master_config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Extract configs
        student_policy_config = master_config["student_policy"]
        teacher_config = master_config["teacher"]
        kd_config = master_config["kd"]
        cluster_config = master_config["cluster"]
        
        set_seed(kd_config["seed"])
        
        # Setup logger
        logging.info("Setting up logger...")
        self.logger = self._setup_logger(master_config["logger"])
        
        # Setup checkpointing
        logging.info("Setting up checkpointing...")
        (
            self.checkpointer,
            self.kd_save_state,
            last_checkpoint_path,
        ) = self._setup_checkpointing(master_config["checkpointing"])
        
        # Setup dataloaders
        logging.info("Setting up dataloaders...")
        (
            self.train_dataloader,
            self.val_dataloader,
        ) = self._setup_dataloaders(
            train_dataset,
            val_dataset,
            master_config["data"],
            student_policy_config,
            kd_config,
            last_checkpoint_path,
        )
        
        # Setup clusters (student + teacher)
        logging.info("Setting up compute clusters...")
        (
            self.student_cluster,
            self.teacher_cluster,
        ) = self._setup_clusters(cluster_config, teacher_config)
        
        # Load checkpoint paths if resuming
        if last_checkpoint_path:
            student_weights_path = Path(last_checkpoint_path) / "student" / "weights"
            student_optimizer_path = Path(last_checkpoint_path) / "student" / "optimizer"
        else:
            student_weights_path = None
            student_optimizer_path = None
        
        # Initialize student policy (trainable)
        logging.info("Initializing student policy...")
        self.student_policy = self._initialize_student_policy(
            self.student_cluster,
            student_policy_config,
            tokenizer,
            student_weights_path,
            student_optimizer_path,
        )
        
        # Initialize teacher policy (frozen)
        logging.info("Initializing teacher policy...")
        self.teacher_policy = self._initialize_teacher_policy(
            self.teacher_cluster,
            teacher_config,
            tokenizer,
        )
        
        # Validate tokenizers match
        self._validate_tokenizers()
        
        # Setup loss function
        logging.info("Setting up loss function...")
        base_loss = NLLLoss()
        
        # Get KD hyperparameters with defaults
        alpha = kd_config.get("alpha", KD_DEFAULT_ALPHA)
        temperature = kd_config.get("temperature", KD_DEFAULT_TEMPERATURE)
        
        kd_loss = KnowledgeDistillationLoss(temperature=temperature)
        
        self.loss_fn = CombinedKDLoss(
            base_loss=base_loss,
            kd_loss=kd_loss,
            alpha=alpha,
        )
        
        logging.info(f"  ✓ KD loss: alpha={alpha}, temperature={temperature}")
        
        logging.info("  ✓ KD Trainer initialized successfully")
    
    def _setup_logger(self, logger_config: KDLoggerConfig) -> Logger:
        """Setup logger and log hyperparameters."""
        logger = Logger(logger_config)
        logger.log_hyperparams(self.master_config)
        return logger
    
    def _setup_checkpointing(
        self, checkpoint_config: CheckpointingConfig
    ) -> tuple[CheckpointManager, KDSaveState, Optional[str]]:
        """Setup checkpointing and load previous state if resuming."""
        checkpointer = CheckpointManager(checkpoint_config)
        last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
        kd_save_state = cast(
            Optional[KDSaveState],
            checkpointer.load_training_info(last_checkpoint_path),
        )
        if kd_save_state is None:
            kd_save_state = _default_kd_save_state()
        return checkpointer, kd_save_state, last_checkpoint_path
    
    def _setup_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        data_config: DataConfig,
        policy_config: PolicyConfig,
        kd_config: KDConfig,
        last_checkpoint_path: Optional[str],
    ) -> tuple[StatefulDataLoader, Optional[StatefulDataLoader]]:
        """Setup train/val dataloaders (reuse SFT pattern)."""
        
        # Collate function for supervised learning data
        def collate_fn(batch):
            return {k: [x[k] for x in batch] for k in batch[0]}
        
        # Calculate batch size from global batch size and micro batch size
        train_batch_size = policy_config["train_global_batch_size"] // policy_config["train_micro_batch_size"]
        
        train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=data_config["shuffle"],
            drop_last=True,
            collate_fn=collate_fn,
        )
        
        # Restore dataloader state if resuming
        if last_checkpoint_path is not None:
            dataloader_state_path = os.path.join(
                last_checkpoint_path, "train_dataloader.pt"
            )
            if os.path.exists(dataloader_state_path):
                dataloader_state_dict = torch.load(dataloader_state_path)
                train_dataloader.load_state_dict(dataloader_state_dict)
        
        logging.info(f"  ✓ Training dataloader loaded with {len(train_dataset)} samples")
        
        # Setup validation dataloader if enabled
        val_dataloader = None
        if kd_config["val_period"] > 0 or kd_config["val_at_start"]:
            assert val_dataset is not None, "Validation dataset required if validation enabled"
            val_batch_size = kd_config["val_global_batch_size"] // kd_config["val_micro_batch_size"]
            val_dataloader = StatefulDataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            logging.info(f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples")
        
        return train_dataloader, val_dataloader
    
    def _validate_cluster_allocation(
        self, cluster_config: ClusterConfig, teacher_cluster_config: dict
    ) -> None:
        """Validate that cluster resources are properly allocated.
        
        Ensures that teacher + student don't exceed total available resources.
        
        Args:
            cluster_config: Total cluster resources
            teacher_cluster_config: Teacher's dedicated cluster allocation
            
        Raises:
            ValueError: If resource allocation is invalid
        """
        total_gpus = cluster_config["num_nodes"] * cluster_config["gpus_per_node"]
        teacher_gpus = teacher_cluster_config["num_nodes"] * teacher_cluster_config["gpus_per_node"]
        student_gpus = total_gpus - teacher_gpus
        
        if teacher_gpus <= 0:
            raise ValueError(
                f"Teacher cluster allocation is invalid: {teacher_gpus} GPUs. "
                "Must allocate at least 1 GPU for teacher."
            )
        
        if student_gpus <= 0:
            raise ValueError(
                f"No GPUs available for student after allocating {teacher_gpus} GPUs to teacher. "
                f"Total cluster has {total_gpus} GPUs. "
                "Please increase total cluster size or decrease teacher cluster allocation."
            )
        
        if teacher_gpus > total_gpus:
            raise ValueError(
                f"Teacher cluster allocation ({teacher_gpus} GPUs) exceeds total "
                f"cluster resources ({total_gpus} GPUs)."
            )
        
        logging.info(f"✓ Cluster allocation validated:")
        logging.info(f"  • Total GPUs: {total_gpus}")
        logging.info(f"  • Teacher GPUs: {teacher_gpus}")
        logging.info(f"  • Student GPUs: {student_gpus}")
    
    def _setup_clusters(
        self,
        cluster_config: ClusterConfig,
        teacher_config: TeacherConfig,
    ) -> tuple[RayVirtualCluster, RayVirtualCluster]:
        """Setup separate clusters for student training and teacher inference.
        
        Follows GRPO's pattern of separate train/inference clusters.
        """
        # Teacher cluster (explicit allocation)
        teacher_num_nodes = teacher_config["cluster"]["num_nodes"]
        teacher_gpus_per_node = teacher_config["cluster"]["gpus_per_node"]
        
        teacher_cluster = RayVirtualCluster(
            name="kd_teacher_cluster",
            bundle_ct_per_node_list=[teacher_gpus_per_node] * teacher_num_nodes,
            use_gpus=True,
            num_gpus_per_node=teacher_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        logging.info(
            f"  ✓ Teacher cluster: {teacher_num_nodes} nodes × {teacher_gpus_per_node} GPUs"
        )
        
        # Student cluster (remaining resources)
        total_nodes = cluster_config["num_nodes"]
        total_gpus_per_node = cluster_config["gpus_per_node"]
        
        # Validate cluster allocation
        # Calculate total GPUs
        total_gpus = total_nodes * total_gpus_per_node
        teacher_total_gpus = teacher_num_nodes * teacher_gpus_per_node
        
        # Check if teacher allocation is valid
        if teacher_num_nodes > total_nodes:
            raise ValueError(
                f"Invalid cluster allocation: Teacher requires {teacher_num_nodes} nodes, "
                f"but only {total_nodes} total nodes available.\n"
                f"  Teacher GPUs: {teacher_num_nodes} nodes × {teacher_gpus_per_node} GPUs/node = {teacher_total_gpus} GPUs\n"
                f"  Total GPUs: {total_nodes} nodes × {total_gpus_per_node} GPUs/node = {total_gpus} GPUs\n"
                f"Solution: Reduce teacher.cluster.num_nodes or increase cluster.num_nodes"
            )
        
        if teacher_gpus_per_node != total_gpus_per_node:
            raise ValueError(
                f"Invalid cluster allocation: teacher.cluster.gpus_per_node ({teacher_gpus_per_node}) "
                f"must equal cluster.gpus_per_node ({total_gpus_per_node}).\n"
                f"NeMo RL requires uniform GPUs per node across the cluster."
            )
        
        student_num_nodes = total_nodes - teacher_num_nodes
        student_gpus_per_node = total_gpus_per_node
        student_total_gpus = student_num_nodes * student_gpus_per_node
        
        if student_num_nodes == 0:
            raise ValueError(
                f"Invalid cluster allocation: No nodes remaining for student training.\n"
                f"  Total nodes: {total_nodes}\n"
                f"  Teacher nodes: {teacher_num_nodes}\n"
                f"  Student nodes: {student_num_nodes} (= total - teacher)\n"
                f"Solution: Increase cluster.num_nodes or reduce teacher.cluster.num_nodes"
            )
        
        logging.info(
            f"  ✓ Cluster allocation validated:")
        logging.info(
            f"    - Total: {total_nodes} nodes × {total_gpus_per_node} GPUs = {total_gpus} GPUs"
        )
        logging.info(
            f"    - Teacher: {teacher_num_nodes} nodes × {teacher_gpus_per_node} GPUs = {teacher_total_gpus} GPUs"
        )
        logging.info(
            f"    - Student: {student_num_nodes} nodes × {student_gpus_per_node} GPUs = {student_total_gpus} GPUs"
        )
        
        student_cluster = RayVirtualCluster(
            name="kd_student_cluster",
            bundle_ct_per_node_list=[student_gpus_per_node] * student_num_nodes,
            use_gpus=True,
            num_gpus_per_node=student_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        logging.info(
            f"  ✓ Student cluster: {student_num_nodes} nodes × {student_gpus_per_node} GPUs"
        )
        
        return student_cluster, teacher_cluster
    
    def _initialize_student_policy(
    self,
    cluster: RayVirtualCluster,
    policy_config: PolicyConfig,
    tokenizer: AutoTokenizer,
    weights_path: Optional[Path],
    optimizer_path: Optional[Path],
) -> Policy:
    """Initialize trainable student policy.
    
    NOTE: Vocab parallelism (tensor_parallel_size > 1) is currently not supported
    for knowledge distillation. See validation below for details.
    """
    # Validate no vocab parallelism before initializing
    student_tp_size = policy_config["dtensor_v2_cfg"].get("tensor_parallel_size", 1)
    if student_tp_size > 1:
        raise NotImplementedError(
            f"Knowledge Distillation does not currently support student vocab parallelism. "
            f"Student is configured with tensor_parallel_size={student_tp_size}, "
            f"but KD requires computing KL divergence over the full vocabulary distribution.\n"
            f"\n"
            f"Current issue: KnowledgeDistillationLoss computes log_softmax on the student logits, "
            f"which with TP only contains a vocab shard [batch, seq, vocab_size/tp_size]. "
            f"This produces incorrect probability distributions and KL divergence values.\n"
            f"\n"
            f"Solutions:\n"
            f"  1. Set student_policy.dtensor_v2_cfg.tensor_parallel_size=1 (recommended)\n"
            f"  2. Use data parallelism instead (increase num_nodes)\n"
            f"  3. Use pipeline parallelism if the student model is large\n"
            f"\n"
            f"Note: Student models for KD are typically small (1B-7B) and don't require TP. "
            f"If you need TP for the student, consider if distillation is the right approach.\n"
            f"\n"
            f"Future work: Support for matched TP sharding between teacher and student is planned. "
            f"This would allow computing KL divergence correctly across vocab shards using "
            f"distributed partition function computation. Estimated effort: 3-5 days."
        )
    
    return Policy(
            cluster=cluster,
            config=policy_config,
            tokenizer=tokenizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=True,  # Student is trainable
            init_reference_model=False,  # No reference model needed for KD
            use_hf_checkpoint=self.master_config["checkpointing"].get("hf_checkpoint", False),
        )
    
    def _initialize_teacher_policy(
    self,
    cluster: RayVirtualCluster,
    teacher_config: TeacherConfig,
    tokenizer: AutoTokenizer,
) -> Policy:
    """Initialize frozen teacher policy (inference only).
    
    Teacher policy:
    - No optimizer (frozen weights)
    - No reference model
    - Set to eval mode (disables dropout)
    
    NOTE: Vocab parallelism (tensor_parallel_size > 1) is currently not supported
    for knowledge distillation. See validation below for details.
    """
    # Validate no vocab parallelism before initializing
    teacher_tp_size = teacher_config.get("tensor_parallel_size", 1)
    if teacher_tp_size > 1:
        raise NotImplementedError(
            f"Knowledge Distillation does not currently support teacher vocab parallelism. "
            f"Teacher is configured with tensor_parallel_size={teacher_tp_size}, "
            f"but KD requires the full vocabulary distribution from the teacher.\n"
            f"\n"
            f"Current issue: Policy.get_logprobs() returns per-token log probabilities "
            f"with shape [batch, seq_len], not full distributions [batch, seq_len, vocab_size]. "
            f"With TP, the full vocabulary distribution is never materialized - each rank only "
            f"computes logprobs for tokens in its vocab shard. KD needs the complete distribution "
            f"to compute KL divergence.\n"
            f"\n"
            f"Solutions:\n"
            f"  1. Set teacher.tensor_parallel_size=1 (recommended)\n"
            f"  2. Use pipeline parallelism for the teacher instead (if model is large)\n"
            f"  3. Wait for full logprob gathering support (future work)\n"
            f"\n"
            f"Future work: Add Policy.get_full_logprobs() method that returns "
            f"[batch, seq, vocab_size] tensors by gathering across TP ranks, or implement "
            f"matched TP sharding approach. Estimated effort: 3-5 days."
        )
    
    # Build PolicyConfig for teacher
        # Use student's logprob batch size for consistency
        student_logprob_batch_size = self.master_config["student_policy"]["logprob_batch_size"]
        
        teacher_policy_config: PolicyConfig = {
            "model_name": teacher_config["model_name"],
            "tokenizer": {"name": teacher_config["model_name"]},
            "precision": teacher_config["precision"],
            "max_total_sequence_length": teacher_config["max_total_sequence_length"],
            
            # Teacher-specific parallelism
            "dtensor_v2_cfg": {
                "enabled": True,
                "tensor_parallel_size": teacher_config.get("tensor_parallel_size", 1),
                "pipeline_parallel_size": teacher_config.get("pipeline_parallel_size", 1),
                "context_parallel_size": 1,
                "expert_parallel_size": 1,
                "cpu_offload": False,
                "sequence_parallel": False,
                "activation_checkpointing": False,
                "custom_parallel_plan": None,
            },
            
            # Inference-only config (placeholders for unused training params)
            "train_global_batch_size": 1,
            "train_micro_batch_size": 1,
            "logprob_batch_size": student_logprob_batch_size,
            
            # These aren't used for teacher but Policy requires them
            "max_grad_norm": 1.0,
            "optimizer": {
                "name": "torch.optim.AdamW",
                "kwargs": {"lr": 1e-5},
            },
        }
        
        # Load teacher weights
        teacher_weights_path = teacher_config.get("checkpoint_path", None)
        
        teacher = Policy(
            cluster=cluster,
            config=teacher_policy_config,
            tokenizer=tokenizer,
            weights_path=teacher_weights_path,
            init_optimizer=False,  # Key: no optimizer for teacher
            init_reference_model=False,
            use_hf_checkpoint=False,
        )
        
        # Set teacher to eval mode (disable dropout, etc.)
        futures = teacher.worker_group.run_all_workers_single_data("eval")
        ray.get(futures)
        
        logging.info(f"  ✓ Teacher loaded from {teacher_config['model_name']}")
        
        return teacher
    
    def _validate_tokenizers(self) -> None:
    """Validate student and teacher use compatible tokenizers.
    
    Ensures:
    - Same vocab size
    - Same tokenization for test string
    """
    # Test tokenization consistency
    test_text = "Hello, world! This is a test."
    test_tokens = self.tokenizer(test_text, return_tensors="pt")
    
    # Get vocab size from tokenizer
    vocab_size = len(self.tokenizer)
    
    # Validate vocab size consistency
    # Note: Both teacher and student use the same tokenizer object in current implementation,
    # so this check is somewhat redundant. However, it's here for future-proofing in case
    # we support different tokenizers for teacher/student.
    if vocab_size <= 0:
        raise ValueError(
            f"Invalid vocabulary size: {vocab_size}. "
            f"Tokenizer may not be properly initialized."
        )
    
    logging.info(f"  ✓ Vocabulary size: {vocab_size}")
    
    # Validate sequence lengths match
        student_max_len = self.master_config["student_policy"]["max_total_sequence_length"]
        teacher_max_len = self.master_config["teacher"]["max_total_sequence_length"]
        
        if student_max_len != teacher_max_len:
            raise ValueError(
                f"Teacher and student must have the same max_total_sequence_length. "
                f"student={student_max_len}, teacher={teacher_max_len}"
            )

        logging.info(f"  ✓ Tokenizer validation passed (vocab_size={vocab_size})")
    
    def _process_batch(self, batch: BatchedDataDict) -> BatchedDataDict:
        """Process batch for training (reuse SFT pattern).
        
        Converts tokenized data into the format expected by Policy.train().
        """
        max_seq_len = self.master_config["student_policy"]["max_total_sequence_length"]
        max_batch_len = min(max([len(x) for x in batch["input_ids"]]), max_seq_len)
        batch_size = len(batch["input_ids"])
        
        train_data = {
            "input_ids": [None for _ in range(batch_size)],
            "input_lengths": [None for _ in range(batch_size)],
            "token_mask": [None for _ in range(batch_size)],
            "sample_mask": [None for _ in range(batch_size)],
        }
        
        truncated = 0
        
        for i, (input_ids, token_mask, sample_mask) in enumerate(zip(
            batch["input_ids"],
            batch["token_mask"],
            batch["sample_mask"]
        )):
            if len(input_ids) > max_batch_len:
                # Truncate sample if too long
                input_ids = input_ids[:max_batch_len]
                token_mask = token_mask[:max_batch_len]
                truncated += 1
            
            train_data["input_ids"][i] = _pad_tensor(
                torch.tensor(input_ids), 
                max_batch_len, 
                "right", 
                pad_value=self.tokenizer.pad_token_id
            )
            train_data["input_lengths"][i] = torch.tensor(len(input_ids))
            train_data["token_mask"][i] = _pad_tensor(
                torch.tensor(token_mask), 
                max_batch_len, 
                "right", 
                pad_value=0
            )
            train_data["sample_mask"][i] = torch.tensor(sample_mask)
        
        if truncated > 0:
            logging.warning(
                f"Truncated {truncated} samples from the batch due to exceeding "
                f"the maximum sequence length ({max_seq_len})"
            )
        
        return BatchedDataDict({k: torch.stack(v) for k, v in train_data.items()})
    
    async def _get_teacher_logprobs(
        self, data: BatchedDataDict
    ) -> torch.Tensor:
        """Get teacher log probabilities for the batch (teacher inference).
        
        Note: Policy.get_logprobs() returns log probabilities,
        not raw logits, because vLLM (the inference backend) doesn't expose logits.
        
        Returns:
            teacher_logprobs: Log probabilities from teacher [batch, seq_len, vocab]
        """
        # Prepare teacher for logprob inference
        self.teacher_policy.prepare_for_lp_inference()
        
        # Get teacher log probabilities
        teacher_output = self.teacher_policy.get_logprobs(data)
        
        # Extract log probabilities from output dict
        teacher_logprobs = teacher_output["logprobs"]
        
        # Optionally convert to fp16 to save memory (teacher logprobs can be large)
        # For 32k vocab and 4096 seq len, this saves ~250MB per batch
        if self.master_config["kd"].get("teacher_logprobs_fp16", False):
            teacher_logprobs = teacher_logprobs.half()
        
        return teacher_logprobs
    
    async def validate(
        self, step: int
    ) -> Optional[tuple[KDValidationMetrics, KDTimingMetrics]]:
        """Run validation on the validation dataset.
        
        Args:
            step: Current training step
            
        Returns:
            Optional tuple of (validation_metrics, timing_metrics)
        """
        if self.val_dataloader is None:
            logging.info("No validation dataloader provided, skipping validation")
            return None
        
        timer = Timer()
        kd_config = self.master_config["kd"]
        
        with timer.time("total_validation_time"):
            logging.info(f"▶ Starting validation at step {step}...")
            
            val_metrics = {"val_loss": 0.0, "val_base_loss": 0.0, "val_kd_loss": 0.0}
            num_valid_batches = 0
            
            self.student_policy.prepare_for_training()
            
            max_val_batches = kd_config.get("val_batches", -1)
            for batch_idx, raw_val_batch in enumerate(self.val_dataloader):
                # Stop if we've reached the max validation batches
                if max_val_batches > 0 and batch_idx >= max_val_batches:
                    break
                val_batch = BatchedDataDict(raw_val_batch)
                
                # Process batch
                val_data = self._process_batch(val_batch)
                
                # Get teacher log probabilities
                teacher_logprobs = await self._get_teacher_logprobs(val_data)
                val_data["teacher_logprobs"] = teacher_logprobs
                
                # Run validation (eval_mode=True, no gradient updates)
                val_results = await self.student_policy.train(
                    val_data,
                    self.loss_fn,
                    eval_mode=True,
                    gbs=kd_config["val_global_batch_size"],
                    mbs=kd_config["val_micro_batch_size"],
                )
                
                if len(val_results["all_mb_metrics"]) == 0:
                    warnings.warn(
                        "No validation metrics were collected for this batch. "
                        "This is likely because there were no valid samples."
                    )
                else:
                    val_metrics["val_loss"] += float(val_results["loss"])
                    # Extract component losses if available
                    if "all_mb_metrics" in val_results and len(val_results["all_mb_metrics"]) > 0:
                        first_mb = val_results["all_mb_metrics"][0]
                        val_metrics["val_base_loss"] += first_mb.get("base_loss", 0.0)
                        val_metrics["val_kd_loss"] += first_mb.get("kd_loss", 0.0)
                    num_valid_batches += 1
                
                # Limit validation batches if configured
                if (
                    kd_config["val_batches"] > 0
                    and batch_idx >= kd_config["val_batches"] - 1
                ):
                    break
            
            if num_valid_batches > 0:
                val_metrics["val_loss"] /= num_valid_batches
                val_metrics["val_base_loss"] /= num_valid_batches
                val_metrics["val_kd_loss"] /= num_valid_batches
            else:
                # Set metrics to NaN when no valid batches to avoid confusion
                val_metrics["val_loss"] = float('nan')
                val_metrics["val_base_loss"] = float('nan')
                val_metrics["val_kd_loss"] = float('nan')
                warnings.warn(
                    "No validation metrics were collected. "
                    "This is likely because there were no valid samples in the validation set."
                )
            
            self.student_policy.prepare_for_training()
        
        timing_metrics = timer.get_timing_metrics(reduction_op="sum")
        
        if num_valid_batches > 0:
            logging.info("\n📊 Validation Results:")
            logging.info(f"    • Validation loss: {val_metrics['val_loss']:.4f}")
            logging.info(f"    • Base loss: {val_metrics['val_base_loss']:.4f}")
            logging.info(f"    • KD loss: {val_metrics['val_kd_loss']:.4f}")
            
            logging.info("\n  ⏱️  Validation Timing:")
            validation_time = timing_metrics.get("total_validation_time", 0)
            logging.info(f"    • Total validation time: {validation_time:.2f}s")
        
        timer.reset()
        
        return val_metrics, timing_metrics
    
    async def train(self) -> None:
        """Main training loop for knowledge distillation."""
        timer = Timer()
        timeout = TimeoutChecker(
            timeout=self.master_config["checkpointing"]["checkpoint_must_save_by"],
            fit_last_save_time=True,
        )
        timeout.start_iterations()
        
        current_epoch = self.kd_save_state["epoch"]
        current_step = self.kd_save_state["step"]
        total_steps = self.kd_save_state["total_steps"]
        consumed_samples = self.kd_save_state["consumed_samples"]
        
        kd_config = self.master_config["kd"]
        max_num_epochs = kd_config["max_num_epochs"]
        max_num_steps = kd_config["max_num_steps"]
        val_period = kd_config.get("val_period", KD_DEFAULT_VAL_PERIOD)
        val_at_start = kd_config.get("val_at_start", KD_DEFAULT_VAL_AT_START)
        
        # Initial validation
        if val_at_start and total_steps == 0:
            logging.info("\n🔍 Running initial validation...")
            val_result = await self.validate(step=0)
            if val_result:
                val_metrics, val_timings = val_result
                self.logger.log_metrics(val_metrics, 0, prefix="validation")
                self.logger.log_metrics(val_timings, 0, prefix="timing/validation")
        
        # Prepare student for training
        self.student_policy.prepare_for_training()
        
        # Training loop
        while current_epoch < max_num_epochs and total_steps < max_num_steps:
            logging.info(f"\n{'='*25} Epoch {current_epoch + 1}/{max_num_epochs} {'='*25}")
            
            for raw_batch in self.train_dataloader:
                logging.info(f"\n{'='*25} Step {total_steps + 1} {'='*25}")
                
                batch = BatchedDataDict(raw_batch)
                maybe_gpu_profile_step(self.student_policy, total_steps + 1)
                
                with timer.time("total_step_time"):
                    # 1. Process batch
                    logging.debug("Processing batch...")
                    with timer.time("data_processing"):
                        train_data = self._process_batch(batch)
                    
                    # 2. Get teacher log probabilities (synchronous)
                    logging.debug("Computing teacher log probabilities...")
                    with timer.time("teacher_inference"):
                        teacher_logprobs = await self._get_teacher_logprobs(train_data)
                        train_data["teacher_logprobs"] = teacher_logprobs
                    
                    # 3. Train student
                    logging.debug("Training student policy...")
                    with timer.time("student_training"):
                        train_results = await self.student_policy.train(
                            train_data, 
                            self.loss_fn
                        )
                    
                    # 4. Validation
                    val_metrics, val_timings = None, None
                    if val_period > 0 and (total_steps + 1) % val_period == 0:
                        val_result = await self.validate(total_steps + 1)
                        if val_result:
                            val_metrics, val_timings = val_result
                            self.logger.log_metrics(val_metrics, total_steps + 1, prefix="validation")
                            self.logger.log_metrics(val_timings, total_steps + 1, prefix="timing/validation")
                    
                    # 5. Checkpointing
                    consumed_samples += self.master_config["student_policy"]["train_global_batch_size"]
                    is_last_step = (total_steps + 1 >= max_num_steps)
                    should_save_by_step = (
                        is_last_step or 
                        (total_steps + 1) % self.master_config["checkpointing"]["save_period"] == 0
                    )
                    should_save_by_timeout = timeout.check_save()
                    
                    if should_save_by_step or should_save_by_timeout:
                        self._save_checkpoint(
                            total_steps + 1,
                            current_epoch,
                            current_step + 1,
                            val_metrics,
                            consumed_samples,
                            timer,
                        )
                
                # 6. Logging
                self._log_training_step(total_steps + 1, train_results, timer)
                
                timer.reset()
                timeout.mark_iteration()
                current_step += 1
                total_steps += 1
                
                if total_steps >= max_num_steps:
                    break
            
            current_epoch += 1
            current_step = 0
        
        logging.info("\n✅ KD training completed!")
    
    def _save_checkpoint(
        self,
        total_steps: int,
        epoch: int,
        step: int,
        val_metrics: Optional[dict],
        consumed_samples: int,
        timer: Timer,
    ) -> None:
        """Save training checkpoint."""
        if not self.master_config["checkpointing"]["enabled"]:
            return
        
        self.student_policy.prepare_for_training()
        
        # Update save state
        self.kd_save_state["step"] = step
        self.kd_save_state["total_steps"] = total_steps
        self.kd_save_state["epoch"] = epoch
        if val_metrics is not None:
            self.kd_save_state["val_loss"] = val_metrics["val_loss"]
        elif "val_loss" in self.kd_save_state:
            del self.kd_save_state["val_loss"]
        self.kd_save_state["consumed_samples"] = consumed_samples
        
        # Check if metric-based checkpointing is configured
        if self.master_config["checkpointing"]["metric_name"] is not None:
            metric_name = self.master_config["checkpointing"]["metric_name"]
            if metric_name not in self.kd_save_state:
                warnings.warn(
                    f"You asked to save checkpoints based on {metric_name} but the metric "
                    "is not found in the save state. Saving most recent k checkpoints instead."
                )
                self.master_config["checkpointing"]["metric_name"] = None
        
        with timer.time("checkpointing"):
            logging.info(f"Saving checkpoint for step {total_steps}...")
            checkpoint_path = self.checkpointer.init_tmp_checkpoint(
                total_steps, self.kd_save_state, self.master_config
            )
            
            # Save student policy
            self.student_policy.save_checkpoint(
                weights_path=os.path.join(checkpoint_path, "student", "weights"),
                optimizer_path=os.path.join(checkpoint_path, "student", "optimizer"),
                tokenizer_path=os.path.join(checkpoint_path, "student", "tokenizer"),
            )
            
            # Save dataloader state
            torch.save(
                self.train_dataloader.state_dict(),
                os.path.join(checkpoint_path, "train_dataloader.pt"),
            )
            
            # Save teacher metadata (for reproducibility)
            teacher_metadata = {
                "model_name": self.master_config["teacher"]["model_name"],
                "checkpoint_path": self.master_config["teacher"].get("checkpoint_path"),
                "precision": self.master_config["teacher"]["precision"],
            }
            torch.save(
                teacher_metadata,
                os.path.join(checkpoint_path, "teacher_metadata.pt"),
            )
            
            self.checkpointer.finalize_checkpoint(checkpoint_path)
            logging.info(f"  ✓ Checkpoint saved to {checkpoint_path}")
    
    def _log_timing_metrics(self, timing_metrics: dict, step: int) -> None:
        """Log timing metrics to console and tracking systems."""
        total_time = timing_metrics.get("total_step_time", 0)
        
        logging.info("\n  ⏱️  Timing:")
        logging.info(f"  • Total step time: {total_time:.2f}s")
        
        for k, v in sorted(
            timing_metrics.items(), key=lambda item: item[1], reverse=True
        ):
            if k != "total_step_time":
                percent = (v / total_time * 100) if total_time > 0 else 0
                logging.info(f"  • {k}: {v:.2f}s ({percent:.1f}%)")
        
        # Log to tracking systems
        self.logger.log_metrics(timing_metrics, step, prefix="timing/train")
    
    def _log_training_step(
        self, step: int, train_results: dict, timer: Timer
    ) -> None:
        """Log training metrics."""
        # Extract metrics from training results
        metrics = {
            "loss": train_results["loss"].item() if torch.is_tensor(train_results["loss"]) else train_results["loss"],
            "grad_norm": train_results["grad_norm"].item() if torch.is_tensor(train_results["grad_norm"]) else train_results["grad_norm"],
        }
        
        # Add microbatch metrics
        if "all_mb_metrics" in train_results:
            mb_metrics = train_results["all_mb_metrics"]
            if len(mb_metrics) > 0:
                # Aggregate metrics across microbatches
                for key in ["base_loss", "kd_loss", "total_loss", "alpha", "temperature"]:
                    values = [mb.get(key, 0.0) for mb in mb_metrics if key in mb]
                    if values:
                        metrics[key] = np.mean(values).item()
        
        # Get timing metrics
        timing_metrics = timer.get_timing_metrics(reduction_op="sum")
        
        # Print to console
        logging.info("\n📊 Training Results:")
        logging.info(f"  • Total Loss: {metrics.get('loss', 0.0):.4f}")
        if "base_loss" in metrics:
            logging.info(f"  • Base Loss: {metrics['base_loss']:.4f}")
        if "kd_loss" in metrics:
            logging.info(f"  • KD Loss: {metrics['kd_loss']:.4f}")
        if "alpha" in metrics:
            logging.info(f"  • Alpha (distillation weight): {metrics['alpha']:.3f}")
        if "temperature" in metrics:
            logging.info(f"  • Temperature: {metrics['temperature']:.2f}")
        logging.info(f"  • Grad Norm: {metrics['grad_norm']:.4f}")
        
        # Log timing metrics using utility method
        self._log_timing_metrics(timing_metrics, step)
        
        # Log metrics to tracking systems
        self.logger.log_metrics(metrics, step, prefix="train")
