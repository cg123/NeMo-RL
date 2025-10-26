from typing import TypedDict, NotRequired

from rlkit.config.rl.policy import PolicyConfig
from rlkit.config.data import DataConfig
from rlkit.config.logging import LoggerConfig
from rlkit.config.cluster import ClusterConfig
from rlkit.config.checkpointing import CheckpointingConfig


# Default values for optional KD parameters
KD_DEFAULT_ALPHA = 0.5
KD_DEFAULT_TEMPERATURE = 2.0
KD_DEFAULT_TEACHER_LOGPROBS_FP16 = False
KD_DEFAULT_VAL_PERIOD = 0
KD_DEFAULT_VAL_AT_START = False


class KDLoggerConfig(LoggerConfig):
    """Logger configuration for knowledge distillation."""
    num_val_samples_to_print: int


class TeacherClusterConfig(TypedDict):
    """Cluster allocation for teacher model inference."""
    num_nodes: int
    gpus_per_node: int


class TeacherConfig(TypedDict):
    """Configuration for teacher model."""
    model_name: str  # HuggingFace model name or path to checkpoint
    checkpoint_path: NotRequired[str]  # Optional: local checkpoint path
    precision: str  # e.g., "float16", "bfloat16"
    max_total_sequence_length: int
    
    # Cluster allocation for teacher inference
    cluster: TeacherClusterConfig
    
    # Parallelism for teacher (if large model)
    tensor_parallel_size: NotRequired[int]
    pipeline_parallel_size: NotRequired[int]
    expert_parallel_size: NotRequired[int]  # For MoE models


class KDConfig(TypedDict):
    """Configuration for knowledge distillation algorithm.
    
    Defaults:
        alpha: 0.5
        temperature: 2.0
        teacher_logprobs_fp16: False
        val_period: 0 (disabled)
        val_at_start: False
    """
    # Training limits
    max_num_steps: int
    max_num_epochs: int
    seed: int
    
    # Distillation hyperparameters
    # alpha coefficient: total_loss = (1-alpha)*base_loss + alpha*kd_loss
    # Range: [0.0, 1.0] where 0.0 = pure supervised, 1.0 = pure distillation
    alpha: NotRequired[float]  # Default: 0.5
    
    # Temperature for softening logits (Hinton et al. 2015)
    # Range: >= 1.0 where 1.0 = no softening, 2.0-4.0 = typical KD range
    temperature: NotRequired[float]  # Default: 2.0
    
    # Store teacher logprobs in fp16 to save memory (~50% reduction)
    # Recommended for large vocabularies (>32k) or long sequences (>2048)
    teacher_logprobs_fp16: NotRequired[bool]  # Default: False
    
    # Validation
    val_period: NotRequired[int]  # Default: 0 (disabled)
    val_at_start: NotRequired[bool]  # Default: False
    val_batches: int  # Number of batches to use for validation (-1 = all)
    val_global_batch_size: int
    val_micro_batch_size: int


class KDMasterConfig(TypedDict):
    """Master configuration for KD training."""
    student_policy: PolicyConfig  # Trainable student model
    teacher: TeacherConfig  # Frozen teacher model
    kd: KDConfig
    data: DataConfig
    logger: KDLoggerConfig
    cluster: ClusterConfig  # Total cluster resources
    checkpointing: CheckpointingConfig
