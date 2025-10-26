from typing import TypedDict, NotRequired

from rlkit.config.rl.policy import PolicyConfig
from rlkit.config.data import DataConfig
from rlkit.config.logging import LoggerConfig
from rlkit.config.cluster import ClusterConfig
from rlkit.config.checkpointing import CheckpointingConfig


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


class KDConfig(TypedDict):
    """Configuration for knowledge distillation algorithm."""
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
