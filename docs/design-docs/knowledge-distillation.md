# Knowledge Distillation: Design & Implementation

This document explains the design and implementation of the knowledge distillation (KD) feature in NeMo RL.

## Current Limitations

**Tensor Parallelism Not Supported**: This implementation currently **does not support tensor parallelism** (TP > 1). Both `teacher.tensor_parallel_size` and `student_policy.dtensor_v2_cfg.tensor_parallel_size` must be set to 1.

**Impact**: Maximum model size is limited to what fits on a single GPU (~7B-14B models with fp16/bf16).

**Future Work**: See `kd-vocab-parallelism-future-work.md` for the planned implementation.

## Overview

Knowledge distillation allows a smaller **student** model to learn from a larger **teacher** model by minimizing the KL divergence between their output distributions. This implementation follows NeMo RL's existing patterns (similar to SFT/GRPO) and supports:

- **Online distillation**: Teacher generates log probabilities on-the-fly during training
- **Dedicated clusters**: Separate GPU allocation for teacher and student
- **Flexible architectures**: Teacher and student can be different models/sizes
- **Combined loss**: Mix supervised learning with distillation

## Architecture

### Components

```
rlkit/
├── algorithms/
│   ├── kd.py                    # KDTrainer: Main training orchestration
│   └── loss_functions.py        # KnowledgeDistillationLoss, CombinedKDLoss
├── config/
│   ├── kd.py                    # TypedDict configs (KDConfig, TeacherConfig, etc.)
│   └── __init__.py              # Config exports
├── models/
    └── policy/
        └── lm_policy.py         # Policy class (reused for teacher/student)
```

### Design Principles

1. **Reuse existing infrastructure**: Teacher and student both use `Policy` class
2. **Composition over inheritance**: `CombinedKDLoss` wraps base loss
3. **Fail loudly**: Clear error messages for misconfigurations
4. **Explicit is better than implicit**: Cluster allocation is explicit in config
5. **Follow existing patterns**: Mirrors SFT/GRPO structure

## Configuration Schema

### KDMasterConfig Structure

```python
class KDMasterConfig(TypedDict):
    student_policy: PolicyConfig      # Trainable student
    teacher: TeacherConfig            # Frozen teacher
    kd: KDConfig                      # Algorithm settings
    data: DataConfig
    logger: KDLoggerConfig
    cluster: ClusterConfig            # Total resources
    checkpointing: CheckpointingConfig
```

### TeacherConfig Structure

```python
class TeacherConfig(TypedDict):
    model_name: str                   # HF model or checkpoint path
    checkpoint_path: NotRequired[str] # Optional local checkpoint
    precision: str                    # "float16" or "bfloat16"
    max_total_sequence_length: int
    cluster: TeacherClusterConfig     # Dedicated GPU allocation
    tensor_parallel_size: NotRequired[int]
    pipeline_parallel_size: NotRequired[int]
```

**Key design decision**: Teacher gets its own cluster config, separate from student. This enables:
- Explicit resource control
- No memory contention
- Independent scaling

### KDConfig Structure

```python
class KDConfig(TypedDict):
    max_num_steps: int
    max_num_epochs: int
    seed: int
    
    alpha: float      # α ∈ [0, 1]
    temperature: float    # T ≥ 1.0
    
    val_period: int
    val_at_start: bool
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
```

## Loss Function Design

### KnowledgeDistillationLoss

**Purpose**: Compute KL divergence between teacher and student log probabilities

**Implementation**:

```python
class KnowledgeDistillationLoss(LossFunction):
    def __call__(self, student_logits, data, ...):
        teacher_logprobs = data["teacher_logprobs"]
        
        # Temperature scaling
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        # Convert teacher log probs to probs with temperature scaling
        teacher_logprobs_scaled = teacher_logprobs / T
        teacher_logprobs_scaled = teacher_logprobs_scaled - torch.logsumexp(
            teacher_logprobs_scaled, dim=-1, keepdim=True
        )
        teacher_probs = torch.exp(teacher_logprobs_scaled)
        
        # KL divergence
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='none')
        kl_div = kl_div.sum(dim=-1)  # Sum over vocab
        kl_div = kl_div * (T ** 2)   # Scale by T²
        
        # Apply masks
        masked_kl = masked_mean(kl_div, token_mask, sample_mask, ...)
        
        return masked_kl, {"kd_loss": masked_kl.item()}
```

**Key aspects**:
- **Float32 casting**: For numerical stability
- **T² scaling**: Standard in KD literature (preserves gradient magnitudes)
- **Token-level masking**: Only compute loss on valid tokens
- **Vocab parallelism**: Supports tensor parallel training

### CombinedKDLoss

**Purpose**: Mix supervised loss with KD loss

**Implementation**:

```python
class CombinedKDLoss(LossFunction):
    def __call__(self, student_logits, data, ...):
        # Base supervised loss (e.g., cross-entropy)
        base_loss, base_metrics = self.base_loss(student_logits, data, ...)
        
        # KD loss (fails loudly if teacher_logprobs missing)
        if "teacher_logprobs" not in data:
            raise ValueError("teacher_logprobs must be provided")
        
        kd_loss, kd_metrics = self.kd_loss(student_logits, data, ...)
        
        # Weighted combination
        total_loss = (1 - self.alpha) * base_loss + self.alpha * kd_loss
        
        return total_loss, {**base_metrics, **kd_metrics, "total_loss": ...}
```

**Key aspects**:
- **Composition pattern**: Wraps any base loss (NLLLoss, etc.)
- **Fail loudly**: Raises ValueError if teacher log probabilities missing
- **Comprehensive metrics**: Returns base_loss, kd_loss, total_loss
- **Inherits loss_type**: From base loss (token-level or sequence-level)

## KDTrainer Implementation

### Initialization Flow

```
1. Setup logger
2. Setup checkpointing (load previous state if resuming)
3. Setup dataloaders (train + validation)
4. Setup clusters:
   - Teacher cluster (explicit allocation)
   - Student cluster (remaining resources)
5. Initialize policies:
   - Student: Policy(init_optimizer=True)
   - Teacher: Policy(init_optimizer=False, eval mode)
6. Validate tokenizers match
7. Create CombinedKDLoss
```

### Cluster Allocation Strategy

**Pattern**: Follows GRPO's train/inference cluster split

```python
def _setup_clusters(self, cluster_config, teacher_config):
    # Teacher gets explicit allocation
    teacher_cluster = RayVirtualCluster(
        name="kd_teacher_cluster",
        bundle_ct_per_node_list=[teacher_gpus] * teacher_nodes,
        ...
    )
    
    # Student gets remaining
    student_nodes = total_nodes - teacher_nodes
    student_cluster = RayVirtualCluster(
        name="kd_student_cluster",
        bundle_ct_per_node_list=[student_gpus] * student_nodes,
        ...
    )
    
    return student_cluster, teacher_cluster
```

**Validation**:
- Ensures teacher_nodes ≤ total_nodes
- Ensures student gets at least 1 node
- Fails with clear error if allocation invalid

### Policy Initialization

**Student (Trainable)**:

```python
student = Policy(
    cluster=student_cluster,
    config=student_policy_config,
    tokenizer=tokenizer,
    weights_path=checkpoint_path,      # Resume from checkpoint
    optimizer_path=optimizer_path,
    init_optimizer=True,               # Create optimizer
    init_reference_model=False,        # No reference model for KD
)
```

**Teacher (Frozen)**:

```python
teacher = Policy(
    cluster=teacher_cluster,
    config=teacher_policy_config,
    tokenizer=tokenizer,
    weights_path=teacher_checkpoint,   # Load teacher weights
    init_optimizer=False,              # No optimizer (frozen)
    init_reference_model=False,
)

# Set to eval mode
teacher.worker_group.run_all_workers_single_data("eval")
```

**Key differences**:
- Teacher: `init_optimizer=False` (no gradients)
- Teacher: Set to eval mode (disables dropout)
- Teacher: Different checkpoint/model allowed

### Training Loop Flow

```python
async def train(self):
    for epoch in range(max_epochs):
        for batch in dataloader:
            # 1. Process batch (tokenize, pad, mask)
            train_data = self._process_batch(batch)
            
            # 2. Get teacher log probabilities (synchronous)
            teacher_logprobs = await self._get_teacher_logprobs(train_data)
            train_data["teacher_logprobs"] = teacher_logprobs

            # 3. Train student (forward + backward)
            train_results = await self.student_policy.train(
                train_data, 
                self.loss_fn  # CombinedKDLoss
            )
            
            # 4. Validation (periodic)
            if step % val_period == 0:
                await self.validate(step)
            
            # 5. Checkpointing (periodic)
            if step % save_period == 0:
                self._save_checkpoint(...)
            
            # 6. Logging
            self._log_training_step(...)
```

**Synchronous design**:
- Teacher inference completes before student training
- Simple, correct, no race conditions
- Future: Can add pipelining (teacher for batch N+1 while training batch N)

### Teacher Log Probability Computation

```python
async def _get_teacher_logprobs(self, data: BatchedDataDict) -> torch.Tensor:
    # Prepare teacher for inference
    self.teacher_policy.prepare_for_lp_inference()
    
    # Get log probabilities from teacher policy
    teacher_output = self.teacher_policy.get_logprobs(data)
    teacher_logprobs = teacher_output["logprobs"]
    
    return teacher_logprobs
```

**Note**: The `get_logprobs()` interface correctly returns log probabilities, not raw logits.

## Checkpointing

### What Gets Saved

```
checkpoint_dir/step_1000/
├── student/
│   ├── weights/           # Student model weights
│   ├── optimizer/         # Student optimizer state
│   └── tokenizer/         # Tokenizer config
├── train_dataloader.pt    # Dataloader state (for exact resuming)
├── teacher_metadata.pt    # Teacher info (for reproducibility)
└── training_info.json     # KDSaveState (step, epoch, metrics)
```

### Teacher Metadata

For reproducibility, we save teacher information:

```python
teacher_metadata = {
    "model_name": "Qwen/Qwen2.5-7B",
    "checkpoint_path": "/path/to/teacher/checkpoint",
    "precision": "float16",
}
```

This allows you to:
- Verify which teacher was used
- Reproduce distillation experiments
- Debug if teacher changed mid-training

### Resuming Training

When resuming:
1. Load student weights + optimizer
2. Load dataloader state (exact data position)
3. Load KDSaveState (epoch, step, metrics)
4. Re-initialize teacher (always from config, not checkpoint)

**Note**: Teacher is always loaded fresh from config. We don't checkpoint teacher since it's frozen.

## Data Pipeline

### Data Format

Reuses SFT's data format:

```python
{
    "input_ids": List[int],        # Tokenized input
    "token_mask": List[float],     # Mask for loss computation
    "sample_mask": float,          # Sample-level mask (0 or 1)
}
```

Datasets must be in HuggingFace chat format. See [chat datasets documentation](chat-datasets.md).

### Processing Pipeline

```python
def _process_batch(self, batch):
    # 1. Pad/truncate to max_batch_len
    # 2. Create token masks (which tokens to compute loss on)
    # 3. Create sample masks (which samples are valid)
    # 4. Convert to tensors
    # 5. Return BatchedDataDict
```

Follows SFT's `_process_batch` implementation exactly.

## Loss Computation Flow

### In Policy Worker

The policy worker computes logits from the model and calls the loss function:

```python
# In v2_policy_worker.py (existing code, no changes needed)
def train_step(data, loss_fn):
    logits = model(data["input_ids"])  # Forward pass
    loss, metrics = loss_fn(
        logits,                        # Student logits
        data,                          # Includes teacher_logprobs
        global_valid_seqs,
        global_valid_toks,
    )
    loss.backward()  # Backward pass
```

### In Loss Function

```python
def CombinedKDLoss.__call__(student_logits, data, ...):
    # 1. Compute supervised loss
    base_loss = NLLLoss(student_logits, data["labels"], ...)
    
    # 2. Compute KD loss
    teacher_logprobs = data["teacher_logprobs"]  # From KDTrainer
    kd_loss = KL_divergence(student_logits, teacher_logprobs, T)

    # 3. Combine
    total_loss = (1 - α) * base_loss + α * kd_loss
    
    return total_loss
```

### Gradient Flow

Only student receives gradients:

```
Teacher: Forward only (no backward)
         ↓ (logits)
         │
         ↓
      KL loss  <---- Student: Forward + Backward
         ↓
      Gradients → Student only
```

Teacher is frozen, so `teacher_logprobs` are detached from computation graph.

## Distributed Training

### Cluster Architecture

```
        Total Cluster (5 nodes × 8 GPUs = 40 GPUs)
                        │
        ┌───────────────┼──────────────────┐
        │               │                  │
   Teacher Cluster  Student Cluster
   (1 node, 8 GPUs) (4 nodes, 32 GPUs)
        │               │
   Inference only   Training (fwd+bwd)
```

### Ray Virtual Clusters

We use Ray's resource management to allocate GPUs:

```python
teacher_cluster = RayVirtualCluster(
    name="kd_teacher_cluster",
    bundle_ct_per_node_list=[8] * 1,  # 1 node × 8 GPUs
    use_gpus=True,
    ...
)

student_cluster = RayVirtualCluster(
    name="kd_student_cluster",
    bundle_ct_per_node_list=[8] * 4,  # 4 nodes × 8 GPUs
    use_gpus=True,
    ...
)
```

Each cluster spawns worker groups that run on their dedicated GPUs.

### Parallelism Support

**Teacher parallelism**:
- **Tensor Parallel (TP): NOT SUPPORTED** (must be 1)
- **Pipeline Parallel (PP): NOT SUPPORTED** (must be 1)
- **Expert Parallel (EP): SUPPORTED** (for MoE models like Mixtral, Qwen-MoE)
- Future work: See `kd-vocab-parallelism-future-work.md`

**Student parallelism**:
- **Tensor Parallel (TP): NOT SUPPORTED** (must be 1)
- Pipeline Parallel (PP), Context Parallel (CP), Expert Parallel (EP): Supported

**Example**:
```yaml
teacher:
  tensor_parallel_size: 1  # TP > 1 not currently supported

student_policy:
  dtensor_v2_cfg:
    tensor_parallel_size: 1  # TP > 1 not currently supported
    pipeline_parallel_size: 1
```

## Memory Management

### Teacher Memory Optimization

1. **Lower precision**: Use `float16` instead of `bfloat16`
   ```yaml
   teacher:
     precision: "float16"  # Saves ~50% memory
   ```

2. **No optimizer**: Teacher has no optimizer state (saves ~2x model size)

3. **No activation checkpointing**: Inference doesn't need gradient checkpointing

4. **Eval mode**: Disables dropout (deterministic, slightly faster)

### Student Memory Optimization

1. **Activation checkpointing**: For long sequences
   ```yaml
   student_policy:
     dtensor_v2_cfg:
       activation_checkpointing: true
   ```

2. **Sequence packing**: Pack multiple sequences per batch
   ```yaml
   student_policy:
     sequence_packing:
       enabled: true
   ```

3. **Gradient accumulation**: Via global batch size > micro batch size

## Validation

### Validation Process

```python
def validate(self, step):
    for val_batch in val_dataloader:
        # 1. Process batch
        val_data = self._process_batch(val_batch)
        
        # 2. Get teacher log probabilities
        teacher_logprobs = await self._get_teacher_logprobs(val_data)
        val_data["teacher_logprobs"] = teacher_logprobs

        # 3. Run student in eval mode (no gradient updates)
        val_results = await self.student_policy.train(
            val_data,
            self.loss_fn,
            eval_mode=True,
        )
        
        # 4. Aggregate metrics
```

**Note**: Validation also requires teacher inference (for KD loss computation).

### Validation Metrics

- `val_loss`: Total loss on validation set
- `val_base_loss`: Supervised component
- `val_kd_loss`: Distillation component

Use these to tune hyperparameters:
- If `val_kd_loss` is high, teacher and student disagree (increase T or reduce α)
- If `val_base_loss` is high, model not learning task (reduce α)

## Logging

### Logged Metrics

**Per-step training metrics**:
```python
{
    "train/loss": 2.1234,           # Total loss
    "train/base_loss": 2.3456,      # Supervised loss
    "train/kd_loss": 1.8901,        # KD loss
    "train/alpha": 0.7,         # α coefficient
    "train/temperature": 3.0,       # T value
    "train/grad_norm": 0.8765,      # Gradient norm
}
```

**Timing metrics**:
```python
{
    "timing/train/total_step_time": 12.34,
    "timing/train/teacher_inference": 3.21,
    "timing/train/student_training": 8.45,
    "timing/train/data_processing": 0.68,
}
```

**Validation metrics** (periodic):
```python
{
    "validation/val_loss": 1.9876,
    "validation/val_base_loss": 2.1234,
    "validation/val_kd_loss": 1.7890,
}
```

### Console Output

Example output during training:

```
========================= Step 1 =========================
Processing batch...
Computing teacher log probabilities...
Training student policy...

📊 Training Results:
  • Total Loss: 2.1234
  • Base Loss: 2.3456
  • KD Loss: 1.8901
  • KD Weight: 0.700
  • Temperature: 3.00
  • Grad Norm: 0.8765

  ⏱️  Timing:
  • Total step time: 12.34s
  • teacher_inference: 3.21s (26.0%)
  • student_training: 8.45s (68.5%)
  • data_processing: 0.68s (5.5%)
```

## Future Enhancements

### Planned Features

1. **Offline distillation**: Pre-compute teacher log probabilities, cache to disk
   - Faster training (no teacher inference overhead)
   - Enables using very large teachers without GPU cost

2. **KD weight scheduling**: Gradually shift from teacher to labels
   ```python
   alpha_schedule:
     start: 0.9  # Emphasize teacher early
     end: 0.5    # Balance later
     steps: 10000
   ```

3. **Alternative losses**: Beyond KL divergence
   - Jensen-Shannon divergence
   - Reverse KL
   - Total variation distance

4. **Asynchronous pipelining**: Overlap teacher inference with student training
   - Teacher computes batch N+1 while student trains on batch N
   - Requires careful synchronization

5. **Multi-teacher distillation**: Ensemble of teachers
   ```python
   kd_loss = KL(student, 0.5 * teacher1 + 0.5 * teacher2)
   ```

### Extension Points

The current design is extensible:

**Custom loss functions**:
```python
# Create your own distillation loss
class CustomKDLoss(LossFunction):
    def __call__(self, student_logits, data, ...):
        # Your custom logic here
        pass

# Use in KDTrainer
self.loss_fn = CombinedKDLoss(
    base_loss=NLLLoss(),
    kd_loss=CustomKDLoss(),  # Instead of KnowledgeDistillationLoss
    alpha=0.5,
)
```

**Custom teacher policies**:
```python
# Wrap teacher with preprocessing
class CustomTeacherPolicy:
    def __init__(self, base_teacher):
        self.teacher = base_teacher
    
    def get_logprobs(self, data):
        # Custom preprocessing
        modified_data = self.preprocess(data)
        return self.teacher.get_logprobs(modified_data)
```

## Implementation Notes

### Why Not Extend SFT/GRPO?

We chose a **standalone algorithm** (`kd.py`) instead of adding KD as a feature flag to SFT/GRPO because:

✅ **Cleaner separation of concerns**: KD logic isolated
✅ **Easier testing**: Can test KD in isolation
✅ **Follows existing pattern**: SFT, GRPO, RM are all separate
✅ **Simpler to maintain**: No feature flags complicating SFT/GRPO

Future: Could create `kd_grpo.py` for RL + distillation if needed.

### Why Fail Loudly on Missing Teacher Log Probabilities?

Original design had graceful fallback (use base loss only if teacher log probabilities missing). We changed this to **fail loudly** because:

✅ **Catches bugs immediately**: Missing teacher log probabilities indicates pipeline error
✅ **Clear intent**: If using KD, teacher must be present
✅ **No silent degradation**: Training doesn't silently become pure SFT

If you want pure supervised learning, use `run_sft.py` instead.

### Why Synchronous Teacher Inference?

Current implementation is synchronous (teacher completes before student trains). We chose this over pipelining because:

✅ **Simpler implementation**: No async coordination needed
✅ **Easier debugging**: Clear sequence of operations
✅ **Correct baseline**: Get it working first, optimize later

Future enhancement: Add pipelining if teacher becomes bottleneck (>40% of step time).

## Testing

### Unit Tests

Test coverage for KD components:

```python
# test_kd_loss.py
def test_kl_divergence_computation():
    """Verify KL divergence is computed correctly."""
    
def test_temperature_scaling():
    """Verify temperature softens distributions."""
    
def test_combined_loss_weighting():
    """Verify loss mixing is correct."""

# test_kd_trainer.py  
def test_cluster_allocation():
    """Verify student/teacher cluster split."""
    
def test_teacher_initialization():
    """Verify teacher is frozen (no optimizer)."""
    
def test_tokenizer_validation():
    """Verify tokenizer mismatch detection."""
```

### Integration Tests

```python
def test_end_to_end_training():
    """Train tiny model for 10 steps, verify loss decreases."""
    
def test_checkpoint_resume():
    """Train, checkpoint, resume, verify state matches."""
    
def test_validation_loop():
    """Run validation, verify metrics computed correctly."""
```

## Comparison with Other Frameworks

### vs. Hugging Face Transformers

HuggingFace doesn't have built-in KD support. Users typically:
- Manually load teacher/student
- Manually compute KL loss
- No cluster management

**NeMo RL advantages**:
- Integrated cluster allocation
- Distributed training out-of-the-box
- Checkpointing with teacher metadata
- Comprehensive logging

### vs. FastChat/vLLM

These are inference frameworks, not training frameworks. They don't support KD.

### vs. DeepSpeed

DeepSpeed has KD utilities but:
- No dedicated teacher/student cluster allocation
- Manual configuration required
- Limited documentation

**NeMo RL advantages**:
- Declarative configuration (YAML)
- Automatic resource management
- Built-in validation and logging

## Related Documentation

- [User Guide](../guides/kd.md) - How to use KD
- [SFT Design](training-backends.md) - SFT implementation details
- [GRPO Design](loss-functions.md) - GRPO loss functions
- [Cluster Setup](../cluster.md) - Multi-node configuration

## References

- [Hinton et al. (2015): Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Gou et al. (2021): Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)
