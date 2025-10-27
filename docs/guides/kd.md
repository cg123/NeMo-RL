# Knowledge Distillation in NeMo RL

This guide explains how to perform knowledge distillation (KD) in NeMo RL, where a smaller student model learns to mimic a larger teacher model. Knowledge distillation can produce compact models that retain most of the teacher's capabilities while being significantly faster and more efficient.

## Table of Contents

- [Quick Start](#quick-start)
- [What is Knowledge Distillation?](#what-is-knowledge-distillation)
- [When to Use KD](#when-to-use-kd)
- [Configuration](#configuration)
- [Understanding KD Hyperparameters](#understanding-kd-hyperparameters)
- [Cluster Allocation](#cluster-allocation)
- [Running KD Training](#running-kd-training)
- [Monitoring Training](#monitoring-training)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Quick Start

Launch a KD training job using the provided example configurations:

```bash
# Example: 4B teacher → 1.5B student (2 nodes, 8 GPUs total)
uv run examples/run_kd.py --config examples/configs/kd/qwen3_4B_to_1B.yaml
```

**Prerequisites**: Set your `HF_HOME`, `WANDB_API_KEY`, and run `huggingface-cli login` if using gated models.

## ⚠️ Current Limitations

**Tensor Parallelism Not Supported**: This implementation currently **does not support tensor parallelism** (TP > 1). Both teacher and student models must run with `tensor_parallel_size: 1`. This limits the maximum model size to what fits on a single GPU (~7B-70B depending on GPU memory and precision).

**Pipeline Parallelism Not Tested**: While `pipeline_parallel_size > 1` is not explicitly blocked, it has not been tested and may not work correctly with the teacher inference path. Use at your own risk.

**Expert Parallelism Supported**: For MoE (Mixture of Experts) teacher models, you can use `expert_parallel_size > 1` to distribute experts across GPUs. This is useful for models like Mixtral, Qwen-MoE, or DeepSeek-MoE.

**Workaround**: For larger models, use lower precision (`float16` or `bfloat16`) and techniques like:
- Flash Attention 2
- Gradient checkpointing
- CPU offloading (if supported)

**Future Work**: Vocab parallelism support is planned. See `docs/design-docs/kd-vocab-parallelism-future-work.md` for the design.

## What is Knowledge Distillation?

Knowledge distillation is a model compression technique where:

1. **Teacher Model**: A large, well-trained model that provides "soft labels" (probability distributions)
2. **Student Model**: A smaller model that learns from both:
   - **Ground truth labels** (hard labels from the dataset)
   - **Teacher predictions** (soft labels with rich information)

### How It Works

The student minimizes a combined loss:

```
L_total = (1 - α) * L_supervised + α * L_KD
```

Where:
- **L_supervised**: Standard cross-entropy with ground truth labels
- **L_KD**: KL divergence between student and teacher log probabilities (softened by temperature)
- **α** (alpha): Balances the two losses (0 = pure supervised, 1 = pure distillation)

### Temperature Scaling

Log probabilities are softened using temperature **T** before computing KL divergence:

```
# Teacher already provides log probabilities
p_teacher_scaled = logprobs_teacher / T  # Temperature scaling in log space
p_teacher = exp(p_teacher_scaled - logsumexp(p_teacher_scaled))  # Renormalize

# Student provides logits, apply softmax with temperature
p_student = log_softmax(logits_student / T)
L_KD = KL(p_teacher || p_student) * T²
```

Higher temperatures produce softer probability distributions, transferring more "dark knowledge" about class relationships.

## When to Use KD

### Good Use Cases

✅ **Model compression**: Deploy smaller models with similar performance
- Example: 7B teacher → 1.5B student for inference efficiency

✅ **Transfer learning**: Distill knowledge from one domain to another
- Example: General-purpose teacher → domain-specific student

✅ **Ensemble distillation**: Combine multiple teachers into one student
- Example: Multiple specialized models → single general model

✅ **Noisy label datasets**: Teacher provides cleaner signal than labels
- Example: Web-scraped data with annotation errors

### When NOT to Use KD

❌ **Teacher underperforms**: If teacher isn't better than student could be
- Use standard SFT instead

❌ **Very similar sizes**: Marginal gains for small compression ratios
- Example: 7B → 6B (minimal benefit)

❌ **Limited compute**: KD requires running both teacher and student
- Needs ~2x GPU resources compared to SFT

## Configuration

A KD configuration has four main sections:

### 1. Student Policy Configuration

```yaml
student_policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  tokenizer:
    name: ${student_policy.model_name}
  
  train_global_batch_size: 256
  train_micro_batch_size: 4
  logprob_batch_size: 4
  max_total_sequence_length: 4096
  precision: "bfloat16"
  
  # Standard policy config (TP, optimizer, scheduler, etc.)
  dtensor_v2_cfg:
    enabled: true
    tensor_parallel_size: 1  # TP > 1 not currently supported for KD
    context_parallel_size: 1
    pipeline_parallel_size: 1
    expert_parallel_size: 1
  
  optimizer:
    name: "torch.optim.AdamW"
    # ... optimizer settings
```

**Key points**:
- Uses same config structure as SFT/GRPO (familiar to NeMo RL users)
- Must use `student_policy` instead of `policy`
- Trainable model with optimizer

### 2. Teacher Configuration

```yaml
teacher:
  model_name: "Qwen/Qwen2.5-7B"
  checkpoint_path: null  # Optional: load from local checkpoint
  precision: "float16"   # Lower precision for memory efficiency
  max_total_sequence_length: ${student_policy.max_total_sequence_length}
  
  # Dedicated cluster for teacher inference
  cluster:
    num_nodes: 1
    gpus_per_node: 4
  
  # Parallelism for teacher (inference only)
  tensor_parallel_size: 1  # TP > 1 not currently supported for KD
  pipeline_parallel_size: 1
```

**Key points**:
- Teacher is **frozen** (no optimizer, no gradients)
- Can be different architecture/size than student
- **Must use same tokenizer** as student
- Uses dedicated GPUs (no memory contention)
- Typically use `float16` for memory efficiency

### 3. KD Algorithm Configuration

```yaml
kd:
  max_num_steps: 50000
  max_num_epochs: 3
  seed: 42
  
  # Distillation hyperparameters
  alpha: 0.7        # α in loss formula
  temperature: 3.0      # T for softening logits
  
  # Validation
  val_period: 500       # Validate every N steps
  val_at_start: true
  val_batches: 20
  val_global_batch_size: 128
  val_micro_batch_size: 4
```

**Key parameters**:
- `alpha`: How much to emphasize teacher (see [Understanding KD Hyperparameters](#understanding-kd-hyperparameters))
- `temperature`: How soft to make distributions

### 4. Cluster Allocation

```yaml
cluster:
  num_nodes: 5          # Total available nodes
  gpus_per_node: 8      # GPUs per node

# Teacher gets explicit allocation
teacher:
  cluster:
    num_nodes: 1        # Teacher uses 1 node × 8 GPUs = 8 GPUs
    gpus_per_node: 8

# Student automatically gets: (5 - 1) nodes × 8 GPUs = 32 GPUs
```

See [Cluster Allocation](#cluster-allocation) for details.

## Understanding KD Hyperparameters

### KD Weight (α)

**Definition**: Balances supervised loss vs. distillation loss

```
L_total = (1 - α) * L_supervised + α * L_KD
```

**Choosing `alpha`**:

| Scenario | Recommended α | Reasoning |
|----------|--------------|----------|
| **High-quality teacher** (e.g., GPT-4 → GPT-3.5) | 0.7 - 0.9 | Trust teacher more than labels |
| **Similar-sized models** (e.g., 7B → 3B) | 0.5 | Balanced mix |
| **Noisy labels** | 0.7 - 0.8 | Teacher provides cleaner signal |
| **High-quality labels** | 0.3 - 0.5 | Preserve ground truth |
| **Uncertain teacher quality** | 0.5 | Start balanced, adjust based on validation |

**Experimentation tips**:
- Start with `α = 0.5` (equal weight)
- If validation loss improves with higher α, teacher is helpful
- If validation loss degrades, reduce α or check teacher quality

### Temperature (T)

**Definition**: Softens probability distributions before computing KL divergence

```
p_soft = softmax(logits / T)
```

**Choosing `temperature`**:

| Model Gap | Recommended T | Reasoning |
|-----------|--------------|----------|
| **Large gap** (7B → 1B) | 3.0 - 5.0 | More "dark knowledge" needed |
| **Medium gap** (4B → 1.5B) | 2.0 - 3.0 | Standard KD range |
| **Small gap** (3B → 2B) | 1.5 - 2.0 | Minimal softening |
| **Same architecture** | 2.0 | Hinton et al. (2015) default |

**Effects of temperature**:
- **T = 1.0**: No softening (standard softmax)
- **T = 2.0 - 4.0**: Typical KD range, reveals class relationships
- **T > 5.0**: Very soft, may lose signal

**Example**: 
```
Logits: [5.0, 2.0, 1.0]

T=1.0 (no softening):
  Probs: [0.84, 0.12, 0.04]  # Teacher is very confident
  
T=3.0 (softened):
  Probs: [0.52, 0.29, 0.19]  # Reveals that option 2 is closer to option 1
```

The softened distribution transfers knowledge about **relative rankings**, not just the top class.

## Cluster Allocation

NeMo RL KD uses **separate clusters** for student training and teacher inference, following the GRPO pattern.

### How Allocation Works

```yaml
cluster:
  num_nodes: 5
  gpus_per_node: 8

teacher:
  cluster:
    num_nodes: 1
    gpus_per_node: 8

# Student automatically gets:
# (total_nodes - teacher_nodes) × gpus_per_node
# = (5 - 1) × 8 = 32 GPUs
```

### Why Separate Clusters?

✅ **No memory contention**: Teacher and student don't compete for GPU memory
✅ **Clear resource allocation**: Explicit control over resource split
✅ **Easier debugging**: Isolate performance bottlenecks
✅ **Flexible scaling**: Adjust teacher/student GPUs independently

### Allocation Examples

**Example 1: Single-node setup**

```yaml
cluster:
  num_nodes: 1
  gpus_per_node: 8

teacher:
  cluster:
    num_nodes: 1   # Use all GPUs for teacher
    gpus_per_node: 4  # Only use 4 GPUs

# Student gets remaining 4 GPUs on same node
```

**Example 2: Multi-node setup**

```yaml
cluster:
  num_nodes: 10
  gpus_per_node: 8

teacher:
  cluster:
    num_nodes: 2   # Dedicate 2 full nodes to teacher
    gpus_per_node: 8

# Student gets 8 nodes (64 GPUs)
```

### Sizing Guidelines

**Teacher allocation**:
- **Note**: Currently TP=1 only, so teacher must fit on single GPU
- 7B model → 1 GPU (with fp16/bf16)
- 14B model → 1 GPU (A100 80GB with optimizations)
- Larger models not currently supported (requires TP > 1)
- Add more GPUs if teacher inference is slow (<30% of step time)

**Student allocation**:
- Use remaining GPUs for student training
- Should be majority of cluster (70-80%)
- Adjust if student training is bottleneck

### Validation Errors

The trainer validates cluster allocation:

```
ValueError: Teacher requires 3 nodes but only 2 total nodes available
```
→ Reduce `teacher.cluster.num_nodes` or increase `cluster.num_nodes`

```
ValueError: No nodes remaining for student training after teacher allocation
```
→ Teacher is using all available nodes, increase `cluster.num_nodes`

## Running KD Training

### Basic Usage

```bash
uv run examples/run_kd.py --config examples/configs/kd/qwen3_4B_to_1B.yaml
```

### With Configuration Overrides

```bash
# Adjust KD hyperparameters
uv run examples/run_kd.py \\
  --config examples/configs/kd/qwen3_4B_to_1B.yaml \\
  kd.alpha=0.8 \\
  kd.temperature=3.5

# Use custom dataset
uv run examples/run_kd.py \\
  --config examples/configs/kd/qwen3_4B_to_1B.yaml \\
  data.dataset_name="my/custom-dataset" \\
  data.dataset_type="chat"

# Adjust cluster allocation
uv run examples/run_kd.py \\
  --config examples/configs/kd/qwen3_4B_to_1B.yaml \\
  cluster.num_nodes=10 \\
  teacher.cluster.num_nodes=1

# Change learning rate
uv run examples/run_kd.py \\
  --config examples/configs/kd/qwen3_4B_to_1B.yaml \\
  student_policy.optimizer.kwargs.lr=3.0e-6
```

### Multi-node with Slurm

For distributed training across multiple nodes, refer to the [cluster documentation](../cluster.md).

Example Slurm script:

```bash
#!/bin/bash
#SBATCH --nodes=5
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1

uv run examples/run_kd.py \\
  --config examples/configs/kd/qwen3_7B_to_1.5B.yaml \\
  cluster.num_nodes=4 \\
  cluster.gpus_per_node=8 \\
  teacher.cluster.num_nodes=1
```

## Monitoring Training

### Key Metrics

NeMo RL KD logs comprehensive metrics to W&B, TensorBoard, or MLflow:

#### Training Metrics

- **`train/total_loss`**: Combined supervised + KD loss (what's being minimized)
- **`train/base_loss`**: Supervised component (cross-entropy with labels)
- **`train/kd_loss`**: Distillation component (KL divergence with teacher)
- **`train/alpha`**: Current α value (for verification)
- **`train/temperature`**: Current T value (for verification)
- **`train/grad_norm`**: Gradient norm (for stability monitoring)

#### Validation Metrics

- **`validation/val_loss`**: Total validation loss
- **`validation/val_base_loss`**: Supervised validation loss
- **`validation/val_kd_loss`**: KD validation loss

#### Timing Metrics

- **`timing/train/total_step_time`**: Total time per training step
- **`timing/train/teacher_inference`**: Time spent on teacher inference
- **`timing/train/student_training`**: Time spent on student forward+backward
- **`timing/train/data_processing`**: Time spent processing batches

### Healthy Training

Your training is healthy if:

✅ **`kd_loss` < `base_loss`**: Teacher provides better signal than labels
✅ **`val_loss` decreases**: Model is learning
✅ **`teacher_inference` < 30%** of `total_step_time`: Teacher not a bottleneck
✅ **`grad_norm` stable**: No gradient explosions

### Troubleshooting Signals

🔴 **`kd_loss` not improving**:
- Temperature too high/low → Adjust `kd.temperature`
- Teacher not helpful → Reduce `kd.alpha`
- Tokenizer mismatch → Check initialization logs

🔴 **`val_loss` increasing**:
- Overfitting → Add regularization, reduce learning rate
- Learning rate too high → Reduce `student_policy.optimizer.kwargs.lr`

🔴 **`teacher_inference` > 50%** of step time:
- Teacher is bottleneck → Increase `teacher.cluster.num_nodes`
- Consider using lower precision (fp16 vs bf16)
- Ensure teacher model fits on single GPU (TP > 1 not supported)

### Console Output

During training, you'll see detailed logs:

```
========================= Step 1 =========================
Processing batch...
Computing teacher log probabilities...
Training student policy...

📊 Training Results:
  • Total Loss: 2.1234
  • Base Loss: 2.3456
  • KD Loss: 1.8901
  • KD Weight: 0.500
  • Temperature: 2.00
  • Grad Norm: 0.8765

  ⏱️  Timing:
  • Total step time: 12.34s
  • teacher_inference: 3.21s (26.0%)
  • student_training: 8.45s (68.5%)
  • data_processing: 0.68s (5.5%)
```

## Best Practices

### 1. Start with Balanced Settings

```yaml
kd:
  alpha: 0.5      # Equal mix
  temperature: 2.0    # Standard
```

Then adjust based on validation metrics.

### 2. Validate Teacher Quality First

Before distilling, verify teacher performs well on your task:
```bash
# Run teacher inference on validation set
# Check that teacher predictions are good
```

### 3. Match Sequence Lengths

Ensure student and teacher use the same `max_total_sequence_length`:

```yaml
student_policy:
  max_total_sequence_length: 4096

teacher:
  max_total_sequence_length: ${student_policy.max_total_sequence_length}
```

### 4. Use Lower Precision for Teacher

```yaml
teacher:
  precision: "float16"   # Saves memory

student_policy:
  precision: "bfloat16" # Better for training
```

### 5. Monitor Resource Utilization

Check timing breakdown:
- **Teacher > 30% of step time**: Increase teacher GPUs
- **Student > 80% of step time**: Reduce teacher GPUs, give to student
- **Data processing > 10%**: Use faster data pipeline or more workers

### 6. Checkpoint Regularly

```yaml
checkpointing:
  enabled: true
  save_period: 500      # Save every 500 steps
  keep_top_k: 5         # Keep best 5 checkpoints
  metric_name: "val_loss"
  higher_is_better: false
```

Checkpoints include teacher metadata for reproducibility.

### 7. Use Validation to Guide Hyperparameters

```yaml
kd:
  val_period: 100       # Validate frequently when tuning
  val_at_start: true    # Establish baseline
```

Compare `val_loss` with different `alpha` and `temperature` values.

## Troubleshooting

### Common Errors

#### Tokenizer Mismatch

```
ValueError: teacher_logprobs must be provided in data dict for CombinedKDLoss
```

**Cause**: Student and teacher use incompatible tokenizers

**Fix**: Ensure both use the same tokenizer:
```yaml
student_policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  tokenizer:
    name: ${student_policy.model_name}

teacher:
  model_name: "Qwen/Qwen2.5-7B"  # Must have same tokenizer family
```

#### Cluster Allocation Failure

```
ValueError: No nodes remaining for student training
```

**Cause**: Teacher allocation uses all available nodes

**Fix**: Adjust allocation:
```yaml
cluster:
  num_nodes: 5  # Increase total

teacher:
  cluster:
    num_nodes: 1  # Reduce teacher
```

#### Out of Memory (OOM)

**Symptoms**: CUDA OOM errors during teacher inference

**Solutions**:
1. **Note**: Tensor parallelism (TP > 1) not currently supported
2. Use lower precision:
   ```yaml
   teacher:
     precision: "float16"  # Or "bfloat16"
   ```
3. Reduce batch size:
   ```yaml
   student_policy:
     train_micro_batch_size: 2  # Reduce from 4
   ```
4. Use gradient checkpointing:
   ```yaml
   student_policy:
     dtensor_v2_cfg:
       activation_checkpointing: true
   ```

2. Use lower precision:
   ```yaml
   teacher:
     precision: "float16"  # From bfloat16
   ```

3. Reduce batch size:
   ```yaml
   student_policy:
     logprob_batch_size: 2  # Reduce from 4
   ```

#### Slow Training

**Symptom**: Training much slower than expected

**Diagnosis**: Check timing breakdown in logs

**Solutions**:
- If teacher is slow: Increase `teacher.cluster.num_nodes`
- If student is slow: Reduce teacher allocation, give GPUs to student
- If both: Reduce batch size or sequence length

### Performance Issues

#### KD Loss Not Decreasing

**Possible causes**:
1. **Temperature mismatch**: Try different T values (1.5 - 5.0)
2. **Teacher quality**: Teacher may not be helpful for this task
3. **Learning rate**: Student LR may be too high/low

**Debug steps**:
1. Check `base_loss` - if it's decreasing, student is learning
2. Compare teacher predictions to labels - is teacher actually better?
3. Try pure supervised (α=0) as baseline

#### Validation Loss Worse Than Baseline

**Possible causes**:
1. **alpha too high**: Reduce α to preserve ground truth
2. **Teacher overfitting**: Teacher memorized training data
3. **Different domains**: Teacher trained on different distribution

**Solutions**:
1. Reduce `kd.alpha` to 0.3-0.5
2. Increase regularization (dropout, weight decay)
3. Use domain-specific teacher or reduce α

## Advanced Topics

### Mixed Precision Training

For large-scale KD:

```yaml
teacher:
  precision: "float16"       # Teacher inference

student_policy:
  precision: "bfloat16"      # Student training
  dtensor_v2_cfg:
    activation_checkpointing: true  # For long sequences
```

### Curriculum Learning

Start with high α, gradually reduce:

```python
# In scheduler config (future feature)
alpha_schedule:
  start: 0.9
  end: 0.5
  steps: 10000
```

*Note: Currently use fixed `alpha`. Scheduling is a future enhancement.*

### Different Teacher/Student Architectures

KD works across architectures:

```yaml
student_policy:
  model_name: "Qwen/Qwen2.5-1.5B"  # Qwen architecture

teacher:
  model_name: "meta-llama/Llama-3.2-3B"  # Llama architecture
```

**Requirements**:
- Compatible tokenizer (same vocab)
- Same vocab size
- Same max sequence length

### Sequence Packing with KD

KD supports sequence packing for efficiency:

```yaml
student_policy:
  sequence_packing:
    enabled: true
    train_mb_tokens: 8192
    algorithm: "modified_first_fit_decreasing"
```

Teacher inference is automatically run on packed sequences.

### MoE Teacher Models with Expert Parallelism

For Mixture-of-Experts (MoE) teachers, use expert parallelism to distribute experts across GPUs:

```yaml
teacher:
  model_name: "mistralai/Mixtral-8x7B-v0.1"
  expert_parallel_size: 4  # Distribute 8 experts across 4 GPUs
  tensor_parallel_size: 1  # TP still not supported
  cluster:
    num_nodes: 1
    gpus_per_node: 4
```

**Supported MoE models**: Mixtral, Qwen-MoE, DeepSeek-MoE, and other HuggingFace MoE architectures.

**Why use EP for MoE teachers**:
- Reduces memory per GPU (expert weights distributed)
- Enables using larger MoE teachers that wouldn't fit on single GPU
- Maintains full vocabulary logprobs needed for KD (unlike TP)

**Configuration tips**:
- Set `expert_parallel_size` to divide evenly into teacher GPU count
- Use `precision: "float16"` to save memory
- Monitor `teacher_inference` timing to ensure no bottleneck

### Checkpointing Teacher Metadata

Checkpoints save teacher info for reproducibility:

```python
# Saved in checkpoint_dir/teacher_metadata.pt
{
    "model_name": "Qwen/Qwen2.5-7B",
    "checkpoint_path": null,
    "precision": "float16",
}
```

Use this to verify which teacher was used for each checkpoint.

## Related Documentation

- [SFT Guide](sft.md) - Standard supervised fine-tuning
- [GRPO Guide](grpo.md) - Reinforcement learning with policy optimization
- [Cluster Setup](../cluster.md) - Multi-node training configuration
- [Chat Datasets](../design-docs/chat-datasets.md) - Dataset format requirements

## References

- Hinton et al. (2015): [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- Gou et al. (2021): [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review example configs in `examples/configs/kd/`
3. Check timing metrics in logs for performance bottlenecks
