# Knowledge Distillation Example Configurations

This directory contains example configurations for knowledge distillation (KD) training.

## Available Configs

### `qwen3_4B_to_1B.yaml`
**Small-scale example for testing**

- **Student**: Qwen2.5-1.5B
- **Teacher**: Qwen2.5-4B  
- **Cluster**: 2 nodes × 4 GPUs = 8 GPUs total
  - Teacher: 1 node × 4 GPUs (TP=2)
  - Student: 1 node × 4 GPUs (TP=1)
- **Training**:
  - Global batch size: 64
  - Max sequence length: 2048
  - KD weight: 0.5 (50/50 mix of supervised + distillation)
  - Temperature: 2.0
- **Use case**: Quick testing, development, small GPU clusters

### `qwen3_32B_to_4B.yaml`
**Production-scale example**

- **Student**: Qwen2.5-4B
- **Teacher**: Qwen2.5-32B
- **Cluster**: 5 nodes × 8 GPUs = 40 GPUs total
  - Teacher: 1 node × 8 GPUs (TP=8)
  - Student: 4 nodes × 8 GPUs (TP=2)
- **Training**:
  - Global batch size: 256
  - Max sequence length: 4096
  - KD weight: 0.7 (emphasize teacher)
  - Temperature: 3.0 (softer distributions)
- **Use case**: Production distillation, large-scale clusters

## How to Run

### Basic Usage

```bash
# Small example (2 nodes)
uv run examples/run_kd.py --config examples/configs/kd/qwen3_4B_to_1B.yaml

# Large example (5 nodes)
uv run examples/run_kd.py --config examples/configs/kd/qwen3_32B_to_4B.yaml
```

### With Overrides

```bash
# Change KD weight
uv run examples/run_kd.py \\
  --config examples/configs/kd/qwen3_4B_to_1B.yaml \\
  kd.alpha=0.8

# Change temperature
uv run examples/run_kd.py \\
  --config examples/configs/kd/qwen3_4B_to_1B.yaml \\
  kd.temperature=4.0

# Use different dataset
uv run examples/run_kd.py \\
  --config examples/configs/kd/qwen3_4B_to_1B.yaml \\
  data.dataset_name="your/dataset"

# Adjust cluster allocation
uv run examples/run_kd.py \\
  --config examples/configs/kd/qwen3_32B_to_4B.yaml \\
  cluster.num_nodes=8 \\
  teacher.cluster.num_nodes=2
```

## Key Configuration Parameters

### Distillation Hyperparameters

- **`kd.alpha`** (0.0 - 1.0): Weight for distillation loss
  - 0.0 = pure supervised learning (ignore teacher)
  - 0.5 = equal mix of supervised + distillation
  - 1.0 = pure distillation (ignore labels)
  - **Recommendation**: Start with 0.5, increase if teacher is high quality

- **`kd.temperature`** (≥ 1.0): Temperature for softening logits
  - 1.0 = no softening (standard softmax)
  - 2.0-4.0 = typical range for KD
  - Higher = softer distributions, more "dark knowledge"
  - **Recommendation**: Start with 2.0, increase for larger teacher-student gaps

### Cluster Allocation

- **`cluster.num_nodes`**: Total nodes available
- **`cluster.gpus_per_node`**: GPUs per node
- **`teacher.cluster.num_nodes`**: Nodes dedicated to teacher
- **`teacher.cluster.gpus_per_node`**: GPUs per node for teacher
- **Student gets**: `(total_nodes - teacher_nodes) × gpus_per_node`

**Example**:
```yaml
cluster:
  num_nodes: 5
  gpus_per_node: 8
teacher:
  cluster:
    num_nodes: 1      # Teacher gets 8 GPUs
    gpus_per_node: 8
# Student automatically gets 4 nodes × 8 GPUs = 32 GPUs
```

### Teacher Configuration

- **`teacher.model_name`**: HuggingFace model or path
- **`teacher.checkpoint_path`**: (Optional) Local checkpoint to load
- **`teacher.precision`**: "float16" or "bfloat16" (float16 saves memory)
- **`teacher.tensor_parallel_size`**: TP for teacher model
  - Rule of thumb: `model_size / 3B` (e.g., 32B → TP=8-10)

### Student Configuration

Same as standard NeMo-RL policy configuration. Key differences:
- Use `student_policy` instead of `policy`
- Teacher and student must use **same tokenizer**
- Sequence lengths must match

## Tips & Best Practices

### Choosing KD Weight

1. **High-quality teacher** (e.g., GPT-4 → GPT-3.5): Use high KD weight (0.7-0.9)
2. **Similar-sized models** (e.g., 7B → 3B): Use balanced weight (0.5)
3. **Noisy labels**: Increase KD weight to rely more on teacher
4. **High-quality labels**: Decrease KD weight to preserve ground truth

### Choosing Temperature

1. **Large gap** (32B → 1B): Use higher temperature (3.0-5.0)
2. **Small gap** (7B → 3B): Use moderate temperature (2.0-3.0)
3. **Same architecture**: Start with 2.0
4. **Different architectures**: May need experimentation

### Resource Planning

**Teacher overhead**: ~20-30% of total training time
- If teacher inference is slow, dedicate more GPUs to teacher
- If student training is slow, reduce teacher allocation

**Memory optimization**:
- Use `teacher.precision="float16"` (vs. `bfloat16` for student)
- Teacher doesn't need activation checkpointing
- Student should use activation checkpointing for large contexts

### Debugging

**Cluster allocation errors**:
```
ValueError: No nodes remaining for student training
```
→ Reduce `teacher.cluster.num_nodes` or increase `cluster.num_nodes`

**Tokenizer mismatch**:
```
ValueError: teacher_logits must be provided
```
→ Check that teacher and student use compatible tokenizers

**Slow training**:
- Check timing breakdown in logs (`teacher_inference` vs `student_training`)
- Adjust cluster allocation to balance teacher/student time
- Consider pipelining (future feature)

## Monitoring

### Key Metrics to Watch

- **`train/total_loss`**: Combined supervised + KD loss
- **`train/base_loss`**: Supervised component (cross-entropy with labels)
- **`train/kd_loss`**: Distillation component (KL divergence with teacher)
- **`validation/val_loss`**: Total validation loss
- **`timing/teacher_inference`**: Time spent on teacher inference
- **`timing/student_training`**: Time spent on student training

### Healthy Training

- `kd_loss` should be **lower** than `base_loss` (teacher provides better signal)
- `val_loss` should **decrease** over time
- `teacher_inference` should be **<30%** of total step time (otherwise bottleneck)

### Troubleshooting

**KD loss not improving**:
- Temperature too high/low → Adjust `kd.temperature`
- Teacher not helpful → Reduce `kd.alpha`
- Tokenizer mismatch → Check `_validate_tokenizers` logs

**Training too slow**:
- Teacher bottleneck → Increase `teacher.cluster.num_nodes`
- Student bottleneck → Increase student cluster allocation
- Both slow → Reduce batch size or sequence length

