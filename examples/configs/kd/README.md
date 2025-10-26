# Knowledge Distillation Example Configurations

This directory contains example configurations for knowledge distillation (KD) training.

## ⚠️ Current Limitations

**Tensor Parallelism Not Supported**: Both teacher and student must use `tensor_parallel_size: 1`. This limits model sizes to what fits on a single GPU (~7B-14B with fp16/bf16 on A100 80GB).

**Expert Parallelism Supported**: For MoE teacher models, you can use `expert_parallel_size > 1` to distribute experts across GPUs.

## Available Configs

### `qwen3_4B_to_1B.yaml`
**Small-scale example for testing**

- **Student**: Qwen2.5-1.5B
- **Teacher**: Qwen2.5-4B  
- **Cluster**: 2 nodes × 4 GPUs = 8 GPUs total
  - Teacher: 1 node × 4 GPUs (TP=1)
  - Student: 1 node × 4 GPUs (TP=1)
- **Training**:
  - Global batch size: 64
  - Max sequence length: 2048
  - KD weight: 0.5 (50/50 mix of supervised + distillation)
  - Temperature: 2.0
- **Use case**: Quick testing, development, small GPU clusters

### `qwen3_7B_to_1.5B.yaml`
**Multi-node example**

- **Student**: Qwen2.5-1.5B
- **Teacher**: Qwen2.5-7B
- **Cluster**: 3 nodes × 8 GPUs = 24 GPUs total
  - Teacher: 1 node × 8 GPUs (TP=1)
  - Student: 2 nodes × 8 GPUs (TP=1)
- **Training**:
  - Global batch size: 256
  - Max sequence length: 4096
  - KD weight: 0.6
  - Temperature: 2.5
- **Use case**: Multi-node training, larger models

### `qwen_moe_to_1.5B.yaml`
**MoE teacher with expert parallelism**

- **Student**: Qwen2.5-1.5B (dense model)
- **Teacher**: Qwen2.5-14B-Instruct (MoE model placeholder)
- **Cluster**: 2 nodes × 8 GPUs = 16 GPUs total
  - Teacher: 1 node × 8 GPUs (TP=1, **EP=4**)
  - Student: 1 node × 8 GPUs (TP=1)
- **Training**:
  - Global batch size: 64
  - Max sequence length: 2048
  - KD weight: 0.6
  - Temperature: 2.5
- **Use case**: Distilling MoE models to dense models, expert parallelism

## How to Run

### Basic Usage

```bash
# Small example (2 nodes)
uv run examples/run_kd.py --config examples/configs/kd/qwen3_4B_to_1B.yaml

# Multi-node example (3 nodes)
uv run examples/run_kd.py --config examples/configs/kd/qwen3_7B_to_1.5B.yaml

# MoE teacher with expert parallelism
uv run examples/run_kd.py --config examples/configs/kd/qwen_moe_to_1.5B.yaml
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
  --config examples/configs/kd/qwen3_7B_to_1.5B.yaml \\
  cluster.num_nodes=4 \\
  teacher.cluster.num_nodes=1
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
- **`teacher.tensor_parallel_size`**: Must be 1 (TP > 1 not currently supported)
- **`teacher.expert_parallel_size`**: For MoE models (e.g., Mixtral, Qwen-MoE)
  - Distributes expert computation across GPUs
  - Example: `expert_parallel_size: 2` for Mixtral-8x7B

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

1. **Large gap** (7B → 1B): Use higher temperature (3.0-5.0)
2. **Medium gap** (4B → 1.5B): Use moderate temperature (2.0-3.0)
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
ValueError: teacher_logprobs must be provided
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

## Using MoE Teacher Models

### Expert Parallelism

For Mixture-of-Experts (MoE) teacher models, you can use expert parallelism to distribute expert computation across GPUs:

```yaml
teacher:
  model_name: "mistralai/Mixtral-8x7B-v0.1"  # 8 experts
  expert_parallel_size: 4  # Distribute across 4 GPUs
  tensor_parallel_size: 1  # TP not supported
  cluster:
    num_nodes: 1
    gpus_per_node: 4
```

### Supported MoE Models

- **Mixtral**: `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x22B-v0.1`
- **Qwen-MoE**: Any Qwen2.5 MoE variant
- **DeepSeek-MoE**: DeepSeek MoE models
- **Custom MoE**: Any HuggingFace MoE architecture

### EP Configuration Guidelines

**Choosing expert_parallel_size**:
- Should divide evenly into the number of GPUs allocated to teacher
- Typical values: 2, 4, 8 depending on model size and GPU count
- Example: Mixtral-8x7B with EP=4 puts 2 experts per GPU

**Memory considerations**:
- EP reduces memory per GPU (distributes expert weights)
- Still limited by TP=1 constraint (shared layers must fit on single GPU)
- Use lower precision (`float16`) if memory constrained

**Performance**:
- EP adds communication overhead for expert routing
- Best for inference-heavy workloads (like teacher in KD)
- Monitor `teacher_inference` timing to ensure not a bottleneck

