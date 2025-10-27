# Knowledge Distillation Quick Reference

A cheat sheet for common KD operations in NeMo RL.

## ⚠️ Current Limitations

**Tensor Parallelism Not Supported**: Both teacher and student must use `tensor_parallel_size: 1`. Models must fit on single GPU (~7B-70B depending on GPU memory and precision).

## Launch Commands

```bash
# Small example (2 nodes)
uv run examples/run_kd.py --config examples/configs/kd/qwen3_4B_to_1B.yaml

# Multi-node example (3 nodes)
uv run examples/run_kd.py --config examples/configs/kd/qwen3_7B_to_1.5B.yaml

# With overrides
uv run examples/run_kd.py \\
  --config examples/configs/kd/qwen3_4B_to_1B.yaml \\
  kd.alpha=0.8 \\
  kd.temperature=3.0
```

## Configuration Cheat Sheet

### Minimal Config

```yaml
cluster:
  num_nodes: 2
  gpus_per_node: 4

student_policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  # ... standard policy config ...

teacher:
  model_name: "Qwen/Qwen2.5-4B"
  precision: "float16"
  max_total_sequence_length: \${student_policy.max_total_sequence_length}
  cluster:
  num_nodes: 1
    gpus_per_node: 4
  tensor_parallel_size: 1  # TP > 1 not currently supported

kd:
  alpha: 0.5
  temperature: 2.0
  max_num_steps: 5000
  val_period: 100
```

## Hyperparameter Guidelines

### KD Weight (α)

| Use Case | Value | Example |
|----------|-------|--------|
| High-quality teacher | 0.7-0.9 | GPT-4 → GPT-3.5 |
| Balanced | 0.5 | Similar-sized models |
| Noisy labels | 0.6-0.8 | Web-scraped data |
| Good labels | 0.3-0.5 | Curated datasets |

### Temperature (T)

| Model Gap | Value | Example |
|-----------|-------|--------|
| Large | 3.0-5.0 | 7B → 1B |
| Medium | 2.0-3.0 | 4B → 1.5B |
| Small | 1.5-2.0 | 3B → 2B |

## Common Overrides

```bash
# Adjust KD weight
kd.alpha=0.8

# Adjust temperature
kd.temperature=4.0

# Change learning rate
student_policy.optimizer.kwargs.lr=1.0e-5

# Use custom dataset
data.dataset_name="my/dataset"

# Adjust cluster
cluster.num_nodes=10 teacher.cluster.num_nodes=2

# Change batch size
student_policy.train_global_batch_size=128

# Validation frequency
kd.val_period=200

# Save checkpoints more often
checkpointing.save_period=100
```

## Cluster Allocation Calculator

```
Total GPUs = num_nodes × gpus_per_node
Teacher GPUs = teacher.cluster.num_nodes × teacher.cluster.gpus_per_node
Student GPUs = Total - Teacher
```

**Example**:
```
cluster.num_nodes=5, cluster.gpus_per_node=8     → Total = 40 GPUs
teacher.cluster.num_nodes=1, gpus_per_node=8     → Teacher = 8 GPUs
                                                  → Student = 32 GPUs
```

## Troubleshooting Quick Fixes

### Cluster Allocation Error
```
ValueError: No nodes remaining for student
```
**Fix**: `teacher.cluster.num_nodes=1` (reduce teacher allocation)

### Tokenizer Mismatch
```
ValueError: teacher_logprobs must be provided
```
**Fix**: Ensure teacher/student use same tokenizer

### OOM on Teacher
```
CUDA out of memory (teacher inference)
```
**Fix**: Use `teacher.precision="float16"` or reduce batch size (TP > 1 not supported)

### Training Too Slow
```
teacher_inference: 45% of step time
```
**Fix**: `teacher.cluster.num_nodes=2` (add more teacher GPUs)

## Metrics Interpretation

### Good Training
```
train/base_loss: 2.5 → 2.1  ✓ Decreasing
train/kd_loss: 1.8 → 1.5    ✓ Decreasing  
train/kd_loss < base_loss    ✓ Teacher helps
val_loss: 2.2 → 1.9         ✓ Generalizing
```

### Warning Signs
```
train/kd_loss > base_loss    ⚠️ Teacher may not be helpful
val_loss increasing          ⚠️ Overfitting or bad hyperparams
grad_norm > 10.0             ⚠️ Gradient instability
```

## File Locations

```
examples/
├── run_kd.py                      # Entry point
├── configs/kd/
    ├── qwen3_4B_to_1B.yaml         # Small example
    ├── qwen3_7B_to_1.5B.yaml       # Multi-node example
    └── README.md                   # Config guide

rlkit/
├── algorithms/
│   ├── kd.py                      # KDTrainer
│   └── loss_functions.py          # KD losses
└── config/
    └── kd.py                      # Config schema

docs/
├── guides/
│   └── kd.md                      # User guide
└── design-docs/
    └── knowledge-distillation.md  # Design doc
```

## See Also

- [Full User Guide](../guides/kd.md)
- [Design Documentation](../design-docs/knowledge-distillation.md)
- [Example Configs](../../examples/configs/kd/README.md)
