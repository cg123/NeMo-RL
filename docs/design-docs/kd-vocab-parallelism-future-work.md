# Future Work: Vocab Parallelism Support for Knowledge Distillation

## Status

**Current**: Vocab parallelism (TP > 1) is **not supported** for KD  
**Priority**: Medium (only needed for very large student models)  
**Estimated Effort**: 3-5 days including testing  

## Problem Summary

Knowledge Distillation requires computing KL divergence over the **full vocabulary distribution**:

```
KL(teacher || student) = ∑_i P_teacher(i) * log(P_teacher(i) / P_student(i))
```

With tensor parallelism (vocab parallelism), each rank only has a **shard** of the vocabulary:
- Student logits: `[batch, seq, vocab_size/tp_size]` on each rank
- Teacher logprobs: Currently returns per-token values `[batch, seq]`, not distributions

This creates two issues:

1. **Teacher Issue**: `Policy.get_logprobs()` doesn't return full vocab distributions
2. **Student Issue**: Computing `softmax(logits_shard)` gives wrong probabilities because the partition function is incomplete

## Why Current Code Fails

### Teacher Problem

The `Policy.get_logprobs()` method is designed for RL algorithms that only need the log probability of the **actual next token**:

```python
# In dtensor_policy_worker.py, get_logprobs() returns:
return_data["logprobs"] = torch.cat(all_log_probs, dim=0)  # Shape: [batch, seq]
```

With TP, this efficiently gathers just the one value needed from each rank's vocab shard. But KD needs the **entire distribution** for all vocab entries.

### Student Problem

The `KnowledgeDistillationLoss` naively computes:

```python
student_log_probs = F.log_softmax(next_token_logits / T, dim=-1)
```

With TP, `next_token_logits` has shape `[batch, seq, vocab_size/tp_size]`. The softmax normalization:

```python
P(i) = exp(logit_i) / ∑_j exp(logit_j)
```

Only sums over the **local vocab shard**, giving completely wrong probabilities!

Correct normalization requires:
```python
P(i) = exp(logit_i) / ∑_all_vocab exp(logit_j)  # Need all-reduce here!
```

## Recommended Solution: Matched TP Sharding (Option 3)

If teacher and student use **identical TP configurations**, we can compute KL divergence correctly by properly handling the partition function across ranks.

### Mathematical Foundation

The key insight is that KL divergence is a **sum over vocabulary**:

```
KL(P||Q) = ∑_all_i P(i) * log(P(i) / Q(i))
         = ∑_ranks [ ∑_i_in_shard P(i) * log(P(i) / Q(i)) ]
```

We can distribute the computation across vocab shards **if and only if** we compute probabilities P(i) and Q(i) correctly with the **full partition function**.

### Implementation Approach

#### Step 1: Compute Global Partition Function

For numerical stability, use the log-sum-exp trick:

```
Z = ∑_i exp(logit_i) = exp(max_logit) * ∑_i exp(logit_i - max_logit)
```

With TP, we need all-reduce operations:

```python
# Step 1: Find global maximum across all vocab shards
local_max = student_logits.max(dim=-1, keepdim=True)[0]  # [batch, seq, 1]
global_max = local_max.clone()
torch.distributed.all_reduce(
    global_max, 
    op=torch.distributed.ReduceOp.MAX,
    group=vocab_parallel_group
)

# Step 2: Compute global partition sum
local_exp_sum = (student_logits - global_max).exp().sum(dim=-1, keepdim=True)
global_exp_sum = local_exp_sum.clone()
torch.distributed.all_reduce(
    global_exp_sum,
    op=torch.distributed.ReduceOp.SUM,
    group=vocab_parallel_group
)

# Step 3: Compute correct log probabilities
student_log_probs = student_logits - global_max - global_exp_sum.log()
```

#### Step 2: Modify KnowledgeDistillationLoss

Add vocab parallel path:

```python
class KnowledgeDistillationLoss(LossFunction):
    def __call__(
        self,
        next_token_logits: Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        
        teacher_logprobs = data["teacher_logprobs"]  # Shape: [batch, seq, vocab_shard]
        T = self.temperature
        
        if vocab_parallel_group is not None:
            # VOCAB PARALLEL PATH
            student_log_probs = self._compute_tp_log_softmax(
                next_token_logits / T, vocab_parallel_group
            )
            teacher_logprobs_normalized = self._renormalize_with_tp(
                teacher_logprobs / T, vocab_parallel_group
            )
            
            # Compute KL divergence on local shard
            local_kl_div = F.kl_div(
                student_log_probs,
                teacher_logprobs_normalized,
                reduction='none',
                log_target=True
            )
            local_kl_div = local_kl_div.sum(dim=-1)  # Sum over local vocab
            
            # Sum KL contributions across all vocab shards
            kl_div = local_kl_div.clone()
            torch.distributed.all_reduce(
                kl_div, op=torch.distributed.ReduceOp.SUM,
                group=vocab_parallel_group
            )
        else:
            # NO VOCAB PARALLEL: existing simple path
            student_log_probs = F.log_softmax(next_token_logits / T, dim=-1)
            # ... existing code ...
        
        kl_div = kl_div * (T ** 2)
        masked_kl = masked_mean(kl_div, token_mask, sample_mask, ...)
        return masked_kl, {"kd_loss": masked_kl.item(), "temperature": T}
    
    def _compute_tp_log_softmax(
        self, logits: Tensor, vocab_parallel_group
    ) -> Tensor:
        """Compute log_softmax with global partition function."""
        local_max = logits.max(dim=-1, keepdim=True)[0]
        global_max = local_max.clone()
        torch.distributed.all_reduce(
            global_max, op=torch.distributed.ReduceOp.MAX, 
            group=vocab_parallel_group
        )
        
        local_exp_sum = (logits - global_max).exp().sum(dim=-1, keepdim=True)
        global_exp_sum = local_exp_sum.clone()
        torch.distributed.all_reduce(
            global_exp_sum, op=torch.distributed.ReduceOp.SUM, 
            group=vocab_parallel_group
        )
        
        return logits - global_max - global_exp_sum.log()
    
    def _renormalize_with_tp(
        self, teacher_logprobs: Tensor, vocab_parallel_group
    ) -> Tensor:
        """Renormalize teacher log probs using global partition."""
        # Similar logic to _compute_tp_log_softmax
        local_max = teacher_logprobs.max(dim=-1, keepdim=True)[0]
        global_max = local_max.clone()
        torch.distributed.all_reduce(
            global_max, op=torch.distributed.ReduceOp.MAX,
            group=vocab_parallel_group
        )
        
        local_sum = (teacher_logprobs - global_max).exp().sum(dim=-1, keepdim=True)
        global_sum = local_sum.clone()
        torch.distributed.all_reduce(
            global_sum, op=torch.distributed.ReduceOp.SUM,
            group=vocab_parallel_group
        )
        
        return teacher_logprobs - global_max - global_sum.log()
```

#### Step 3: Update Teacher to Return Full Distributions

Option A: Add new method `get_full_logprobs()`

```python
class Policy:
    def get_full_logprobs(self, data) -> BatchedDataDict:
        """Get full vocabulary log probabilities [batch, seq, vocab_size].
        
        With TP, returns vocab-sharded distributions [batch, seq, vocab_size/tp_size]
        where probabilities are correctly computed with global partition function.
        """
        # Similar to get_logprobs() but return full log_probs tensor
        # instead of gathering per-token values
        pass
```

Option B: Add flag to existing `get_logprobs()` method

```python
def get_logprobs(self, data, return_full_distributions=False):
    if return_full_distributions:
        # Return [batch, seq, vocab_shard] instead of [batch, seq]
        pass
```

#### Step 4: Add Validation

```python
def _validate_matched_tp_config(self):
    """Validate teacher and student have identical TP configuration."""
    student_tp = self.master_config["student_policy"]["dtensor_v2_cfg"]["tensor_parallel_size"]
    teacher_tp = self.master_config["teacher"].get("tensor_parallel_size", 1)
    
    if student_tp != teacher_tp:
        raise ValueError(
            f"With vocab parallelism enabled, teacher and student must have matching TP size. "
            f"Got teacher_tp={teacher_tp}, student_tp={student_tp}. "
            f"This is required for correct KL divergence computation across vocab shards."
        )
```

## Benefits of Matched TP Sharding

1. **Memory Efficient**: Only partition function scalars are all-reduced (~few KB)
2. **Numerically Stable**: Uses log-sum-exp trick throughout  
3. **Scalable**: Supports arbitrary TP sizes (as long as teacher=student)
4. **Correct**: Mathematically equivalent to non-TP version
5. **Communication Efficient**: All-reduce is cheaper than all-gather

## Testing Plan

### Unit Tests

1. Test `_compute_tp_log_softmax()` matches `F.log_softmax()` when gathered
2. Test KL divergence with TP matches non-TP version (within numerical precision)
3. Test with different TP sizes (2, 4, 8)
4. Test numerical stability with extreme logit values (±1000)

### Integration Tests  

1. Train tiny model (e.g., 125M) with TP=2 and verify loss values
2. Compare training curves: TP=1 vs TP=2 (should be identical)
3. Test gradient correctness by comparing with gathered version

### Example Test

```python
def test_kd_loss_with_vocab_parallel():
    """Test KD loss with TP matches non-TP version."""
    # Setup
    batch, seq, vocab = 2, 4, 128
    tp_size = 4
    vocab_per_rank = vocab // tp_size
    
    # Create test data
    student_logits_full = torch.randn(batch, seq, vocab)
    teacher_logprobs_full = F.log_softmax(
        torch.randn(batch, seq, vocab), dim=-1
    )
    
    # Non-TP version
    loss_fn = KnowledgeDistillationLoss(temperature=2.0)
    loss_no_tp, _ = loss_fn(student_logits_full, ...)
    
    # TP version (simulate sharding)
    losses_tp = []
    for rank in range(tp_size):
        start = rank * vocab_per_rank
        end = (rank + 1) * vocab_per_rank
        
        student_shard = student_logits_full[:, :, start:end]
        teacher_shard = teacher_logprobs_full[:, :, start:end]
        
        # Compute with TP logic (mocked all-reduce)
        loss_tp, _ = loss_fn(
            student_shard, ...,
            vocab_parallel_group=mock_pg
        )
        losses_tp.append(loss_tp)
    
    # Should match
    assert torch.allclose(loss_no_tp, loss_tp, rtol=1e-5)
```

## Alternative: All-Gather Full Distributions (Option 2)

Instead of matched sharding, gather the full vocabulary on all ranks:

```python
# Gather student logits
logits_list = [torch.zeros_like(next_token_logits) 
               for _ in range(tp_size)]
torch.distributed.all_gather(
    logits_list, next_token_logits, 
    group=vocab_parallel_group
)
full_student_logits = torch.cat(logits_list, dim=-1)

# Similarly for teacher
full_teacher_logprobs = ...

# Now compute KD loss normally
```

**Drawbacks**:
- **High memory**: 32k vocab × 4k seq × batch=4 × fp16 = 1GB per batch
- **Communication overhead**: All-gather more expensive than all-reduce
- **Doesn't scale**: Memory grows O(vocab_size)

**When to use**: Only if teacher and student have **different** TP sizes (rare)

## Implementation Checklist

- [ ] Implement `_compute_tp_log_softmax()` helper in KnowledgeDistillationLoss
- [ ] Implement `_renormalize_with_tp()` helper
- [ ] Add vocab_parallel_group handling to `KnowledgeDistillationLoss.__call__()`
- [ ] Add `Policy.get_full_logprobs()` method or modify existing
- [ ] Update `KDTrainer._get_teacher_logprobs()` to use full distributions
- [ ] Add `_validate_matched_tp_config()` validation to KDTrainer
- [ ] Remove "TP not supported" error messages (or make conditional)
- [ ] Write unit tests for TP log_softmax computation
- [ ] Write integration tests comparing TP vs non-TP training
- [ ] Update user documentation with TP support details
- [ ] Add example config with TP enabled
- [ ] Performance testing: measure overhead of all-reduce operations

## References

- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Megatron-LM: https://arxiv.org/abs/1909.08053
- PyTorch FSDP2 and DTensor: https://pytorch.org/docs/stable/distributed.tensor.html
- Log-sum-exp trick: https://en.wikipedia.org/wiki/LogSumExp

## Related Issues

- Current validation in `rlkit/algorithms/kd.py:_initialize_student_policy()`
- Current validation in `rlkit/algorithms/kd.py:_initialize_teacher_policy()`
- See error messages for detailed explanation of current limitations
