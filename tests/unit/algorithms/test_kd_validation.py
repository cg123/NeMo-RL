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

"""Tests for Knowledge Distillation validation checks."""

import pytest
import torch

from rlkit.algorithms.loss_functions import (
    KnowledgeDistillationLoss,
    CombinedKDLoss,
    NLLLoss,
)


def test_kd_loss_temperature_validation():
    """Test that KnowledgeDistillationLoss rejects invalid temperature values."""
    # Valid temperatures should work
    loss_fn = KnowledgeDistillationLoss(temperature=1.0)
    assert loss_fn.temperature == 1.0

    loss_fn = KnowledgeDistillationLoss(temperature=5.0)
    assert loss_fn.temperature == 5.0

    # Zero temperature should fail
    with pytest.raises(ValueError, match="Temperature must be positive"):
        KnowledgeDistillationLoss(temperature=0.0)

    # Negative temperature should fail
    with pytest.raises(ValueError, match="Temperature must be positive"):
        KnowledgeDistillationLoss(temperature=-1.0)


def test_combined_kd_loss_alpha_validation():
    """Test that CombinedKDLoss rejects invalid alpha values."""
    base_loss = NLLLoss()
    kd_loss = KnowledgeDistillationLoss(temperature=2.0)

    # Valid alphas should work
    combined = CombinedKDLoss(base_loss, kd_loss, alpha=0.0)
    assert combined.alpha == 0.0

    combined = CombinedKDLoss(base_loss, kd_loss, alpha=0.5)
    assert combined.alpha == 0.5

    combined = CombinedKDLoss(base_loss, kd_loss, alpha=1.0)
    assert combined.alpha == 1.0

    # Alpha > 1 should fail
    with pytest.raises(ValueError, match="Alpha must be in"):
        CombinedKDLoss(base_loss, kd_loss, alpha=1.5)

    # Alpha < 0 should fail
    with pytest.raises(ValueError, match="Alpha must be in"):
        CombinedKDLoss(base_loss, kd_loss, alpha=-0.1)


# Note: Integration tests for KDTrainer TP validation would go here,
# but they require full KDTrainer setup with Ray clusters, datasets, etc.
# Those are better suited for integration test suites.
#
# The key validation logic is:
# 1. KDTrainer._initialize_student_policy() rejects student TP > 1
# 2. KDTrainer._initialize_teacher_policy() rejects teacher TP > 1
#
# Example integration test structure:
#
# def test_kd_trainer_rejects_teacher_vocab_parallelism():
#     config = create_minimal_kd_config()
#     config["teacher"]["tensor_parallel_size"] = 2
#
#     with pytest.raises(NotImplementedError, match="vocab parallelism"):
#         trainer = KDTrainer(config, tokenizer, train_ds, val_ds)
#
# def test_kd_trainer_rejects_student_vocab_parallelism():
#     config = create_minimal_kd_config()
#     config["student_policy"]["dtensor_v2_cfg"]["tensor_parallel_size"] = 2
#
#     with pytest.raises(NotImplementedError, match="vocab parallelism"):
#         trainer = KDTrainer(config, tokenizer, train_ds, val_ds)
