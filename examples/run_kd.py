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

import argparse
import asyncio
import logging
import os
import pprint

# Prevent Ray from dumping a full copy of all of our venvs into /tmp every time this runs.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

from datasets import Dataset, load_dataset, load_from_disk
from omegaconf import OmegaConf
import torch
from transformers import AutoTokenizer

from rlkit.config import (
    KDMasterConfig as MasterConfig,
    KD_DEFAULT_ALPHA,
    KD_DEFAULT_TEMPERATURE,
)
from rlkit.algorithms.kd import KDTrainer
from rlkit.data.datasets import transform_dataset
from rlkit.algorithms.utils import get_tokenizer
from rlkit.config import DataConfig
from rlkit.distributed.virtual_cluster import init_ray
from rlkit.utils.config import load_config, parse_hydra_overrides
from rlkit.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

# Avoid asyncio spamming console on crash
logging.getLogger("asyncio").setLevel(logging.ERROR)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Knowledge Distillation training with configuration"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig):
    """Setup training and validation datasets."""
    logging.info("Setting up data...")
    
    dataset_name = data_config["dataset_name"]
    dataset_type = data_config.get("dataset_type", "pretokenized")
    on_disk = data_config.get("on_disk", False)
    
    if on_disk:
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name)
    
    assert "train" in dataset.keys(), "Dataset must contain a train split"
    train_dataset = dataset["train"]
    val_dataset = dataset.get("validation", None)

    if dataset_type != "native":
        logging.info(
            f"Using non-native '{dataset_type}' dataset type, applying transformation "
            "(this may take a while)"
        )
    
    train_dataset = transform_dataset(train_dataset, dataset_type, tokenizer)
    if val_dataset is not None:
        val_dataset = transform_dataset(val_dataset, dataset_type, tokenizer)

    return train_dataset, val_dataset


def main():
    """Main entry point for KD training."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        raise ValueError(
            "A config file is required. Please specify a config file using the "
            "--config argument."
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")
    
    # Check P2P availability (important for distributed training)
    if not torch.cuda.can_device_access_peer(0, 1):
        os.environ["NCCL_SHM_DISABLE"] = "1"
        logging.warning(
            "Detected that P2P via shared memory is not available. "
            "Setting NCCL_SHM_DISABLE to 1."
        )
        if not config["checkpointing"].get("hf_checkpoint", False):
            raise ValueError(
                "Running on a system configuration with bugged DCP checkpointing. "
                "Please set `checkpointing.hf_checkpoint` to `True` to use centralized "
                "HuggingFace checkpoints."
            )

    # Print config
    print("\nFinal config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"\n📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    # Initialize Ray
    init_ray()

    # Setup tokenizer (same for both student and teacher)
    tokenizer = get_tokenizer(config["student_policy"]["tokenizer"])

    teacher_model_name = config["teacher"]["model_name"]
    
    # Verify teacher uses compatible tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    if tokenizer.vocab_size != teacher_tokenizer.vocab_size:
        raise ValueError(
            f"Teacher and student tokenizers have different vocab sizes: "
            f"student={tokenizer.vocab_size}, teacher={teacher_tokenizer.vocab_size}. "
            f"Knowledge distillation requires compatible tokenizers with matching vocabularies."
        )
    logging.info(f"✓ Verified teacher tokenizer compatibility (vocab_size={tokenizer.vocab_size})")
    
    print(f"\n🔍 Student model: {config['student_policy']['model_name']}")
    print(f"🔍 Teacher model: {teacher_model_name}")
    print(f"🔍 KD alpha (distillation weight): {config['kd'].get('alpha', KD_DEFAULT_ALPHA)}")
    print(f"🔍 Temperature: {config['kd'].get('temperature', KD_DEFAULT_TEMPERATURE)}")

    # Setup data
    train_dataset, val_dataset = setup_data(tokenizer, config["data"])

    # Initialize KD trainer
    trainer = KDTrainer(config, tokenizer, train_dataset, val_dataset)
    
    # Run training with proper cleanup
    try:
        asyncio.run(trainer.train())
        print("\n✅ KD training completed successfully!")
    finally:
        # Ensure cleanup happens even if training fails
        print("\nCleaning up...")
        trainer.student_policy.shutdown()
        trainer.teacher_policy.shutdown()


if __name__ == "__main__":
    main()
