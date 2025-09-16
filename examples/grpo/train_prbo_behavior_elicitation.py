#!/usr/bin/env python3
"""
GRPO Training Script for PRBO Behavior Elicitation Environment

This script trains a model using GRPO with the PRBO (Policy Regularized Behavior Optimization) 
environment for behavior elicitation research.

Usage:
    # 1. Start vLLM inference server for target model:
    CUDA_VISIBLE_DEVICES=0 vf-vllm --model gpt-oss-20b \
        --enforce-eager --disable-log-requests --port 8000

    # 2. Start vLLM inference server for judge model (optional, can use same):
    CUDA_VISIBLE_DEVICES=1 vf-vllm --model gpt-oss-20b \
        --enforce-eager --disable-log-requests --port 8001

    # 3. Train the model:
    CUDA_VISIBLE_DEVICES=2 accelerate launch --num-processes 1 \
        --config-file configs/zero3.yaml examples/grpo/train_prbo_behavior_elicitation.py

Environment Setup:
    The PRBO environment requires two model endpoints:
    - Target model: Generates responses to attack attempts
    - Judge model: Scores prompts and responses for harmfulness
    
    Both can be the same model on the same endpoint for simplicity.
"""

import verifiers as vf
import sys
import os

# Add the behavior elicitation environment to the path
sys.path.append('/workspace/verifiers/environments/behavior_elicitation')

from behavior_elicitation import ModelConfig, load_environment

# Model configuration for GRPO training
model_name = "willcb/Qwen3-8B"  # Model to train with GRPO

# vLLM endpoint configurations (for PRBO environment)
# Using openai/gpt-oss-20b as both judge and target model via vLLM
target_model_config = ModelConfig(
    model_name="openai/gpt-oss-20b",
    base_url="http://localhost:8000/v1",  # gpt-oss-20b vLLM server
    api_key="EMPTY"
)

judge_model_config = ModelConfig(
    model_name="openai/gpt-oss-20b", 
    base_url="http://localhost:8000/v1",  # Same endpoint for simplicity
    api_key="EMPTY"
)

print("🚀 Initializing PRBO Behavior Elicitation Environment...")
print(f"Training model: {model_name} (GRPO training target)")
print(f"Target model: {target_model_config.model_name} at {target_model_config.base_url} (PRBO environment)")
print(f"Judge model: {judge_model_config.model_name} at {judge_model_config.base_url} (PRBO environment)")

# Load the PRBO environment
vf_env = load_environment(
    target_model=target_model_config,
    judge_model=judge_model_config,
    dataset_size=100  # Start with 100 samples for faster training
)

print(f"✅ Environment loaded with {len(vf_env.dataset)} behavior examples")

# Load model and tokenizer for training
print(f"🔧 Loading model and tokenizer: {model_name}")
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Configure GRPO trainer
print("⚙️ Setting up GRPO trainer...")
args = vf.grpo_defaults(run_name="prbo-no-vllm")

# 2-GPU optimized settings
args.beta = 0.0  # No reference model  
args.max_steps = 5
args.per_device_train_batch_size = 1
args.gradient_accumulation_steps = 2
args.num_generations = 2
args.learning_rate = 2e-6
args.max_tokens = 64

trainer = vf.GRPOTrainer(
    env=vf_env,
    model=model,
    processing_class=tokenizer,
    args=args,
)

print("🎯 Starting PRBO-GRPO training...")
print("=" * 60)
print("TRAINING OBJECTIVES:")
print("• Generate effective attack prompts for behavior elicitation")
print("• Maximize PRBO reward: attack_score + (log p_M - log q) + response_score")
print("• Learn to elicit harmful behaviors while maintaining naturalness")
print("=" * 60)

# Start training
trainer.train()

print("🎉 Training completed!")
print("Check the logs and saved checkpoints for results.")
