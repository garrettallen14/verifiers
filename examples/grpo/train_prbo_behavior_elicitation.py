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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../environments/behavior_elicitation'))

from behavior_elicitation import ModelConfig, load_environment

# Model configuration for GRPO training
model_name = "willcb/Qwen3-8B"  # Model to train with GRPO

# vLLM endpoint configurations (for PRBO environment)
# Using gpt-oss-20b as both judge and target model via vLLM
target_model_config = ModelConfig(
    model_name="gpt-oss-20b",
    base_url="http://localhost:8000/v1",  # gpt-oss-20b vLLM server
    api_key="EMPTY"
)

judge_model_config = ModelConfig(
    model_name="gpt-oss-20b", 
    base_url="http://localhost:8000/v1",  # Same endpoint for simplicity
    api_key="EMPTY"
)

print("🚀 Initializing PRBO Behavior Elicitation Environment...")
print(f"Training model: {model_name} (GRPO training target)")
print(f"Target model: {target_model_config.model_name} at {target_model_config.base_url} (PRBO environment)")
print(f"Judge model: {judge_model_config.model_name} at {judge_model_config.base_url} (PRBO environment)")
print("⚠️  Note: Qwen3 models may have chat template issues. Consider using Qwen2.5 if problems occur.")

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
trainer = vf.GRPOTrainer(
    env=vf_env,
    model=model,
    processing_class=tokenizer,
    args=vf.grpo_defaults(
        run_name="prbo-qwen3-4b-behavior-elicitation",
        # PRBO-specific hyperparameters for Qwen3-4B
        learning_rate=2e-6,  # Conservative LR for smaller model
        per_device_train_batch_size=4,  # Larger batch for 4B model
        gradient_accumulation_steps=4,  # Maintain effective batch size
        num_generations=16,  # Group size for GRPO
        max_steps=500,  # Total training steps
        warmup_steps=20,  # Warm up for stability
        logging_steps=5,  # Frequent logging for monitoring
        save_steps=50,  # Save checkpoints regularly
        eval_steps=25,  # Regular evaluation
        # GRPO-specific
        temperature=0.8,  # Higher exploration for attack generation
        beta=0.001,  # KL penalty (conservative)
        max_tokens=64,  # Match PRBO response length
        # Stability settings for Qwen3
        max_grad_norm=0.01,  # Aggressive clipping
        gradient_checkpointing=True,  # Memory efficiency
    ),
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
