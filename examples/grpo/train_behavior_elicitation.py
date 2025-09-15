import verifiers as vf
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../environments/behavior_elicitation'))
from behavior_elicitation import ModelConfig, load_environment

"""
# Single server setup (memory-optimized for 12GB GPU):
# Terminal 1: Keep your current server running on port 8000
# Terminal 2: python examples/grpo/train_behavior_elicitation.py
"""

# Use single server for both training and target (memory-efficient)
target_model = ModelConfig(
    model_name="Qwen/Qwen2.5-0.5B",
    base_url="http://localhost:8000/v1",  # Same server as training
    api_key="dummy-key"
)

judge_model = ModelConfig(
    model_name="openai/gpt-4.1-mini", 
    base_url="https://openrouter.ai/api/v1",  # Reuse target server for judging
    api_key="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"
)

# Load environment with custom model configs
vf_env = load_environment(
    target_model=target_model,
    judge_model=judge_model,
    dataset_size=5  # Start small for testing
)

# Training model uses default vf-vllm server (port 8000)
model_name = "Qwen/Qwen2.5-0.5B"
# Disable FlashAttention to avoid dependency issues
model, tokenizer = vf.get_model_and_tokenizer(
    model_name, 
    model_kwargs={"attn_implementation": "eager"}
)

# Configure training args to disable distributed features
training_args = vf.grpo_defaults(run_name="behavior-elicitation-test")
training_args.ddp_backend = None  # Disable distributed training

trainer = vf.GRPOTrainer(
    env=vf_env,
    model=model,
    processing_class=tokenizer,
    args=training_args,
)

print("🚀 Starting PRBO training with dual servers!")
print(f"📊 Dataset size: {len(vf_env.get_dataset())} behaviors")
trainer.train()