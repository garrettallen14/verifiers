import verifiers as vf

"""
# install
vf-install behavior-elicitation (-p /path/to/environments)

# quick eval
vf-eval behavior-elicitation (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-0.5B \
    --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_behavior_elicitation.py
"""

model_name = "Qwen/Qwen2.5-0.5B"
vf_env = vf.load_environment(env_id="behavior-elicitation")
model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(
    env=vf_env,
    model=model,
    processing_class=tokenizer,
    args=vf.grpo_defaults(run_name="behavior-elicitation"),
)
trainer.train()