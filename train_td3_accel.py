import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import os
import sys

# Add current directory to path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from env.accel_traffic_env import AccelTrafficEnv

def get_next_run_id(log_dir="./tensorboard_logs", prefix="TD3"):
    """Auto-detect the next run number based on existing tensorboard log directories."""
    if not os.path.exists(log_dir):
        return f"{prefix}_1"
    existing = []
    for item in os.listdir(log_dir):
        if item.startswith(f"{prefix}_"):
            try:
                num = int(item.split("_")[1])
                existing.append(num)
            except ValueError:
                pass
    next_num = max(existing, default=0) + 1
    return f"{prefix}_{next_num}"

import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-tune", type=str, default="", help="Path to base model to fine-tune from")
    parser.add_argument("--timesteps", type=int, default=150000, help="Total timesteps")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gradient-steps", type=int, default=2, help="Gradient steps per env step")
    parsed_args, _ = parser.parse_known_args()

    # Detect next run ID (e.g., TD3_34)
    run_id = get_next_run_id(log_dir=os.path.join(current_dir, "tensorboard_logs"))
    print(f"--- Launching Training Run: {run_id} ---")

    # Instantiate the decentralized VecEnv natively
    raw_env = AccelTrafficEnv()

    # If fine-tuning and vecnormalize pkl exists, load running stats
    vecnorm_pkl = parsed_args.fine_tune.replace(".zip", "_vecnormalize.pkl") if parsed_args.fine_tune else ""
    if vecnorm_pkl and os.path.exists(vecnorm_pkl):
        print(f"Loading VecNormalize statistics from {vecnorm_pkl}")
        env = VecNormalize.load(vecnorm_pkl, raw_env)
        env.training = True
        env.norm_obs = False
        env.norm_reward = True
        env.clip_reward = 10.0
    else:
        env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    env = VecMonitor(env)

    # Initialize action noise (gentle noise during fine-tuning)
    n_actions = env.action_space.shape[-1]
    noise_sigma = 0.20 if parsed_args.fine_tune else 0.35
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma * np.ones(n_actions))

    if parsed_args.fine_tune and os.path.exists(parsed_args.fine_tune):
        print(f"Fine-tuning from base model: {parsed_args.fine_tune}")
        model = TD3.load(
            parsed_args.fine_tune,
            env=env,
            learning_rate=parsed_args.lr,
            action_noise=action_noise,
            tensorboard_log="./tensorboard_logs/"
        )
        model.gradient_steps = parsed_args.gradient_steps
        model.learning_starts = 0
    else:
        print("Training fresh TD3 policy from scratch...")
        model = TD3(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=parsed_args.lr if parsed_args.lr != 0.0001 else 0.0003,
            buffer_size=300000,          # 300k transitions buffer
            learning_starts=3000,        # Initial exploration
            batch_size=256,              # Smooth gradient updates
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=parsed_args.gradient_steps,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[400, 300]),
            tensorboard_log="./tensorboard_logs/"
        )

    # Setup directories for versioned models & intermediate checkpoints
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoints_dir = os.path.join(models_dir, "checkpoints", run_id)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Save periodic checkpoint every 50,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoints_dir,
        name_prefix=run_id
    )

    print(f"Starting TD3 direct acceleration training for {run_id} (Phase 4 Optimized)...")
    # Train the agent
    total_steps = parsed_args.timesteps
    model.learn(total_timesteps=total_steps, log_interval=100, tb_log_name=run_id, callback=checkpoint_callback)
    print("Training finished.")

    # 0. Analytical Zero-Drift Equilibrium Calibration
    import torch
    zeros_obs = torch.zeros((1, env.observation_space.shape[0]), dtype=torch.float32, device=model.device)
    with torch.no_grad():
        x = zeros_obs
        for i in range(len(model.actor.mu) - 2):
            x = model.actor.mu[i](x)
        last_linear_idx = len(model.actor.mu) - 2
        pre_bias = torch.matmul(x, model.actor.mu[last_linear_idx].weight.t())
        model.actor.mu[last_linear_idx].bias.copy_(-pre_bias.squeeze())
        calib_act = model.actor(zeros_obs).item()
        print(f"Equilibrium analytical calibration verified: a(0) = {calib_act:+.8f} m/s^2")

    # 1. Save versioned model to models/<run_id>.zip
    versioned_path = os.path.join(models_dir, run_id)
    model.save(versioned_path)
    env.save(os.path.join(models_dir, f"{run_id}_vecnormalize.pkl"))
    print(f"Versioned model saved as '{versioned_path}.zip'")

    # 2. Also update default active model td3_accel_agent.zip in root
    default_path = os.path.join(current_dir, "td3_accel_agent")
    model.save(default_path)
    print(f"Default model updated as '{default_path}.zip'")

    # Evaluate the trained agent
    print("\nEvaluating model...")
    obs = env.reset()
    total_reward = 0
    
    # Run a short evaluation episode
    for i in range(1000):
        # Predict the action (now expects to predict for 15 agents separately)
        actions, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(actions)
        total_reward += np.sum(rewards)
        
        if any(dones):
            print(f"Episode completed rollout after {i+1} steps.")
            break
            
    print(f"Total Evaluation Reward: {total_reward}")

if __name__ == "__main__":
    train()
