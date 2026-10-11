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

def train():
    # Detect next run ID (e.g., TD3_28)
    run_id = get_next_run_id(log_dir=os.path.join(current_dir, "tensorboard_logs"))
    print(f"--- Launching Training Run: {run_id} ---")

    # Instantiate the decentralized VecEnv natively
    env = AccelTrafficEnv()
    # Normalize rewards (not obs, since we do that manually) to stabilize critic training
    # clip_reward=10.0 prevents extreme outliers from destabilizing the Q-function
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    env = VecMonitor(env)

    # Initialize action noise
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Initialize the TD3 agent
    # We use MlpPolicy since our observations are flattened arrays
    model = TD3(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,       # Reduced from 0.001 for stability over 2M steps
        buffer_size=500000,          # Increased from 100k to retain more diverse experiences
        learning_starts=5000,        # More initial random exploration before learning
        batch_size=256,              # Larger batches for smoother gradient estimates
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,            # Changed from -1: one gradient update per env step (prevents divergence)
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[400, 300]),
        tensorboard_log="./tensorboard_logs/"
    )

    # Setup directories for versioned models & intermediate checkpoints
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoints_dir = os.path.join(models_dir, "checkpoints", run_id)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Save periodic checkpoint every 200,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=200000,
        save_path=checkpoints_dir,
        name_prefix=run_id
    )

    print(f"Starting TD3 direct acceleration training for {run_id} (Option 2 tuned rewards)...")
    # Train the agent
    model.learn(total_timesteps=1000000, log_interval=100, tb_log_name=run_id, callback=checkpoint_callback)
    print("Training finished.")

    # 1. Save versioned model to models/<run_id>.zip
    versioned_path = os.path.join(models_dir, run_id)
    model.save(versioned_path)
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
            print(f"Episode terminated early due to collision after {i+1} steps.")
            break
            
    print(f"Total Evaluation Reward: {total_reward}")

if __name__ == "__main__":
    train()
