from env_snake import SnakeEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import os
import torch
import sys
import signal

model = None  # Declare model globally to access it in save_model_on_exit

def save_model_on_exit(signal, frame):
    if model:  # Check if model is initialized before trying to save
        print("\nInterrupt received, saving model...")
        model.save("models/sac_snake_interrupted")
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers to catch interruptions
    signal.signal(signal.SIGINT, save_model_on_exit)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, save_model_on_exit)  # Handle termination signals

    # Verify CUDA / GPU
    print("CUDA available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))

    # Instantiate and validate the environment
    env = SnakeEnv(render_mode="human")
    check_env(env)

    # Create the SAC model on GPU
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-3,
        batch_size=1028,
        buffer_size=int(9e6),
        tau=0.18,
        gamma=0.5,
        ent_coef="auto",
        target_entropy="auto",
        train_freq=(1, "step"),
        gradient_steps=1,
        tensorboard_log="sac_snake_tensorboard/",
        device="cuda"
    )

    # Create output directories
    os.makedirs("models", exist_ok=True)

    print("Starting training...")
    model.learn(
        total_timesteps=10000,
        log_interval=4,
    )

    # Save the final model
    model.save("models/sac_snake_final")
    env.close()
