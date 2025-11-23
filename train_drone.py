import os
import time
import json
import argparse
import numpy as np

from clearml import Task
from collections import defaultdict, deque

from utils import initialize_agent
from bullet_env import BulletNavigationEnv
from waypoint_manager import WaypointManager

def parse_args():
    parser = argparse.ArgumentParser(description = "Train PyBullet Drone")
    
    # Training Params
    parser.add_argument("--episodes", type = int, default = 2000)
    parser.add_argument("--curriculum_training", type = int, default = 500)
    parser.add_argument("--algo", type = str, default = "sac", choices = ["sac", "ppo", "ddpg"], help = "RL Algorithm")
    parser.add_argument("--max_steps", type = int, default = 5000)
    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--buffer_size", type = int, default = 20000, help = "Lower this if RAM fills up due to images")
    parser.add_argument("--actor_lr", type = float, default = 3e-4)
    parser.add_argument("--critic_lr", type = float, default = 3e-4)

    # PPO Specific
    parser.add_argument("--ppo_epochs", type = int, default = 10, help = "Number of epochs for PPO update.")
    parser.add_argument("--ppo_clip", type = float, default = 0.2, help = "PPO clip epsilon.")
    
    # Sensor Flags
    parser.add_argument("--use_camera", action = "store_true", help = "Enable RGB Camera")
    parser.add_argument("--use_depth", action = "store_true", help = "Enable Depth Camera")
    parser.add_argument("--use_lidar", action = "store_true", help = "Enable Lidar")
    parser.add_argument("--use_obstacles", action = "store_true", help = "Spawn random cubes")
    
    # Reward Shaping (Your original args)
    parser.add_argument("--waypoint_bonus", type = float, default = 100.0)
    parser.add_argument("--crash_penalty", type = float, default = 50.0)
    parser.add_argument("--timeout_penalty", type = float, default = 10.0)
    parser.add_argument("--per_step_penalty", type = float, default = -0.1)
    parser.add_argument("--waypoint_threshold", type = float, default = 0.5)
    parser.add_argument("--max_dist_from_target", type = float, default = 10.0)

    # Logging
    parser.add_argument("--project_name", type = str, default = "PyBullet Training", help = "ClearML project name.")
    parser.add_argument("--task_name", type = str, default = "Training_Run", help = "ClearML task name.")
    parser.add_argument("--save_interval", type = int, default = 100, help = "Save models every N episodes.")
    
    # Misc
    parser.add_argument("--headless", action = "store_true", help = "Run PyBullet in headless mode.")
    parser.add_argument("--visualize", action = "store_true", help = "Show Waypoints")
    parser.add_argument("--run_tag", type = str, default = "run")
    parser.add_argument("--save_dir", type = str, default = "./training_runs")

    return parser.parse_args()

def run_one_episode(env, agent, max_steps, algo, gui = False):
    obs, _ = env.reset()
    episode_reward = 0

    loss_metrics = defaultdict(list)
    info = {"waypoints_reached": 0, "total_waypoints": 0, "final_dist_to_target": 0}
    
    for step in range(max_steps):
        state_vec = obs["state"]
        img_data = obs["image"]
        lidar_data = obs["lidar"]

        if algo == "ppo":
           action, log_prob, value = agent.get_action(state_vec, img = img_data, lidar = lidar_data)
        else:
            # SAC/DDPG
            action = agent.get_action(state_vec, img = img_data, lidar = lidar_data)
            log_prob, value = None, None

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if gui:
            time.sleep(0.05)
        
        next_state = next_obs["state"]
        next_img = next_obs["image"]
        next_lidar = next_obs["lidar"]
        
        if algo == "ppo":
            agent.store(state_vec, img_data, lidar_data, action, reward, terminated, log_prob, value)
        else:
            agent.remember(state_vec, img_data, lidar_data, 
                        action, reward, terminated, 
                        next_state, next_img, next_lidar)
            
            losses = agent.learn()
            if losses and losses[0] is not None:
                if len(losses) == 3: # SAC: actor, critic, alpha
                    loss_metrics["actor_loss"].append(losses[0])
                    loss_metrics["critic_loss"].append(losses[1])
                    loss_metrics["alpha_loss"].append(losses[2])
                    loss_metrics["total_loss"].append(sum(losses))

                elif len(losses) == 2: # DDPG: actor, critic
                    loss_metrics["actor_loss"].append(losses[0])
                    loss_metrics["critic_loss"].append(losses[1])
                    loss_metrics["total_loss"].append(sum(losses))
        
        obs = next_obs
        episode_reward += reward
        
        if done:
            break
    
    if algo == "ppo":
        if terminated:
            last_value = 0  # Episode ended due to crash/completion
        else:
            # Either truncated or still running - bootstrap from current state
            _, _, last_value = agent.get_action(obs["state"], img = obs["image"], lidar = obs["lidar"])
        
        if len(agent.buffer) >= agent.batch_size:
            ppo_losses = agent.learn(last_value, terminated)
            
            if ppo_losses and ppo_losses[0] is not None:
                # PPO returns (actor_loss, critic_loss)
                loss_metrics["actor_loss"].append(ppo_losses[0])
                loss_metrics["critic_loss"].append(ppo_losses[1])
                loss_metrics["total_loss"].append(sum(ppo_losses))
    
    avg_metrics = {k: np.mean(v) for k, v in loss_metrics.items() if len(v) > 0}
            
    return episode_reward, step, avg_metrics, info

def main():
    args = parse_args()
    
    base_dir = "training_runs"
    base_name = f"{args.run_tag}_{args.algo}"
    run_number = 1
    
    while os.path.exists(os.path.join(base_dir, f"{base_name}_{run_number}")):
        run_number += 1
    
    final_run_name = f"{base_name}_{run_number}"
    save_dir = os.path.join(base_dir, final_run_name)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok = True)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent = 4)
        print(f"Configuration saved to {save_dir}/args.json")
    
    task = Task.init(
        project_name = args.project_name, 
        task_name = f"{args.task_name}_{final_run_name}"
    )

    task.connect(args) 
    logger = task.get_logger()

    wpm = WaypointManager()
    waypoints = wpm.generate_square_path()

    env = BulletNavigationEnv(
        waypoints = waypoints,
        use_camera = args.use_camera,
        use_depth = args.use_depth,
        use_lidar = args.use_lidar,
        use_obstacles = args.use_obstacles,
        waypoint_threshold = args.waypoint_threshold,
        waypoint_bonus = args.waypoint_bonus,
        crash_penalty = args.crash_penalty,
        timeout_penalty = args.timeout_penalty,
        per_step_penalty = args.per_step_penalty,
        max_dist_from_target = args.max_dist_from_target,
        gui = not args.headless,
        show_waypoints = args.visualize
    )

    # Initialize Agent
    state_dim = env.state_dim
    action_dim = 4
    max_action = 1.0
    
    agent = initialize_agent(args, state_dim, action_dim, max_action)

    print(f"--- Starting Training: {args.run_tag} ---")
    print(f"Sensors: Cam = {args.use_camera}, Depth = {args.use_depth}, Lidar = {args.use_lidar}")

    reward_window = deque(maxlen = 50)
    length_window = deque(maxlen = 50)

    try:
        for episode in range(args.episodes):
            if episode == args.curriculum_training:
                print("--- Switching to Random Walk Waypoints ---")
            
            if episode >= args.curriculum_training:
                env.waypoints = wpm.generate_random_walk_path()

            print("Episode:", episode)    
            reward, length, metrics, info = run_one_episode(env, agent, args.max_steps, args.algo, not args.headless)

            reward_window.append(reward)
            length_window.append(length)
            avg_reward = np.mean(reward_window)
            avg_len = np.mean(length_window)

            logger.report_scalar(title = "Training", series = "Reward", value = float(reward), iteration = episode)
            logger.report_scalar(title = "Training", series = "Avg Reward (50 ep)", value = float(avg_reward), iteration = episode)
            logger.report_scalar(title = "Training", series = "Episode Length", value = int(length), iteration = episode)
            
            wps_reached = info.get("waypoints_reached", 0)
            total_wps = info.get("total_waypoints", 1)
            logger.report_scalar(title = "Performance", series = "Waypoints Reached", value = int(wps_reached), iteration = episode)
            logger.report_scalar(title = "Performance", series = "Completion %", value = float((wps_reached / total_wps) * 100), iteration = episode)
            
            dist_val = info.get("dist_to_current_target", 0)
            logger.report_scalar(title = "Debug", series = "Final Dist to Target", value = float(dist_val), iteration = episode)

            for key, val in metrics.items():
                logger.report_scalar(title = "Loss", series = key, value = float(val), iteration = episode)

            # Ensure data is sent immediately
            logger.flush()

            print(f"Ep {episode} | Reward: {reward:.2f} | Len: {length}")
        
        print(f"Training Complete. Saving final model to {save_dir}...")
        agent.save_models(save_dir)
                
    except KeyboardInterrupt:
        print("Training interrupted.")

    finally:
        env.close()

if __name__ == "__main__":
    main()