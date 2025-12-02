import os
import time
import json
import torch
import argparse
import numpy as np

from clearml import Task
from collections import defaultdict, deque

from utils import initialize_agent
from bullet_env import BulletNavigationEnv
from waypoint_manager import WaypointManager

class DummyLogger:
    def report_scalar(self, title, series, value, iteration):
        pass

    def flush(self):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description = "Train PyBullet Drone")
    
    # Curriculum Flags
    parser.add_argument("--task", type = str, default = "hover", choices = ["hover", "square", "random"], 
                        help = "Curriculum Phase: hover (Phase 1), square (Phase 2), random (Phase 3)")
    parser.add_argument("--load_model", type = str, default = "", 
                        help = "Path to a PREVIOUS run folder (e.g., training_runs/run_sac_1) to load weights from.")
    
    # Action Limits
    parser.add_argument("--action_smoothing", type = float, default = 0.75, 
                        help = "Alpha for action smoothing (0.1=sluggish, 0.9=responsive).")
    parser.add_argument("--max_lin_vel_x", type = float, default = 1.0, 
                        help = "Max linear velocity in m/s (approx). X-direction")
    parser.add_argument("--max_lin_vel_y", type = float, default = 1.0, 
                        help = "Max linear velocity in m/s (approx). Y-direction")
    parser.add_argument("--max_lin_vel_z", type = float, default = 1.0, 
                        help = "Max linear velocity in m/s (approx). Z-direction")
    parser.add_argument("--max_yaw_rate", type = float, default = 1.0, 
                        help = "Max angular velocity in rad/s (approx).")
    
    # Training Params
    parser.add_argument("--episodes", type = int, default = 2000)
    parser.add_argument("--algo", type = str, default = "sac", choices = ["sac", "ppo", "ddpg"])
    parser.add_argument("--max_steps", type = int, default = 1000)
    parser.add_argument("--batch_size", type = int, default = 2048)
    parser.add_argument("--buffer_size", type = int, default = 50000)
    parser.add_argument("--actor_lr", type = float, default = 3e-4)
    parser.add_argument("--critic_lr", type = float, default = 1e-3)

    # PPO Specific
    parser.add_argument("--ppo_epochs", type = int, default = 10)
    parser.add_argument("--ppo_clip", type = float, default = 0.2)
    
    # Sensor Flags
    parser.add_argument("--use_camera", action = "store_true")
    parser.add_argument("--use_depth", action = "store_true")
    parser.add_argument("--use_lidar", action = "store_true")
    parser.add_argument("--use_obstacles", action = "store_true")
    
    # Reward Shaping
    parser.add_argument("--step_reward", type = float, default = 0.1)
    parser.add_argument("--waypoint_bonus", type = float, default = 100.0)
    parser.add_argument("--crash_penalty", type = float, default = -100.0)
    parser.add_argument("--timeout_penalty", type = float, default = -10.0)
    parser.add_argument("--waypoint_threshold", type = float, default = 0.25)
    parser.add_argument("--max_dist_from_target", type = float, default = 7.5)
    parser.add_argument("--episode_completion_reward", type = float, default = 100.0)

    # Logging
    parser.add_argument("--project_name", type = str, default = "PyBullet Training")
    parser.add_argument("--task_name", type = str, default = "Training_Run")
    parser.add_argument("--save_interval", type = int, default = 100)
    
    # Misc
    parser.add_argument("--headless", action = "store_true")
    parser.add_argument("--visualize", action = "store_true")
    parser.add_argument("--run_tag", type = str, default = "run")
    parser.add_argument("--save_dir", type = str, default = "./training_runs")
    parser.add_argument("--no_logging", action = "store_true", help = "Disable ClearML logging for debugging/local runs.")

    return parser.parse_args()

def run_one_episode(env, agent, max_steps, algo, gui = False):
    obs, _ = env.reset()
    episode_reward = 0
    loss_metrics = defaultdict(list)
    info = {"waypoints_reached": 0, "total_waypoints": 0}
    
    for step in range(max_steps):
        state_vec = obs["state"]
        img_data = obs.get("image")
        lidar_data = obs.get("lidar")

        if algo == "ppo":
           action, log_prob, value = agent.get_action(state_vec, img = img_data, lidar = lidar_data)
        else:
            action = agent.get_action(state_vec, img = img_data, lidar = lidar_data)
            log_prob, value = None, None

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if gui: time.sleep(0.01)
        
        # Store / Learn
        if algo == "ppo":
            agent.store(state_vec, img_data, lidar_data, action, reward, terminated, log_prob, value)

        else:
            agent.remember(state_vec, img_data, lidar_data, action, reward, terminated, 
                           next_obs["state"], next_obs.get("image"), next_obs.get("lidar"))
            
            losses = agent.learn()
            if losses and losses[0] is not None:
                if len(losses) == 3: # SAC
                    loss_metrics["actor_loss"].append(losses[0])
                    loss_metrics["critic_loss"].append(losses[1])
                    loss_metrics["alpha_loss"].append(losses[2])
                    loss_metrics["total_loss"].append(sum(losses))

                elif len(losses) == 2: # DDPG
                    loss_metrics["actor_loss"].append(losses[0])
                    loss_metrics["critic_loss"].append(losses[1])
                    loss_metrics["total_loss"].append(sum(losses))
        
        obs = next_obs
        episode_reward += reward
        
        if done: break
    
    # End of Episode PPO Update
    if algo == "ppo":
        if terminated: last_value = 0
        else: _, _, last_value = agent.get_action(obs["state"], img = obs.get("image"), lidar = obs.get("lidar"))
        
        if len(agent.buffer) >=  agent.batch_size:
            print("Training PPO......")
            ppo_metrics = agent.learn(last_value, terminated)

            if isinstance(ppo_metrics, dict):
                for key, value in ppo_metrics.items():
                    loss_metrics[key].append(value)
    
    avg_metrics = {k: np.mean(v) for k, v in loss_metrics.items() if len(v) > 0}
    return episode_reward, step, avg_metrics, info

def main():
    args = parse_args()
    
    base_dir = args.save_dir
    base_name = f"{args.run_tag}_{args.algo}_{args.task}"
    run_number = 1
    while os.path.exists(os.path.join(base_dir, f"{base_name}_{run_number}")):
        run_number += 1
    
    final_run_name = f"{base_name}_{run_number}"
    save_dir = os.path.join(base_dir, final_run_name)
    best_model_dir = os.path.join(save_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok = True)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent = 4)

    if not args.no_logging:
        print(f"Initializing ClearML: {args.project_name} | {args.task_name}")
        task = Task.init(project_name = args.project_name, task_name = final_run_name)
        task.connect(args) 
        logger = task.get_logger()

    else:
        print("Logging Disabled")
        logger = DummyLogger() # Using the dummy class so report_scalar calls don't crash

    wpm = WaypointManager()
    
    env = BulletNavigationEnv(
        waypoints = [],
        obstacles = [],
        use_camera = args.use_camera,
        use_depth = args.use_depth,
        use_lidar = args.use_lidar,
        use_obstacles = args.use_obstacles,
        waypoint_threshold = args.waypoint_threshold,
        waypoint_bonus = args.waypoint_bonus,
        crash_penalty = args.crash_penalty,
        timeout_penalty = args.timeout_penalty,
        step_reward = args.step_reward,
        episode_completion_reward = args.episode_completion_reward,
        max_dist_from_target = args.max_dist_from_target,
        action_smoothing = args.action_smoothing,
        action_limits = [args.max_lin_vel_x, args.max_lin_vel_y, args.max_lin_vel_z, args.max_yaw_rate],
        gui = not args.headless,
        show_waypoints = args.visualize
    )

    max_action = 1.0
    state_dim = env.state_dim
    action_dim = env.action_space.shape[0]
    agent = initialize_agent(args, state_dim, action_dim, max_action)

    # Transfer Learning
    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"Loading weights from: {args.load_model}")
            try:
                agent.load_models(args.load_model)
                print("Models loaded successfully. Continuing training...")

            except Exception as e:
                print(f"Error loading models: {e}")
                return
            
        else:
            print(f"Path {args.load_model} does not exist! Starting from scratch.")
    
    schedulers = []
    schedulers.extend([
        torch.optim.lr_scheduler.StepLR(agent.actor_optimizer, step_size = 500, gamma = 0.5),
        torch.optim.lr_scheduler.StepLR(agent.critic_optimizer, step_size = 500, gamma = 0.5),
    ])

    if args.algo == "sac":
        schedulers.append(
            torch.optim.lr_scheduler.StepLR(agent.alpha_optimizer, step_size = 500, gamma = 0.5)
        )

    print(f"Starting Training: {final_run_name}")
    reward_window = deque(maxlen = 50)
    best_reward = -float('inf')

    try:
        for episode in range(args.episodes):
            print(f"Configuring Task: {args.task.upper()}")
            if args.task == "hover":
                waypoints = wpm.generate_hover_target(altitude = 1.0)
            
            elif args.task == "square":
                waypoints = wpm.generate_square_path(side_length = 2.0, altitude = 1.5)

            elif args.task == "random":
                waypoints = wpm.generate_random_walk_path()
                if args.use_obstacles:
                    obstacles = wpm.generate_obstacles(num_obstacles = 6)
                    env.obstacles = obstacles
            
            env.waypoints = waypoints
            
            reward, length, metrics, info = run_one_episode(env, agent, args.max_steps, args.algo, not args.headless)

            if (length + 1) == args.max_steps: print("Max Steps!")

            reward_window.append(reward)
            avg_reward = np.mean(reward_window)

            logger.report_scalar(title = "Training", series = "Avg Step Reward", value = float(reward) / (length + 1), iteration = episode)
            logger.report_scalar(title = "Training", series = "Avg Total Reward (50 ep)", value = float(avg_reward), iteration = episode)
            logger.report_scalar(title = "Training", series = "Episode Length", value = int(length), iteration = episode)
            
            wps_reached = info.get("waypoints_reached", 0)
            total_wps = info.get("total_waypoints", 1)
            
            logger.report_scalar(title = "Performance", series = "Waypoints Reached", value = int(wps_reached), iteration = episode)
            logger.report_scalar(title = "Performance", series = "Completion %", value = float((wps_reached / total_wps) * 100), iteration = episode)
            
            dist_val = info.get("dist_to_current_target", 0)
            logger.report_scalar(title = "Debug", series = "Final Dist to Target", value = float(dist_val), iteration = episode)

            if len(metrics) > 0:
                for key, val in metrics.items():
                    logger.report_scalar(title = "Loss", series = key, value = float(val), iteration = episode)
                
                for scheduler in schedulers:
                    scheduler.step()

                logger.report_scalar(title = "Debug", series = "Actor Learning Rate", value = schedulers[0].get_last_lr()[0], iteration = episode)
                logger.report_scalar(title = "Debug", series = "Critic Learning Rate", value = schedulers[1].get_last_lr()[0], iteration = episode)
                if args.algo == "sac":
                    logger.report_scalar(title = "Debug", series = "Alpha Learning Rate", value = schedulers[2].get_last_lr()[0], iteration = episode)

            print(f"Ep {episode} | Reward: {reward:.2f} | Avg: {avg_reward:.2f} | WPs: {info['waypoints_reached']}")
            
            logger.flush()

            if episode % args.save_interval == 0:
                agent.save_models(save_dir)
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                print(f"New Best Model Found! Reward: {best_reward:.2f}. Saving...")
                agent.save_models(best_model_dir) 

        print(f"Training Complete. Saving final model to {save_dir}...")
        agent.save_models(save_dir)
                
    except KeyboardInterrupt:
        print("Training interrupted.")
        agent.save_models(save_dir)

    finally:
        env.close()

if __name__ == "__main__":
    main()