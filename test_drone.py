import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import initialize_agent
from bullet_env import BulletNavigationEnv
from waypoint_manager import WaypointManager

def parse_args():
    parser = argparse.ArgumentParser(description = "Test PyBullet Drone")
    parser.add_argument("--episodes", type = int, default = 5)
    parser.add_argument("--max_steps", type = int, default = 1000)
    parser.add_argument("--algo", type = str, default = "sac", choices = ["sac", "ppo", "ddpg"])
    
    # Model Params (Must match training!)
    parser.add_argument("--actor_lr", type = float, default = 3e-4)
    parser.add_argument("--critic_lr", type = float, default = 3e-4)
    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--buffer_size", type = int, default = 100)
    parser.add_argument("--use_camera", action = "store_true")
    parser.add_argument("--use_depth", action = "store_true")
    parser.add_argument("--use_lidar", action = "store_true")
    parser.add_argument("--use_obstacles", action = "store_true")
    
    # Env Params
    parser.add_argument("--waypoint_threshold", type = float, default = 0.5)
    parser.add_argument("--waypoint_bonus", type = float, default = 100.0)
    parser.add_argument("--crash_penalty", type = float, default = -100.0)
    parser.add_argument("--timeout_penalty", type = float, default = -10.0)
    parser.add_argument("--step_reward", type = float, default = 0.1)
    parser.add_argument("--max_dist_from_target", type = float, default = 17.5)
    
    # Path to the specific experiment folder (e.g. ./models/run_sac)
    parser.add_argument("--model_path", type = str, required = True, help = "Path to experiment folder containing actor.pth")
    return parser.parse_args()

def run_one_episode(env, agent, max_steps, algo):
    obs, _ = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        state_vec = obs["state"]
        img_data = obs.get("image")
        lidar_data = obs.get("lidar")

        if img_data is not None:
            # Assuming Depth is the last channel
            depth_channel = img_data[-1] # Shape (64, 64)

            print("Depth")
            if np.random.random() < 0.02:
                plt.figure(figsize=(5, 4))
                # Use 'plasma' or 'magma' cmap which is great for depth (Yellow=Close/High, Blue=Far/Low)
                # vmin=0.0, vmax=1.0 ensures consistent coloring
                plt.imshow(depth_channel, cmap='plasma', vmin=0.0, vmax=1.0)
                plt.colorbar(label='Normalized Depth (0=Near, 1=Far)')
                plt.title(f"Depth View @ Step {step}")
                
                plt.show()
            
            # Center pixel
            center_val = depth_channel[32, 32]
            
            # Since we normalized by Far Plane (20.0m), convert back to read meters
            real_meters = center_val * 20.0 
            
            # if step % 10 == 0:
            #     print(f"Center Pixel Depth: {real_meters:.2f} meters")

        if hasattr(agent, 'get_deterministic_action'):
            action = agent.get_deterministic_action(state_vec, img = img_data, lidar = lidar_data)

        else:
            if algo ==  "ppo":
                action = agent.get_deterministic_action(state_vec, img = img_data, lidar = lidar_data)

            else:
                action = agent.get_action(state_vec, img = img_data, lidar = lidar_data)
        
        # print("Step:", step, "Action:", action)

        lidar_data = obs.get("lidar")
        if lidar_data is not None:
            min_dist = np.min(lidar_data)
            max_dist = np.max(lidar_data)
            
            # Only print if something is close (e.g., < 2.0m)
            # Max range is approx 5.0m based on your bullet_env.py
            if min_dist < 4.5: 
                # Find which angle has the closest object
                min_idx = np.argmin(lidar_data)
                angle = (min_idx / 360.0) * 360
                print(f"LIDAR DETECTED: Dist={min_dist:.2f}m @ {angle:.0f} deg")
            else:
                pass
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = next_obs
        episode_reward += reward

        print("Episode Reward:", episode_reward)
        print()
        
        # Slow down for visualization
        time.sleep(0.1) 
        
        if done:
            break
            
    return episode_reward, step

def main():
    args = parse_args()
    wpm = WaypointManager()
    waypoints = wpm.generate_random_walk_path()

    if args.use_obstacles:
        obstacles = wpm.generate_obstacles(num_obstacles = 1)

    action_limits = [1.0, 1.0, 1.0, 1.0]

    env = BulletNavigationEnv(
        waypoints = waypoints,
        obstacles = obstacles,
        use_camera = args.use_camera,
        use_depth = args.use_depth,
        use_lidar = args.use_lidar,
        use_obstacles = args.use_obstacles,
        waypoint_threshold = args.waypoint_threshold,
        waypoint_bonus = args.waypoint_bonus,
        crash_penalty = args.crash_penalty,
        timeout_penalty = args.timeout_penalty,
        step_reward = args.step_reward,
        max_dist_from_target = args.max_dist_from_target,
        action_limits = action_limits,
        gui = True,             
        show_waypoints = True   
    )

    state_dim = env.state_dim
    action_dim = env.action_space.shape[0]
    max_action = 1.0
    
    try:
        # Initialize fresh agent
        args.use_lidar = False
        args.use_depth = False
        agent = initialize_agent(args, state_dim, action_dim, max_action)
        
        # Load weights
        print(f"Loading models from {args.model_path}...")
        agent.load_models(args.model_path)

    except Exception as e:
        print(f"Error loading agent: {e}")
        return

    print(f"--- Starting Test ---")
    try:
        for episode in range(args.episodes):
            print(f"Episode {episode+1}")
            reward, length = run_one_episode(env, agent, args.max_steps, args.algo)
            print(f"Reward: {reward:.2f} | Length: {length}")

    except KeyboardInterrupt:
        print("Test interrupted.")
        
    finally:
        env.close()

if __name__ == "__main__":
    main()