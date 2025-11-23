import time
import argparse

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
    parser.add_argument("--crash_penalty", type = float, default = 50.0)
    parser.add_argument("--timeout_penalty", type = float, default = 10.0)
    parser.add_argument("--per_step_penalty", type = float, default = -0.1)
    parser.add_argument("--max_dist_from_target", type = float, default = 10.0)
    
    # Path to the specific experiment folder (e.g. ./models/run_sac)
    parser.add_argument("--model_path", type = str, required = True, help = "Path to experiment folder containing actor.pth")
    return parser.parse_args()

def run_one_episode(env, agent, max_steps, algo):
    obs, _ = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        state_vec = obs["state"]
        img_data = obs["image"]
        lidar_data = obs["lidar"]

        if hasattr(agent, 'get_deterministic_action'):
            action = agent.get_deterministic_action(state_vec, img = img_data, lidar = lidar_data)

        else:
            if algo ==  "ppo":
                action = agent.get_deterministic_action(state_vec, img = img_data, lidar = lidar_data)

            else:
                action = agent.get_action(state_vec, img = img_data, lidar = lidar_data)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = next_obs
        episode_reward += reward
        
        # Slow down for visualization
        time.sleep(1/60) 
        
        if done:
            break
            
    return episode_reward, step

def main():
    args = parse_args()
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
        gui = True,             
        show_waypoints = True   
    )

    state_dim = env.state_dim
    action_dim = 4
    max_action = 1.0
    
    try:
        # Initialize fresh agent
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