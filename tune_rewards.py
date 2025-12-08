import numpy as np
from bullet_env import BulletNavigationEnv
from waypoint_manager import WaypointManager

def run_policy(env, policy_type="hover", steps=500):
    obs, _ = env.reset()
    total_reward = 0
    
    # Track reward components to see if any explode
    min_step_reward = float('inf')
    max_step_reward = -float('inf')
    
    print(f"\n--- Testing Policy: {policy_type.upper()} ---")

    for t in range(steps):
        if policy_type == "hover":
            # Command 0 velocity (stay still)
            action = np.zeros(4) 
        elif policy_type == "random":
            # Random actions
            action = env.action_space.sample()
            
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        min_step_reward = min(min_step_reward, reward)
        max_step_reward = max(max_step_reward, reward)

        # Optional: Print the first few steps to verify logic
        if t < 5:
            dist = info['dist_to_current_target']
            print(f"Step {t}: Reward={reward:.4f} | Dist={dist:.4f}")

        if terminated or truncated:
            break
            
    avg_step_reward = (total_reward + 100) / (t + 1)
    
    print(f"Result: {t+1} Steps")
    print(f"Total Episode Reward: {total_reward:.2f}")
    print(f"Avg Reward per Step:  {avg_step_reward:.4f}")
    print(f"Min Step Reward:      {min_step_reward:.4f}")
    print(f"Max Step Reward:      {max_step_reward:.4f}")
    
    return total_reward

def main():
    # Setup simple environment
    wpm = WaypointManager()
    waypoints = wpm.generate_hover_target(altitude=1.5)
    
    # Note: Using obstacles=False to isolate flight dynamics first
    env = BulletNavigationEnv(
        waypoints = waypoints,
        gui = False, # Set True if you want to watch
        use_obstacles = False,
        use_lidar = True,
        waypoint_threshold = 0.25,
        waypoint_bonus = 100,
        crash_penalty = -100,
        timeout_penalty = -10,
        step_reward = 0.1,
        episode_completion_reward = 100,
        max_dist_from_target = 7.5,
        action_smoothing = 0.75,
        action_limits = [1, 1, 1, 1],
        hardcoded_yaw = False,
        max_steps = 1000
    )

    # 1. Test Hover (Should generally be stable, reward shouldn't be massively negative)
    # Ideally, this should be close to 0 or positive if close to target.
    print(">>> TEST 1: Hover Policy (Action = 0)")
    run_policy(env, policy_type="hover", steps=1000)

    # 2. Test Random (Should be negative due to instability, but not -10,000)
    print("\n>>> TEST 2: Random Policy (Action = Random)")
    run_policy(env, policy_type="random", steps=1000)
    
    env.close()

if __name__ == "__main__":
    main()