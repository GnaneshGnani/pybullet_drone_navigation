from ppo import PPOAgent
from sac import SACAgent
from ddpg import DDPGAgent

def initialize_agent(args, state_dim, action_dim, max_action):
    if args.algo == "ddpg":
        return DDPGAgent(
            state_dim, action_dim, max_action,
            actor_lr = args.actor_lr, critic_lr = args.critic_lr,
            buffer_size = args.buffer_size, batch_size = args.batch_size,
            use_camera = args.use_camera, use_depth = args.use_depth, 
            use_lidar = args.use_lidar
        )
    
    elif args.algo == "ppo":
        return PPOAgent(
            state_dim, action_dim, max_action,
            actor_lr = args.actor_lr, critic_lr = args.critic_lr,
            n_epochs = args.ppo_epochs, batch_size = args.batch_size,
            use_camera = args.use_camera, use_depth = args.use_depth,
            use_lidar = args.use_lidar
        )
    
    elif args.algo == "sac":
        return SACAgent(
            state_dim, action_dim, max_action,
            actor_lr = args.actor_lr, critic_lr = args.critic_lr,
            buffer_size = args.buffer_size, batch_size = args.batch_size,
            use_camera = args.use_camera, use_depth = args.use_depth,
            use_lidar = args.use_lidar
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")