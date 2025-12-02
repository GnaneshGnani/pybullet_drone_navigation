import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from sensor_models import SensorFusionEncoder
from replay_buffers import OnPolicyReplayBuffer

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_camera = False, 
                 use_depth = False, use_lidar = False):
        super().__init__()
        self.encoder = SensorFusionEncoder(state_dim, use_camera, use_depth, use_lidar)

        self.layer_1 = nn.Linear(self.encoder.output_dim, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state, img = None, lidar = None):
        x = self.encoder(state, img, lidar)

        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        
        mean = torch.tanh(self.mean_layer(x)) * self.max_action
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min = -5.0, max = 2.0)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, use_camera = False, use_depth = False, use_lidar = False):
        super().__init__()
        self.encoder = SensorFusionEncoder(state_dim, use_camera, use_depth, use_lidar)

        self.layer_1 = nn.Linear(self.encoder.output_dim, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, 1)

    def forward(self, state, img = None, lidar = None):
        x = self.encoder(state, img, lidar)

        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        value = self.layer_3(x)
        return value

class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action, actor_lr, critic_lr,
                 use_camera = False, use_depth = False, use_lidar = False,
                 gamma = 0.99, gae_lambda = 0.95, clip_epsilon = 0.2,
                 n_epochs = 10, batch_size = 64):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_camera = use_camera
        self.use_depth = use_depth
        self.use_lidar = use_lidar

        self.actor = ActorNetwork(state_dim, action_dim, max_action,
                                   use_camera, use_depth, use_lidar).to(self.device)
        self.critic = CriticNetwork(state_dim, use_camera, use_depth, use_lidar).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)

        self.buffer = OnPolicyReplayBuffer(gamma, gae_lambda)
        
        self.max_action = max_action
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    @torch.no_grad()
    def get_action(self, state, img = None, lidar = None): # Update signature
        state_tensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
        
        # Handle optional sensors
        img_tensor = torch.tensor(img, dtype = torch.float32).unsqueeze(0).to(self.device) if img is not None else None
        lidar_tensor = torch.tensor(lidar, dtype = torch.float32).unsqueeze(0).to(self.device) if lidar is not None else None

        # FIX: Pass sensors
        dist = self.actor(state_tensor, img_tensor, lidar_tensor)

        action = dist.sample()
        action = torch.clamp(action, -self.max_action, self.max_action)

        log_prob = dist.log_prob(action).sum(dim = -1)
        
        # FIX: Pass sensors to critic
        value = self.critic(state_tensor, img_tensor, lidar_tensor)
        
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    @torch.no_grad()
    def get_deterministic_action(self, state, img = None, lidar = None):
        state_tensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
        img_t = torch.tensor(img, dtype = torch.float32).unsqueeze(0).to(self.device) if img is not None else None
        lidar_t = torch.tensor(lidar, dtype = torch.float32).unsqueeze(0).to(self.device) if lidar is not None else None

        dist = self.actor(state_tensor, img_t, lidar_t)
        action = dist.mean
        action = torch.clamp(action, -self.max_action, self.max_action)
        return action.squeeze(0).cpu().numpy()

    def store(self, state, img, lidar, action, reward, done, log_prob, value):
        self.buffer.store(state, img, lidar, action, reward, done, log_prob, value)

    def learn(self, last_value, done, target_kl = 0.015):
        if len(self.buffer) == 0:
            return None, None

        count = 0
        total_kl = 0
        total_entropy = 0
        total_actor_loss = 0
        total_critic_loss = 0
            
        # Compute advantages and returns
        advantages, returns = self.buffer.compute_advantages_and_returns(last_value, done)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize
        
        # Get the full batch of data
        states, imgs, lidars, actions, old_log_probs, old_values = self.buffer.get_batch()

        states = torch.tensor(states, dtype = torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype = torch.float32).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype = torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype = torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype = torch.float32).to(self.device)

        imgs_t = torch.tensor(imgs, dtype = torch.float32).to(self.device) if imgs is not None else None
        lidars_t = torch.tensor(lidars, dtype = torch.float32).to(self.device) if lidars is not None else None
        
        actor_losses = []
        critic_losses = []
        
        # Train for n_epochs
        indices = np.arange(len(self.buffer))
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, len(self.buffer), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start : end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                b_img = imgs_t[batch_indices] if imgs_t is not None else None
                b_lidar = lidars_t[batch_indices] if lidars_t is not None else None
                
                # --- Actor Loss ---
                dist = self.actor(batch_states, b_img, b_lidar)
                new_log_probs = dist.log_prob(batch_actions).sum(dim = -1)

                entropy = dist.entropy().mean()
                
                # --- KL Divergence Check ---
                with torch.no_grad():
                    # Calculate how much the policy changed
                    log_ratio = new_log_probs - batch_old_log_probs
                    # Approx KL formula: (r - 1) - log(r)
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()

                # If policy changed too much, stop immediately to prevent collapse
                if approx_kl > 1.5 * target_kl:
                    print(f"KL Spike ({approx_kl:.4f}) detected at epoch {epoch}. Stopping update to prevent collapse.")
                    self.buffer.clear()

                    # Return current metrics so we can see the spike in logs
                    return {
                        "actor_loss": total_actor_loss / max(1, count),
                        "critic_loss": total_critic_loss / max(1, count),
                        "entropy": total_entropy / max(1, count),
                        "kl": total_kl / max(1, count),
                        "stop_epoch": epoch
                    }
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # --- Critic Loss ---
                new_values = self.critic(batch_states, b_img, b_lidar).squeeze(-1)
                critic_loss = F.mse_loss(new_values, batch_returns)
                
                # --- Total Loss and Update ---
                # We can also add an entropy bonus to encourage exploration
                entropy_loss = dist.entropy().mean()
                total_loss = actor_loss + (0.5 * critic_loss) - (0.01 * entropy_loss)
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        self.buffer.clear()
        
        return {
            "actor_loss": total_actor_loss / max(1, count),
            "critic_loss": total_critic_loss / max(1, count),
            "entropy": total_entropy / max(1, count),
            "kl": total_kl / max(1, count)
        }
        
    def save_models(self, save_dir):
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))

    def load_models(self, load_dir):
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, "actor.pth"), map_location = self.device))
        self.critic.load_state_dict(torch.load(os.path.join(load_dir, "critic.pth"), map_location = self.device))