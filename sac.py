import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from replay_buffers import OffPolicyReplayBuffer
from sensor_models import SensorFusionEncoder

LOG_STD_MIN = -5
LOG_STD_MAX = 2

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_camera = False, use_depth = False, use_lidar = False):
        super().__init__()
        self.encoder = SensorFusionEncoder(state_dim, use_camera, use_depth, use_lidar)
        self.layer_1 = nn.Linear(self.encoder.output_dim, 256)
        self.layer_2 = nn.Linear(256, 256)

        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state, img = None, lidar = None):
        x = self.encoder(state, img, lidar)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, state, img = None, lidar = None):
        mean, log_std = self.forward(state, img, lidar)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        x_t = dist.rsample() # Reparameterization
        y_t = torch.tanh(x_t)

        action = y_t * self.max_action

        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim = -1, keepdim = True)

        return action, log_prob

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, use_camera = False, use_depth = False, use_lidar = False):
        super().__init__()
        self.encoder = SensorFusionEncoder(state_dim, use_camera, use_depth, use_lidar)
        
        # Q1
        self.q1_layer_1 = nn.Linear(self.encoder.output_dim + action_dim, 256)
        self.q1_layer_2 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, 1)
        
        # Q2
        self.q2_layer_1 = nn.Linear(self.encoder.output_dim + action_dim, 256)
        self.q2_layer_2 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, 1)

    def forward(self, state, action, img = None, lidar = None):
        x = self.encoder(state, img, lidar)
        x = torch.cat([x, action], dim = -1)
        
        q1 = F.relu(self.q1_layer_1(x))
        q1 = F.relu(self.q1_layer_2(q1))
        q1 = self.q1_out(q1)
        
        q2 = F.relu(self.q2_layer_1(x))
        q2 = F.relu(self.q2_layer_2(q2))
        q2 = self.q2_out(q2)
        
        return q1, q2


class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, actor_lr, critic_lr,
                 buffer_size, batch_size, use_camera, use_depth, use_lidar, 
                 gamma = 0.99, tau = 0.005, alpha = 0.2):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_camera = use_camera
        self.use_depth = use_depth
        self.use_lidar = use_lidar
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.batch_size = batch_size
        
        # Actor
        self.actor = ActorNetwork(state_dim, action_dim, max_action, 
                                   use_camera, use_depth, use_lidar).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        
        # Critic (Twin Q-networks)
        self.critic = CriticNetwork(state_dim, action_dim, 
                                     use_camera, use_depth, use_lidar).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim,
                                            use_camera, use_depth, use_lidar).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)
        
        # Automatic entropy tuning
        self.target_entropy = -action_dim  # Heuristic: -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad = True, device = self.device)
        self.alpha = alpha
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = actor_lr)
        
        self.buffer = OffPolicyReplayBuffer(buffer_size)

    def soft_update_target_networks(self):
        for target_param, param in zip(self.critic_target.parameters(), 
                                        self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    @torch.no_grad()
    def get_action(self, state, img = None, lidar = None):
        state_t = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
        img_t = torch.tensor(img, dtype = torch.float32).unsqueeze(0).to(self.device) if img is not None else None
        lidar_t = torch.tensor(lidar, dtype = torch.float32).unsqueeze(0).to(self.device) if lidar is not None else None
        
        self.actor.eval()
        action, _ = self.actor.sample(state_t, img_t, lidar_t)
        self.actor.train()
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def get_deterministic_action(self, state, img = None, lidar = None):
        state_t = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
        img_t = torch.tensor(img, dtype = torch.float32).unsqueeze(0).to(self.device) if img is not None else None
        lidar_t = torch.tensor(lidar, dtype = torch.float32).unsqueeze(0).to(self.device) if lidar is not None else None
        
        mean, _ = self.actor(state_t, img_t, lidar_t)
        action = torch.tanh(mean) * self.max_action
        return action.squeeze(0).cpu().numpy()

    def remember(self, state, img, lidar, action, reward, done, next_state, next_img, next_lidar):
        self.buffer.remember(state, img, lidar, action, reward, done, next_state, next_img, next_lidar)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None, None, None # actor, critic, alpha losses
        
        states, imgs, lidars, actions, rewards, next_states, next_imgs, next_lidars, dones = self.buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype = torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype = torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype = torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype = torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype = torch.float32).unsqueeze(1).to(self.device)

        imgs_t = torch.tensor(imgs, dtype = torch.float32).to(self.device) if imgs is not None else None
        lidars_t = torch.tensor(lidars, dtype = torch.float32).to(self.device) if lidars is not None else None

        next_imgs_t = torch.tensor(next_imgs, dtype = torch.float32).to(self.device) if next_imgs is not None else None
        next_lidars_t = torch.tensor(next_lidars, dtype = torch.float32).to(self.device) if next_lidars is not None else None

        # --- Critic Loss ---
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states, next_imgs_t, next_lidars_t)
            
            q1_target, q2_target = self.critic_target(next_states, next_actions, next_imgs_t, next_lidars_t)
            q_target = torch.min(q1_target, q2_target)
            
            td_target = rewards + (1 - dones) * self.gamma * (q_target - self.alpha * next_log_prob)

        td_target = td_target.detach()
        current_q1, current_q2 = self.critic(states, actions, imgs_t, lidars_t)
        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm = 1.0)
        self.critic_optimizer.step()

        # --- Actor Loss ---
        pi, log_pi = self.actor.sample(states, imgs_t, lidars_t)
        q1_pi, q2_pi = self.critic(states, pi, imgs_t, lidars_t)
        q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.alpha * log_pi - q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm = 1.0)
        self.actor_optimizer.step()

        # --- Alpha (Temperature) Loss ---
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item() # Update alpha value
        
        # --- Update Target Networks ---
        self.soft_update_target_networks()
        
        return actor_loss.item(), critic_loss.item(), alpha_loss.item()
        
    def save_models(self, save_dir):
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))

    def load_models(self, load_dir):
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, "actor.pth"), map_location = self.device))
        self.critic.load_state_dict(torch.load(os.path.join(load_dir, "critic.pth"), map_location = self.device))
        
        self.critic_target.load_state_dict(self.critic.state_dict())