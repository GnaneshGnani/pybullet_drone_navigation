import os 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sensor_models import SensorFusionEncoder
from replay_buffers import OffPolicyReplayBuffer

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_camera = False, use_depth = False, use_lidar = False):
        super().__init__()
        self.encoder = SensorFusionEncoder(state_dim, use_camera, use_depth, use_lidar)
        
        self.layer_1 = nn.Linear(self.encoder.output_dim, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, action_dim)
        self.max_action = max_action
    
    def forward(self, x, img = None, lidar = None):
        x = self.encoder(x, img, lidar)

        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x)) * self.max_action
        return x

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, use_camera = False, use_depth = False, use_lidar = False):
        super().__init__()
        self.encoder = SensorFusionEncoder(state_dim, use_camera, use_depth, use_lidar)
        
        # Input to Q-layer is Encoder Output + Action
        self.layer_1 = nn.Linear(self.encoder.output_dim + action_dim, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, 1)
    
    def forward(self, state, action, img = None, lidar = None):
        x = self.encoder(state, img, lidar)
        x = torch.cat([x, action], dim = 1) # Concat embedding + action
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, actor_lr, critic_lr,
                 buffer_size, batch_size, use_camera = False, use_depth = False, 
                 use_lidar = False, gamma = 0.99, tau = 0.005):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_camera = use_camera
        self.use_depth = use_depth
        self.use_lidar = use_lidar

        self.main_actor = ActorNetwork(state_dim, action_dim, max_action, 
                                        use_camera, use_depth, use_lidar).to(self.device)
        self.main_critic = CriticNetwork(state_dim, action_dim, use_camera, use_depth, use_lidar).to(self.device)

        self.target_actor = ActorNetwork(state_dim, action_dim, max_action,
                                          use_camera, use_depth, use_lidar).to(self.device)
        self.target_critic = CriticNetwork(state_dim, action_dim, use_camera, use_depth, use_lidar).to(self.device)
        
        self.target_actor.load_state_dict(self.main_actor.state_dict())
        self.target_critic.load_state_dict(self.main_critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.main_actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.main_critic.parameters(), lr = critic_lr)

        self.buffer = OffPolicyReplayBuffer(buffer_size)
        
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_action = max_action

    def soft_update_target_networks(self):
        for target_param, main_param in zip(self.target_actor.parameters(), self.main_actor.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

        for target_param, main_param in zip(self.target_critic.parameters(), self.main_critic.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
    
    @torch.no_grad()
    def get_action(self, state, img = None, lidar = None, exploration_noise = 0.1):
        state_tensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
        img_t = torch.tensor(img, dtype = torch.float32).unsqueeze(0).to(self.device) if img is not None else None
        lidar_t = torch.tensor(lidar, dtype = torch.float32).unsqueeze(0).to(self.device) if lidar is not None else None
        
        self.main_actor.eval()
        action = self.main_actor(state_tensor, img_t, lidar_t).squeeze(0).cpu().numpy()
        self.main_actor.train()
        
        noise = np.random.normal(0, self.max_action * exploration_noise, size = action.shape)
        action = np.clip(action + noise, -self.max_action, self.max_action)
        return action

    @torch.no_grad()
    def get_deterministic_action(self, state, img = None, lidar = None):
        state_tensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)
        img_t = torch.tensor(img, dtype = torch.float32).unsqueeze(0).to(self.device) if img is not None else None
        lidar_t = torch.tensor(lidar, dtype = torch.float32).unsqueeze(0).to(self.device) if lidar is not None else None

        self.main_actor.eval()
        action = self.main_actor(state_tensor, img_t, lidar_t).squeeze(0).cpu().numpy()
        self.main_actor.train()
        return np.clip(action, -self.max_action, self.max_action)

    def remember(self, state, img, lidar, action, reward, done, next_state, next_img, next_lidar):
        self.buffer.remember(state, img, lidar, action, reward, done, next_state, next_img, next_lidar)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None, None
        
        states, imgs, lidars, actions, rewards, next_states, next_imgs, next_lidars, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype = torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype = torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype = torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype = torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype = torch.float32).unsqueeze(1).to(self.device)

        imgs = torch.tensor(imgs, dtype = torch.float32).to(self.device) if imgs is not None else None
        lidars = torch.tensor(lidars, dtype = torch.float32).to(self.device) if lidars is not None else None
        next_imgs = torch.tensor(next_imgs, dtype = torch.float32).to(self.device) if next_imgs is not None else None
        next_lidars = torch.tensor(next_lidars, dtype = torch.float32).to(self.device) if next_lidars is not None else None

        with torch.no_grad():
            next_actions = self.target_actor(next_states, next_imgs, next_lidars)
            next_q_values = self.target_critic(next_states, next_actions, next_imgs, next_lidars)
            td_target = rewards + (self.gamma * next_q_values * (1 - dones))

        td_target = td_target.detach()
        current_q_values = self.main_critic(states, actions, imgs, lidars)
        critic_loss = F.mse_loss(current_q_values, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.main_critic.parameters(), max_norm = 1.0)
        self.critic_optimizer.step()

        predicted_actions = self.main_actor(states, imgs, lidars)
        q_values = self.main_critic(states, predicted_actions, imgs, lidars)
        actor_loss = -q_values.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.main_actor.parameters(), max_norm = 1.0)
        self.actor_optimizer.step()
        
        self.soft_update_target_networks()
        
        return actor_loss.item(), critic_loss.item()
        
    def save_models(self, save_dir):
        torch.save(self.main_actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.main_critic.state_dict(), os.path.join(save_dir, "critic.pth"))

    def load_models(self, load_dir):
        self.main_actor.load_state_dict(torch.load(os.path.join(load_dir, "actor.pth"), map_location = self.device))
        self.main_critic.load_state_dict(torch.load(os.path.join(load_dir, "critic.pth"), map_location = self.device))

        self.target_actor.load_state_dict(self.main_actor.state_dict())
        self.target_critic.load_state_dict(self.main_critic.state_dict())