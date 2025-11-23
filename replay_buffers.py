import random
import numpy as np
from collections import deque

class OffPolicyReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen = buffer_size)

    def remember(self, state, img, lidar, action, reward, done, next_state, next_img, next_lidar):
        self.buffer.append((state, img, lidar, action, reward, done, next_state, next_img, next_lidar))

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        batch = list(zip(*mini_batch))
        
        def stack_if_exists(idx):
            # Check if ANY element is None
            if any(item is None for item in batch[idx]):
                if not all(item is None for item in batch[idx]):
                    raise ValueError(f"Inconsistent sensor data at index {idx}: mix of None and non-None values")
                return None
            return np.array(batch[idx])

        states = np.array(batch[0])
        imgs = stack_if_exists(1)
        lidars = stack_if_exists(2)
        actions = np.array(batch[3])
        rewards = np.array(batch[4])
        dones = np.array(batch[5])
        next_states = np.array(batch[6])
        next_imgs = stack_if_exists(7)
        next_lidars = stack_if_exists(8)

        return (states, imgs, lidars, actions, rewards, 
                next_states, next_imgs, next_lidars, dones)

    def __len__(self):
        return len(self.buffer)
    

class OnPolicyReplayBuffer:
    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def store(self, state, img, lidar, action, reward, done, log_prob, value):
        self.states.append(state)
        self.imgs.append(img)
        self.lidars.append(lidar)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states = []
        self.imgs = []
        self.lidars = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def compute_advantages_and_returns(self, last_value, done):
        advantages = np.zeros(len(self.rewards), dtype = np.float32)
        last_gae_lam = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
                
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + np.array(self.values, dtype = np.float32)
        return advantages, returns

    def get_batch(self):
        img_batch = None
        if len(self.imgs) > 0 and self.imgs[0] is not None:
            img_batch = np.array(self.imgs)
            
        lidar_batch = None
        if len(self.lidars) > 0 and self.lidars[0] is not None:
            lidar_batch = np.array(self.lidars)

        return (
            np.array(self.states),
            img_batch,
            lidar_batch,
            np.array(self.actions),
            np.array(self.log_probs),
            np.array(self.values),
        )
    
    def __len__(self):
        return len(self.states)