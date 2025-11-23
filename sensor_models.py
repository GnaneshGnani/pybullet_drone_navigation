import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorFusionEncoder(nn.Module):
    def __init__(self, state_dim, use_camera, use_depth, use_lidar, lidar_dim = 360):
        super().__init__()
        self.use_camera = use_camera
        self.use_depth = use_depth
        self.use_lidar = use_lidar
        
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        fusion_dim = 64

        self.img_channels = 0
        if use_camera: self.img_channels +=  3
        if use_depth: self.img_channels +=  1
        
        if self.img_channels > 0:
            # Input: (Batch, Channels, 64, 64)
            self.cnn = nn.Sequential(
                nn.Conv2d(self.img_channels, 32, kernel_size = 8, stride = 4), 
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size = 4, stride = 2), 
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1), 
                nn.ReLU(),
                nn.Flatten()
            )
            # 64 filters * 4 * 4 spatial dim = 1024
            self.cnn_fc = nn.Linear(1024, 128)
            fusion_dim += 128

        if use_lidar:
            self.lidar_net = nn.Sequential(
                nn.Linear(lidar_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            fusion_dim += 64
            
        self.output_dim = fusion_dim

    def forward(self, state, img = None, lidar = None):
        s_emb = self.state_net(state)
        embeddings = [s_emb]
        
        if self.img_channels > 0:
            if img is None: 
                raise ValueError(f"Model expects image input (use_camera = {self.use_camera}, use_depth = {self.use_depth}), but got None")

            x = self.cnn(img)
            x = F.relu(self.cnn_fc(x))
            embeddings.append(x)
            
        if self.use_lidar:
            if lidar is None: 
                raise ValueError("Model expects lidar input, but got None")

            l_emb = self.lidar_net(lidar)
            embeddings.append(l_emb)
            
        return torch.cat(embeddings, dim = -1)