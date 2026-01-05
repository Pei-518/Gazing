import torch
import torch.nn as nn
import torchvision.models as models

# ================= 1. 事件編碼器 (不變) =================
class EventCNN(nn.Module):
    def __init__(self, in_channels=5, feature_dim=128):
        super(EventCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.relu(self.fc(x))
        return x

# ================= 2. 雙流模型 (修改輸出層) =================
class DualStreamGazeModel(nn.Module):
    def __init__(self):
        super(DualStreamGazeModel, self).__init__()
        
        # 影像流 (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.image_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.image_dim = 512
        
        # 事件流
        self.event_backbone = EventCNN(in_channels=5, feature_dim=128)
        self.event_dim = 128
        
        # 融合層
        fusion_input_dim = self.image_dim + self.event_dim
        
        # 🔥 修改這裡：輸出變成 2 (X, Y)
        self.output_dim = 2 
        
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim) # 輸出 (Batch, 2)
        )

    def forward(self, image, event_voxel):
        feat_i = self.image_backbone(image).flatten(1)
        feat_e = self.event_backbone(event_voxel)
        feat_fused = torch.cat((feat_i, feat_e), dim=1)
        out = self.regression_head(feat_fused)
        return out