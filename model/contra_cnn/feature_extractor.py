# 定义特征提取器
from torch import nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 36)  # 简单的全连接层提取特征
        self.fc2 = nn.Linear(36, 32)
        self.fc3 = nn.Linear(32,feature_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)