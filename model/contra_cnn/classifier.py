from torch import nn

class Classifier(nn.Module):
    def __init__(self, input_shape):
        super(Classifier, self).__init__()
        self.input_shape = input_shape
        last_channel = 16  # 最后一个卷积层的输出通道数
        # Keras的Conv2D(kernel_size=1)相当于PyTorch的nn.Conv2d(kernel_size=(1,1))
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=last_channel, kernel_size=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(last_channel * input_shape[1] * input_shape[2], 8)  # 输入形状需要调整为16 * H * W
        self.fc2 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 将x转换为4D
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x