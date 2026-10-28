import torch
import torch.nn as nn
from .activations import get_activation

class PathwayA_1DCNN(nn.Module):
    def __init__(self, conv_act='ReLU', dense_act='ReLU', input_len=256, dense_bias_init=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.act1 = get_activation(conv_act)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act2 = get_activation(conv_act)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.act3 = get_activation(conv_act)
        self.pool = nn.MaxPool1d(2)
        flat_dim = 256 * (input_len // 8)
        self.fc_dense = nn.Linear(flat_dim, 512)
        self.act_dense = get_activation(dense_act)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(512, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if dense_bias_init != 0.0:
            nn.init.constant_(self.fc_dense.bias, dense_bias_init)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = self.pool(self.act3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        pre_dense = self.fc_dense(x)
        h = self.dropout(self.act_dense(pre_dense))
        return self.fc_out(h).squeeze(-1), pre_dense

class PathwayB_2DCNN(nn.Module):
    def __init__(self, conv_act='ReLU', dense_act='ReLU', in_h=100, in_w=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.act1 = get_activation(conv_act)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act2 = get_activation(conv_act)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.act3 = get_activation(conv_act)
        self.pool = nn.MaxPool2d(2, 2)
        flat_dim = 256 * (in_h // 8) * (in_w // 8)
        self.fc_dense = nn.Linear(flat_dim, 512)
        self.act_dense = get_activation(dense_act)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(512, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = self.pool(self.act3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        pre_dense = self.fc_dense(x)
        h = self.dropout(self.act_dense(pre_dense))
        return self.fc_out(h).squeeze(-1), pre_dense
