import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch
import torch.nn as nn
from resnet50 import generate_model
import torch.nn as nn
device = torch.device("cuda:0")

class ResNet50_GRU_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2, seq_len=128, pretrain_path='path/to/resnet_50_23dataset.pth'):
        super(ResNet50_GRU_Model, self).__init__()
        self.resnet50model = generate_model(input_W=96, input_H=96, input_D=14, pretrain_path=pretrain_path, pretrained=True)
        for name, param in self.resnet50model.named_parameters():
            if 'conv_seg' not in name and 'reduce_channels' not in name and 'reduce_bn' not in name and 'reduce_relu' not in name:
                param.requires_grad = False
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.seq_len = seq_len
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, image2, image3):
        ddf2 = self.resnet50model(image2)
        ddf3 = self.resnet50model(image3)
        encoder_concatenated = torch.cat([ddf2.squeeze(-3).squeeze(-2).squeeze(-1), ddf3.squeeze(-3).squeeze(-2).squeeze(-1)], dim=1)
        output, _ = self.gru(encoder_concatenated.unsqueeze(1).repeat(1, self.seq_len, 1))
        output = output[:, -1, :]
        output = self.bn(output)
        output = self.fc(output)
        return output