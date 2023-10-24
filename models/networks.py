import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        act_fn,
    )
    return model
def up_conv(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim,out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm3d(out_dim),

        act_fn,
    )
    return model

def double_conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
    )
    return model


def out_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=1, stride=1, padding=0),
    )
    return model

class P_RNet3D(nn.Module):

    def __init__(self, in_dim, num_filter,out_dim, init_weights=True):
        super(P_RNet3D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.ReLU(inplace=True)

        self.down_1 = double_conv_block(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_2 = double_conv_block(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_3 = double_conv_block(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.bridge = double_conv_block(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1 = up_conv(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = double_conv_block(self.num_filter * 12, self.num_filter * 4, act_fn)

        self.trans_2 = up_conv(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = double_conv_block(self.num_filter * 6, self.num_filter * 2, act_fn)

        self.trans_3 = up_conv(self.num_filter * 2, self.num_filter * 2, act_fn)
        self.up_3 = double_conv_block(self.num_filter * 3, self.num_filter, act_fn)

        self.out = out_block(self.num_filter, out_dim)

        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        bridge = self.bridge(pool_3)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        up_3 = self.up_3(concat_3)
        out = self.out(up_3)
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def build_model(config):
    model = P_RNet3D(config.model.c_in, 
                     config.model.c_blk,
                     config.model.n_classes,
                     config.model.init_weights).to(config.exp.device)
    if config.exp.multi_gpu:        
        model = DDP(model, device_ids=[config.exp.rank], output_device=config.exp.rank)
    return model
