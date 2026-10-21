# -*- coding: utf-8 -*-
from torch.nn.modules.pooling import AdaptiveAvgPool2d
import torch.nn as nn
from isecret.model.utils import make_norm, make_paddding, check_architecture
from isecret.model.blocks import ResnetBlock
import torch
import functools
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os
from option import convert_config

import torch
import torch.nn as nn
from isecret.model.backbone import BACKBONE_REGISTRY

@ BACKBONE_REGISTRY.register('u-net-structure')
class UNetGenerator(nn.Module):
    def __init__(self, args):
        super(UNetGenerator, self).__init__()
        self.args = args

        # 检查基本设置
        check_architecture(args)

        # 获取 padding 和 normalization
        padding = make_paddding(args)
        norm_layer, use_bias = make_norm(args)

        # Head 部分，处理输入图像
        self.head = nn.Sequential(
            padding(3),  # 用于配合 7x7 卷积核
            nn.Conv2d(self.args.model.input_nc, self.args.model.n_filters, kernel_size=7, bias=use_bias),
            norm_layer(self.args.model.n_filters),
            nn.ReLU(True)
        )

        # 编码器 (下采样)，4 次下采样
        self.encoder = nn.ModuleList()
        for i in range(self.args.model.n_downs):  # 固定4次下采样
            mult = 2 ** i
            self.encoder.append(nn.Sequential(
                padding(1),  
                nn.Conv2d(self.args.model.n_filters * mult,
                          self.args.model.n_filters * mult * 2, 
                          kernel_size=3, stride=2, bias=use_bias),
                norm_layer(self.args.model.n_filters * mult * 2),
                nn.ReLU(True)
            ))

        # 解码器 (上采样)，用于标准输出
        self.decoder = nn.ModuleList()
        self.conv_after_skip = nn.ModuleList()
        for i in range(self.args.model.n_downs):  # 固定4次上采样
            mult = 2 ** (self.args.model.n_downs - i)
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(self.args.model.n_filters * mult, 
                                   int(self.args.model.n_filters * mult / 2),
                                   kernel_size=3, stride=2, 
                                   padding=1, output_padding=1, 
                                   bias=use_bias),
                norm_layer(int(self.args.model.n_filters * mult / 2)),
                nn.ReLU(True)
            ))
            self.conv_after_skip.append(nn.Conv2d(self.args.model.n_filters * mult, 
                                                  int(self.args.model.n_filters * mult / 2),
                                                  kernel_size=1, bias=use_bias))

        # 解码器 (上采样)，用于病灶输出
        self.structure_decoder = nn.ModuleList()
        self.conv_after_skip_structure = nn.ModuleList()
        for i in range(self.args.model.n_downs):  # 固定4次上采样
            mult = int(2 ** (self.args.model.n_downs - i))
            self.structure_decoder.append(nn.Sequential(
                nn.ConvTranspose2d(self.args.model.n_filters * mult, 
                                   int(self.args.model.n_filters * mult / 2),
                                   kernel_size=3, stride=2, 
                                   padding=1, output_padding=1, 
                                   bias=use_bias),
                norm_layer(int(self.args.model.n_filters * mult / 2)),
                nn.ReLU(True)
            ))
            self.conv_after_skip_structure.append(nn.Conv2d(self.args.model.n_filters * mult, 
                                                         int(self.args.model.n_filters * mult / 2),
                                                         kernel_size=1, bias=use_bias))

        self.final_layer = nn.Sequential(
            padding(3),  
            nn.Conv2d(self.args.model.n_filters, self.args.model.output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )
        self.final_structure_layer = nn.Sequential(
            padding(3),  
            nn.Conv2d(self.args.model.n_filters, self.args.model.structure_output_nc, kernel_size=7, padding=0),
            # nn.Tanh()
        )

    def forward(self, input, layers=None):
        if layers:
            return self.forward_feature(input)
        # Head 部分
        x = self.head(input)
        # print(x.shape)
        # 编码器 (下采样) 过程
        skips = [x]

        for down in self.encoder:
            x = down(x)
            skips.append(x)
        
        structure_skip = []

        structure_x = x
        for i, structure_up in enumerate(self.structure_decoder):
            structure_x = structure_up(structure_x)
            structure_skip_connection = skips[-(i + 2)]  # 从 skip connections 里取出病灶特征图
            structure_x = torch.cat([structure_x, structure_skip_connection], dim=1)  # 拼接病灶 skip connection
            structure_x = self.conv_after_skip_structure[i](structure_x)  # 1x1 卷积调整通道数
            structure_skip.append(structure_x)
        structure_output = self.final_structure_layer(structure_x)
        
        # 解码器 (上采样) 过程，标准解码器带 skip connection
        for i, up in enumerate(self.decoder):
            x = up(x)
            skip_connection = structure_skip[i]  # 从 skip connections 里取出对应的特征图
            x = torch.cat([x, skip_connection], dim=1)  # 拼接 skip connection
            x = self.conv_after_skip[i](x)  # 1x1 卷积调整通道数
        # torch.backends.cudnn.benchmark = False
        output = self.final_layer(x)

        return output, structure_output

    def forward_feature(self, input):
        x = self.head(input)
        # print(x.shape)
        # 编码器 (下采样) 过程
        skips = [x]

        for down in self.encoder:
            x = down(x)
            skips.append(x)
        
        return skips, x

# import argparse
# args = argparse.ArgumentParser(description='the option of the project Fundus-Enhancement')
# model_args = args.add_argument_group(description='model')
# model_args.add_argument('--model', type=str, default='i-secret', help='model. Usable: i-secret')
# model_args.add_argument('--n_layers', type=int, default=3, help='the number of layers used in discriminator')
# model_args.add_argument('--n_downs', type=int, default=4, help='the number of down-samplings used in generator')
# model_args.add_argument('--n_filters', type=int, default=64, help='the number of filters used in generator')
# model_args.add_argument('--n_blocks', type=int, default=9,
#                         help='the number of blocks used in the low-level of generator')
# model_args.add_argument('--input_nc', type=int, default=3, help='the number of input channels')
# model_args.add_argument('--output_nc', type=int, default=3, help='the number of output channels')
# model_args.add_argument('--dis_n_filters', type=int, default=64, help='the number of filters used in discriminator')
# model_args.add_argument('--norm', type=str, default='in', help='Norm layer. Usable: in | bn ')
# model_args.add_argument('--padding', type=str, default='reflect', help='padding layer. '
#                                                                        'Usable: reflect | replication | zero ')
# model_args.add_argument('--gan_loss', type=str, default='lsgan', help='GAN loss. Usable: lsgan ')
# model_args.add_argument('--generator', type=str,
#                         default='reformer', help='Generator backbone. Usable:  resnet ')
# model_args.add_argument('--discriminator', type=str,
#                         default='patchgan', help='Discriminator . Usable: patchgan ')
# model_args.add_argument('--f_model', type=str, default='mlp', help='the f_model used in CutGAN')
# model_args.add_argument('--no_mlp', action='store_true', default=False, help='do not use mlp in CutGAN')
# args = args.parse_args()

# # import debugpy
# # debugpy.listen(5678)
# # print("Wait for Debugger...")
# # debugpy.wait_for_client()
# # print("Debugger Attached")

# args_dict = {}
# sub_args_list = [model_args]
# for sub_args in sub_args_list:
#     sub_options = [action.dest for action in sub_args._group_actions]
#     sub_names = {name: value for (name, value) in args._get_kwargs() if name in sub_options}
#     sub_node = {}
#     for sub_name in sub_names:
#         sub_node[sub_name] = sub_names[sub_name]
#     args_dict[sub_args.description] = sub_node
# args = convert_config(args_dict)

# model = UNetGenerator(args)
# print(model)
# layers = [1,5,9,11,15,18,20]
# img = torch.randn((1, 3, 512, 512))
# output, structure_output = model(img)
# # print(output.shape)
# # print(structure_output.shape)
# features, latent = model(img, layers=layers)
# for f in features:
#     print(f.shape)