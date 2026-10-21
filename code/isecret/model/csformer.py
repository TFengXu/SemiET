import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numbers
from pdb import set_trace as stx
import math
from isecret.model.csformer_module.msa import MSABlock
from isecret.model.csformer_module.sw_w_msa import SwinTransformerBlock
from isecret.model.csformer_module.up_down import Upsample, Downsample
# from isecret.model.csformer_module.mdta import TransformerBlock
from isecret.model.backbone import BACKBONE_REGISTRY

# OverlapPatchEmbed
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel 
        print("Input_proj:{%.2f}"%(flops/1e9))
        return flops

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel 
        print("Output_proj:{%.2f}"%(flops/1e9))
        return flops

@ BACKBONE_REGISTRY.register('csformer')
class CSformer(nn.Module):
    def __init__(self, args, img_size=512, inp_channels=3, out_channels=3,
                 embed_dim=32, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 2], num_heads=[1,2,4,8],
                 bottleneck_depth = 2, bottleneck_num_heads = 8,
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 ffn_expansion_factor=2.66, bias=False, **kwargs):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.patch_embed = OverlapPatchEmbed(in_channel=inp_channels, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=out_channels, kernel_size=3, stride=1)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder
        self.layers = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_i = nn.Sequential(*[
                SwinTransformerBlock(dim=int(embed_dim * 2 ** i_layer), 
                                 input_resolution=(img_size // (2 ** i_layer), img_size // (2 ** i_layer)),
                                 num_heads=num_heads[i_layer], window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])][i],
                                 norm_layer=norm_layer)
            for i in range(depths[i_layer])])  
            # down_sample_i = Downsample(embed_dim * 2 ** i_layer, embed_dim * 2 ** (i_layer+1))
            down_sample_i = Downsample(embed_dim * 2 ** i_layer)
            self.layers.append(layer_i)
            self.down_samples.append(down_sample_i)

        # bulid bottleneck layers
        self.bottleneck = nn.Sequential(*[
            MSABlock(dim=embed_dim* 2 ** 4, num_heads=bottleneck_num_heads, 
                     mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_rate, attn_drop_ratio=attn_drop_rate, drop_path_ratio=0,
                  norm_layer=norm_layer)
            for i in range(bottleneck_depth)
        ])

        # self.bottleneck = nn.Sequential(*[TransformerBlock(
        #     dim=int(embed_dim*2**4), num_heads=bottleneck_num_heads, 
        #     ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(bottleneck_depth)])


        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear_i = nn.Linear(2*int(embed_dim*2**(self.num_layers-i_layer)),
                                      int(embed_dim*2**(self.num_layers-i_layer)))
            # up_sample_i = Upsample(embed_dim*2**(self.num_layers-i_layer), embed_dim*2**(self.num_layers-1-i_layer))
            up_sample_i = Upsample(embed_dim*2**(self.num_layers-i_layer))
            layer_up_i = nn.Sequential(*[
                SwinTransformerBlock(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), 
                                input_resolution=(img_size // (2 ** (self.num_layers-1-i_layer)),
                                                img_size // (2 ** (self.num_layers-1-i_layer))),
                                num_heads=num_heads[(self.num_layers-1-i_layer)], window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths_decoder[:(self.num_layers-1-i_layer)]):sum(depths_decoder[:(self.num_layers-1-i_layer) + 1])][i],
                                norm_layer=norm_layer)
            for i in range(depths_decoder[(self.num_layers-1-i_layer)])])

            self.layers_up.append(layer_up_i)
            self.up_samples.append(up_sample_i)
            self.concat_back_dim.append(concat_linear_i)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder
    def forward_features(self, x):
        x_downsample = []
        for layer, down_sample in zip(self.layers, self.down_samples):
            x = layer(x)
            x = down_sample(x)
            x_downsample.append(x)
        return x, x_downsample
    
    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            x = torch.cat([x,x_downsample[3-inx]],-1)
            x = self.concat_back_dim[inx](x)
            x = self.up_samples[inx](x)
            x = layer_up(x)  
        return x
    
    def languae2pic(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return x

    def forward_cut_feature(self, x):
        feats = []
        feats.append(self.languae2pic(x))
        for layer, down_sample in zip(self.layers, self.down_samples):
            x = layer(x)
            feats.append(self.languae2pic(x))
            x = down_sample(x)
        feats.append(self.languae2pic(x))
        latent = self.bottleneck(x)
        feats.append(self.languae2pic(latent))
        return feats, latent

    def forward(self, input_image, layers=None):
        x = self.patch_embed(input_image)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        if layers is not None:
            return self.forward_cut_feature(x)
        
        x, x_downsample = self.forward_features(x)
        x = self.bottleneck(x)
        x = self.forward_up_features(x,x_downsample)

        # Output Projection
        out = self.output_proj(x) + input_image
        return out.clamp(-1, 1)

# model = CSformer(img_size=512, window_size=8, embed_dim=48)
# model.to(4)
# x = torch.rand(size=(6, 3, 512, 512))
# x = x.to(4)
# out = model(x) 
# # output, x_feature = model(x)
# feats, latent = model(x, layers=0)
# for fm in feats:
#     print(fm.shape)

# print(output.shape)