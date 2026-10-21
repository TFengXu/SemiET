import torch
import torch.nn as nn
from einops import rearrange
import math

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x


# class Downsample(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Downsample, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
#         )
#         self.in_channel = in_channel
#         self.out_channel = out_channel

#     def forward(self, x):
#         B, L, C = x.shape
#         # import pdb;pdb.set_trace()
#         H = int(math.sqrt(L))
#         W = int(math.sqrt(L))
#         x = x.transpose(1, 2).contiguous().view(B, C, H, W)
#         out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
#         return out

#     def flops(self, H, W):
#         flops = 0
#         # conv
#         flops += H/2*W/2*self.in_channel*self.out_channel*4*4
#         print("Downsample:{%.2f}"%(flops/1e9))
#         return flops

# # Upsample Block
# class Upsample(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Upsample, self).__init__()
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
#         )
#         self.in_channel = in_channel
#         self.out_channel = out_channel
        
#     def forward(self, x):
#         B, L, C = x.shape
#         H = int(math.sqrt(L))
#         W = int(math.sqrt(L))
#         x = x.transpose(1, 2).contiguous().view(B, C, H, W)
#         out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
#         return out

#     def flops(self, H, W):
#         flops = 0
#         # conv
#         flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
#         print("Upsample:{%.2f}"%(flops/1e9))
#         return flops


## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return self.body(x).flatten(2).transpose(1,2).contiguous() # B H*W C

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return self.body(x).flatten(2).transpose(1,2).contiguous() # B H*W C