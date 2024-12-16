import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from torchvision import models as resnet_model


#layer-wise feature fusion
class LWFF(nn.Module):
    def __init__(self, channel, r=16):
        super(LWFF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b,c)
        y1 = self.fc(y1)
        y2 = self.fc(y2)
        y = (y1 + y2).view(b,c,1,1)
        y = torch.mul(x, y)
        return y




class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, r ):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=r),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class up_conv_1(nn.Module):
    def __init__(self, ch_in, ch_out, r ):
        super(up_conv_1, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=r),
            #nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        # 如果采用pool进行下采样的话没有特征提取的能力  所有用卷积来代替线下采样 压缩图像
        self.model = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)


# 交叉注意力
class EfficientCrossAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, y: torch.Tensor, x: torch.Tensor, H, W):
        B, N, C = x.shape  # x:24,3136,64 H:56
        q = self.q(y).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)  # 24,1,3136,64

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)  # 24,64,56,56
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)  # 24,49,64
            x = self.norm(sp_x)  # X:24,49,64 ??降低空间

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)  # 2,24,1,49,64
        k, v = kv[0], kv[1]  # 24,1,49,64

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 24,1,3136,49
        attn_score = attn.softmax(dim=-1)  # 24,1,3136,49

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)  # 24,3136,64
        out = self.proj(x_atten)  # out:24,3136,64
        out = self.norm(out)

        return out  # 16,196,320


class EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape  # x:24,3136,64 H:56
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)  # 24,1,3136,64

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)  # 24,64,56,56
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)  # 24,49,64
            x = self.norm(sp_x)  # X:24,49,64 ??降低空间

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)  # 2,24,1,49,64
        k, v = kv[0], kv[1]  # 24,1,49,64

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 24,1,3136,49
        attn_score = attn.softmax(dim=-1)  # 24,1,3136,49

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)  # 24,3136,64
        out = self.proj(x_atten)  # out:24,3136,64

        return out


class SelfAtten(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        if (len(self.reduction_ratio) == 4):
            self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[3], reduction_ratio[3])
            self.sr1 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr2 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        elif (len(self.reduction_ratio) == 3):
            self.sr0 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr1 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if (len(self.reduction_ratio) == 4):
            tem0 = x[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
            tem1 = x[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem2 = x[:, 4704:5684, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem3 = x[:, 5684:6076, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))

        if (len(self.reduction_ratio) == 3):
            tem0 = x[:, :1568, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem1 = x[:, 1568:2548, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem2 = x[:, 2548:2940, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, tem2], -2))

        return reduce_out


class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio  # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class LocalEnhance_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.local_pos = DWConv(dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)
        local_v = v.permute(0, 2, 1, 3).reshape(B, N, C)
        local_pos = self.local_pos(local_v).reshape(B, -1, self.head, C // self.head).permute(0, 2, 1, 3)
        x_atten = ((attn_score @ v) + local_pos).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class DWConv(nn.Module):  # 通道卷积
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)








class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class MixFFN_skip(nn.Module):  # enhanced  仅使用了一层LayerNorm
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MixD_FFN(nn.Module):
    def __init__(self, c1, c2, fuse_mode="add"):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1) if fuse_mode == "add" else nn.Linear(c2 * 2, c1)
        self.fuse_mode = fuse_mode

    def forward(self, x):
        ax = self.dwconv(self.fc1(x), H, W)
        fuse = self.act(ax + self.fc1(x)) if self.fuse_mode == "add" else self.act(torch.cat([ax, self.fc1(x)], 2))
        out = self.fc2(ax)
        return out


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)  # 1.x:24,3,224,224->24,64,56,56 2.
        _, _, H, W = px.shape  # H:56,W:56
        fx = px.flatten(2).transpose(1, 2)  # x:24,3136,64
        nfx = self.norm(fx)  # 24,3136,64
        return nfx, H, W


class TransformerBlock_1(nn.Module):  # Enhanced
    def __init__(self, dim, head, reduction_ratio=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # LayerNorm
        self.attn = EfficientCrossAtten(dim, head, reduction_ratio)  # att
        self.norm2 = nn.LayerNorm(dim)  # LayerNorm
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, y, x, H, W) -> torch.Tensor:  # 修改传入的参数改为交叉注意力
        tx = x + self.attn(self.norm1(y), self.norm1(x), H, W)  # tx:24,3136,64 残差连接 y和x互换
        # mx = tx + self.mlp(self.norm2(tx), H, W)#mix_FNN mx:24,3136.64
        return tx

class cross_TransformerBlock(nn.Module):#Enhanced
    def __init__(self, dim, head, reduction_ratio=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)#LayerNorm
        self.attn = EfficientCrossAtten(dim, head, reduction_ratio)#att
        self.norm2 = nn.LayerNorm(dim)#LayerNorm
        if token_mlp=='mix':
            self.mlp = MixFFN(dim, int(dim*4))
        elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim*4))
        else:
            self.mlp = MLP_FFN(dim, int(dim*4))

    def forward(self, y,x, H, W) -> torch.Tensor:#修改传入的参数改为交叉注意力
        tx = x + self.attn(self.norm1(y),self.norm1(x), H, W)#tx:24,3136,64 残差连接 y和x互换
        mx = tx + self.mlp(self.norm2(tx), H, W)#mix_FNN mx:24,3136.64
        return mx


class TransformerBlock(nn.Module):  # Enhanced
    def __init__(self, dim, head, reduction_ratio=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # LayerNorm
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)  # att
        self.norm2 = nn.LayerNorm(dim)  # LayerNorm
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:  # 修改传入的参数改为交叉注意力
        tx = x + self.attn(self.norm1(x), H, W)  # tx:24,3136,64 残差连接
        mx = tx + self.mlp(self.norm2(tx), H, W)  # mix_FNN mx:24,3136.64
        return mx


class FuseTransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, fuse_mode="add"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixD_FFN(dim, int(dim * 4), fuse_mode)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class MiT(nn.Module):
    def __init__(self, image_size, dims, layers, token_mlp='mix_skip'):  # dims[64, 128, 320, 512]
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        #卷积分支
        resnet = resnet_model.resnet50(pretrained=True)

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(image_size // 4, patch_sizes[1], strides[1], padding_sizes[1],
                                                   dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(image_size // 8, patch_sizes[2], strides[2], padding_sizes[2],
                                                   dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(image_size // 16, patch_sizes[3], strides[3], padding_sizes[3],
                                                   dims[2], dims[3])

        # transformer encoder block
        self.block1 = nn.ModuleList([
            TransformerBlock(dims[0], heads[0], reduction_ratios[0], token_mlp)
            for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList([
            TransformerBlock(dims[1], heads[1], reduction_ratios[1], token_mlp)
            for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList([
            TransformerBlock(dims[2], heads[2], reduction_ratios[2], token_mlp)
            for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList([
            TransformerBlock(dims[3], heads[3], reduction_ratios[3], token_mlp)
            for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])


        self.L1 = LWFF(128)
        self.L2 = LWFF(256)
        self.L3 = LWFF(640)
        self.L4 = LWFF(1024)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512
        self.Conv6 = conv_block(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Conv7 = conv_block(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Conv8 = conv_block(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Conv9 = conv_block(ch_in=dims[0] * 2, ch_out=dims[0])
        self.conv1 = nn.Conv2d(256,64,1,1,0)
        self.conv2 = nn.Conv2d(512,128,1,1,0)
        self.conv3 = nn.Conv2d(1024,320,1,1,0)
        self.conv4 = nn.Conv2d(2048,512,1,1,0)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]  # x:24,3,224,224
        #保存卷积分支特征
        outs_1 = []
        #保存Transformer分支特征
        outs = []
        # stage 1
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.encoder1(x1)
        x1 = self.Maxpool(x1)
        x_t2 = self.conv1(x1)
        outs_1.append(x_t2)
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = torch.cat([x, x_t2], 1)
        x = self.L1(x)
        x = self.Conv9(x)
        outs.append(x)
        # stage2
        x2 = self.encoder2(x1)
        x2_t = self.conv2(x2)
        outs_1.append(x2_t)
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = torch.cat([x2_t, x], 1)
        x = self.L2(x)
        x = self.Conv6(x)
        outs.append(x)
        # stage 3
        x3 = self.encoder3(x2)
        x3_t = self.conv3(x3)
        outs_1.append(x3_t)
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = torch.cat([x3_t, x], 1)
        x = self.L3(x)
        x = self.Conv7(x)
        outs.append(x)
        # stage 4
        x4 = self.encoder4(x3)
        x4_t = self.conv4(x4)
        outs_1.append(x4_t)
        x, H, W = self.patch_embed4(x)

        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = torch.cat([x4_t, x], 1)
        x = self.L4(x)
        x = self.Conv8(x)
        outs.append(x)

        return outs,outs_1


class FuseMiT(nn.Module):
    def __init__(self, image_size, dims, layers, fuse_mode='add'):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(image_size // 4, patch_sizes[1], strides[1], padding_sizes[1],
                                                   dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(image_size // 8, patch_sizes[2], strides[2], padding_sizes[2],
                                                   dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(image_size // 16, patch_sizes[3], strides[3], padding_sizes[3],
                                                   dims[2], dims[3])

        # transformer encoder
        self.block1 = nn.ModuleList([
            FuseTransformerBlock(dims[0], heads[0], reduction_ratios[0], fuse_mode)
            for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList([
            FuseTransformerBlock(dims[1], heads[1], reduction_ratios[1], fuse_mode)
            for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList([
            FuseTransformerBlock(dims[2], heads[2], reduction_ratios[2], fuse_mode)
            for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList([
            FuseTransformerBlock(dims[3], heads[3], reduction_ratios[3], fuse_mode)
            for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])

        # self.head = nn.Linear(dims[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class Decoder(nn.Module):
    def __init__(self, dims, embed_dim, num_classes):
        super().__init__()

        self.linear_c1 = MLP(dims[0], embed_dim)
        self.linear_c2 = MLP(dims[1], embed_dim)
        self.linear_c3 = MLP(dims[2], embed_dim)
        self.linear_c4 = MLP(dims[3], embed_dim)

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim, 1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

        self.conv_seg = nn.Conv2d(128, num_classes, 1)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = inputs
        n = c1.shape[0]
        c1f = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        c2f = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        c2f = F.interpolate(c2f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c3f = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        c3f = F.interpolate(c3f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c4f = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        c4f = F.interpolate(c4f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c = self.linear_fuse(torch.cat([c4f, c3f, c2f, c1f], dim=1))
        c = self.dropout(c)
        return self.linear_pred(c)


segformer_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2], 256],  # [channel dimensions, num encoder layers, embed dim]
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2], 256],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3], 768],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3], 768],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3], 768],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3], 768]
}


class SegFormer(nn.Module):
    def __init__(self, model_name: str = 'B0', num_classes: int = 19, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in segformer_settings.keys(), f"SegFormer model name should be in {list(segformer_settings.keys())}"
        dims, layers, embed_dim = segformer_settings[model_name]

        self.backbone = MiT(image_size, dims, layers)
        self.decode_head = Decoder(dims, embed_dim, num_classes)

    def init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_outs = self.backbone(x)
        return self.decode_head(encoder_outs)

