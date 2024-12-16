import torch
import torch.nn as nn
from .segformer import *
from typing import Tuple
from einops import rearrange

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
        # Fusion
        y = torch.mul(x, y)
        return y




class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)  # 对原特征图重新排列
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())
        return x

class MFSF_Layer(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims, dims * 4)
        self.mixffn2 = MixFFN_skip(dims * 2, dims * 8)
        self.mixffn3 = MixFFN_skip(dims * 5, dims * 20)
        self.mixffn4 = MixFFN_skip(dims * 8, dims * 32)
    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem1 = tx[:, :3136, :].reshape(B, -1, C)
        tem2 = tx[:, 3136:4704, :].reshape(B, -1, C * 2)
        tem3 = tx[:, 4704:5684, :].reshape(B, -1, C * 5)
        tem4 = tx[:, 5684:6076, :].reshape(B, -1, C * 8)

        m1f = self.mixffn1(tem1, 56, 56).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)

        tx2 = tx1 + t1

        return tx2




class MFSF(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.MFSF_Layer1 = MFSF_Layer(dims, head, reduction_ratios)
        self.MFSF_Layer2 = MFSF_Layer(dims, head, reduction_ratios)
        self.MFSF_Layer3 = MFSF_Layer(dims, head, reduction_ratios)
        self.MFSF_Layer4 = MFSF_Layer(dims, head, reduction_ratios)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        MFSF1 = self.MFSF_Layer1(x)
        MFSF2 = self.MFSF_Layer2(MFSF1)
        MFSF3 = self.MFSF_Layer3(MFSF2)
        MFSF4 = self.MFSF_Layer4(MFSF3)

        B, _, C = MFSF4.shape
        outs = []

        sk1 = MFSF4[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
        sk2 = MFSF4[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
        sk3 = MFSF4[:, 4704:5684, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
        sk4 = MFSF4[:, 5684:6076, :].reshape(B, 7, 7, C * 8).permute(0, 3, 1, 2)

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs



class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, token_mlp_mode, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            # transformer decoder 降维回复空间
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                norm_layer=norm_layer)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = cross_TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.Up3 = up_conv(320, 160,2)
        self.Up_conv3 = conv_block(160, 160)
        self.Up2 = up_conv(128, 64,2)
        self.Up_conv2 = conv_block(64, 64)
        self.convd1 = nn.Conv2d(in_channels=256, out_channels=320, kernel_size=1, stride=1, padding=0)
        self.convd2 = nn.Conv2d(in_channels=160, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.convd3 = nn.Conv2d(in_channels=576, out_channels=320, kernel_size=1, stride=1, padding=0)
        self.convd4 = nn.Conv2d(in_channels=288, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.convd5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)

        def init_weights(self):
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

        init_weights(self)

    def forward(self, x1, x2=None, x3=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            if c == 320:  # 16,14,14,320
                cat_linear_x = self.convd3(cat_x.view(b, 14, 14, -1).permute(0, 3, 2, 1))
                tran_layer_1 = self.layer_former_1(x3.view(b, -1, c), cat_linear_x.permute(0, 2, 3, 1).view(b, -1, c),h, w)
                out = self.Up3(tran_layer_1.view(b, 14, 14, -1).permute(0, 3, 2, 1)).permute(0, 2, 3, 1).view(b, -1,160)
            if c == 128:
                cat_linear_x = self.convd4(cat_x.view(b, 28, 28, -1).permute(0, 3, 2, 1))
                tran_layer_1 = self.layer_former_1(x3.view(b, -1, c), cat_linear_x.permute(0, 2, 3, 1).view(b, -1, c),h, w)
                out = self.Up2(tran_layer_1.view(b, 28, 28, -1).permute(0, 3, 2, 1)).permute(0, 2, 3, 1).view(b, -1, 64)
            if c == 64:
                cat_linear_x = self.convd5(cat_x.view(b, 56, 56, -1).permute(0, 3, 2, 1))
                tran_layer_1 = self.layer_former_1(x3.view(b, -1, c), cat_linear_x.permute(0, 2, 3, 1).view(b, -1, c),h, w)


            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_1).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
        else:
            out = self.layer_up(x1)  # 16,196,256
        return out


class PSCTNet(nn.Module):
    def __init__(self, num_classes=9, token_mlp_mode="mix_skip", encoder_pretrained=True):
        super().__init__()

        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        d_base_feat_size = 7  # 16 for 512 inputsize   7for 224
        in_out_chan = [[32, 64], [144, 128], [288, 320], [512, 512]]

        dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]
        self.backbone = MiT(224, dims, layers, token_mlp_mode)

        self.reduction_ratios = [1, 2, 4, 8]
        self.bridge = MFSF(64, 1, self.reduction_ratios)

        self.decoder_3 = MyDecoderLayer((d_base_feat_size, d_base_feat_size), in_out_chan[3], heads[3],
                                        reduction_ratios[3], token_mlp_mode, n_class=num_classes)
        self.decoder_2 = MyDecoderLayer((d_base_feat_size * 2, d_base_feat_size * 2), in_out_chan[2], heads[2],
                                        reduction_ratios[2], token_mlp_mode, n_class=num_classes)
        self.decoder_1 = MyDecoderLayer((d_base_feat_size * 4, d_base_feat_size * 4), in_out_chan[1], heads[1],
                                        reduction_ratios[1], token_mlp_mode, n_class=num_classes)
        self.decoder_0 = MyDecoderLayer((d_base_feat_size * 8, d_base_feat_size * 8), in_out_chan[0], heads[0],
                                        reduction_ratios[0], token_mlp_mode, n_class=num_classes, is_last=True)
        self.Up7 = up_conv_1(512, 512, 8)
        self.Up6 = up_conv_1(320, 320, 4)
        self.Up5 = up_conv_1(128, 128, 2)
        self.convd3 = nn.Conv2d(in_channels=576, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.convd4 = nn.Conv2d(in_channels=288, out_channels=160, kernel_size=1, stride=1, padding=0)
        self.convd5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.L1 = LWFF(1024)
        self.L2 = LWFF(576)
        self.L3 = LWFF(288)
        self.L4 = LWFF(128)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=4)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=8)

        #卷积分支上采样
        self.Up_conv4 = conv_block(576, 320)
        self.Up4 = up_conv(512, 256, 2)

        self.Up3 = up_conv(320, 160, 2)
        self.Up_conv3 = conv_block(288, 128)

        self.Up2 = up_conv(128, 64, 2)
        self.Up_conv2 = conv_block(128, 64)
    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder, cnns = self.backbone(x)
        #MFSF
        bridge = self.bridge(encoder)
        #MFCF
        c1,c2,c3,c4 = bridge
        c2 = self.Up5(c2)
        c3 = self.Up6(c3)
        c4 = self.Up7(c4)
        c = torch.cat([c1,c2,c3,c4],1)
        c = self.L1(c)
        c1 = c[:, :64, :]
        c2 = c[:, 64:192, :]
        c3 = c[:, 192:512, :]
        c4 = c[:, 512:1024, :]
        bridge[0] = c1
        bridge[1] = self.Maxpool1(c2)
        bridge[2] = self.Maxpool2(c3)
        bridge[3] = self.Maxpool3(c4)
        #卷积层上采样分支
        CU = []
        cu_1 = self.Up4(bridge[3])#16,256,14,14
        cu_1 = torch.cat([cu_1,bridge[2]],1)#576
        cu_1 = self.Up_conv4(cu_1)#320,14,14
        CU.append(cu_1)
        cu_2 = self.Up3(cu_1)  # 16,160,28,28
        cu_2 = torch.cat([cu_2, bridge[1]], 1)#16,288,28,28
        cu_2 = self.Up_conv3(cu_2)#128
        CU.append(cu_2)
        cu_3 = self.Up2(cu_2)  # 16,64,56,56
        cu_3 = torch.cat([cu_3, bridge[0]], 1)#16,128,56,56
        cu_3 = self.Up_conv2(cu_3)
        CU.append(cu_3)
        #Transformer上采样分支
        b, c, _, _ = bridge[3].shape
        tmp_3 = self.decoder_3(bridge[3].permute(0, 2, 3, 1).view(b, -1, c), None, cnns[3].permute(0, 2, 3, 1))
        tmp = torch.cat([cu_1,tmp_3.view(b,14,14,-1).permute(0,3,1,2)],1)#16,576,14,14,
        tmp = self.L2(tmp)
        tmp = self.convd3(tmp)#16,320,14,14
        tmp_3 = tmp.permute(0,2,3,1).view(b,-1,256)
        # print("stage2-----")#16,784,160
        tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0, 2, 3, 1), cnns[2].permute(0, 2, 3, 1))

        tmp = torch.cat([cu_2, tmp_2.view(b, 28, 28, -1).permute(0, 3, 1, 2)],1)# 16,288,14,14,
        tmp = self.L3(tmp)
        tmp = self.convd4(tmp)  # 16,160,14,14
        tmp_2 = tmp.permute(0, 2, 3, 1).view(b, -1, 160)

        # print("stage1-----")#16,3136,64
        tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0, 2, 3, 1), cnns[1].permute(0, 2, 3, 1))

        tmp = torch.cat([cu_3, tmp_1.view(b, 56, 56, -1).permute(0, 3, 1, 2)], 1)  # 16,128,56,56,
        tmp = self.L4(tmp)
        tmp = self.convd5(tmp)  # 16,64,56,56
        tmp_1 = tmp.permute(0, 2, 3, 1).view(b, -1, 64)
        # print("stage0-----"
        tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0, 2, 3, 1), cnns[0].permute(0, 2, 3, 1))

        return tmp_0


