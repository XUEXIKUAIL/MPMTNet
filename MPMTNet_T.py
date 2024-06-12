import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
import collections
from collections import OrderedDict
from .mix_transformer import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from timm.models.layers import trunc_normal_
# import math
from mmcv.cnn import build_norm_layer
import cv2
import numpy as np
import matplotlib.pyplot as plt
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, s=1, p=0, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
                )

    def forward(self, x):
        return self.conv(x)

class SalHead(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channel, n_classes, 1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.conv(x)

class AyC_crossfusion(nn.Module):
    def __init__(self, channel):
        super(AyC_crossfusion, self).__init__()
        self.conv13 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=(1, 3), padding=(0, 1))
        self.conv31 = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=(3, 1), padding=(1, 0))
        self.conv1 = nn.Conv2d(channel, channel, 1, 1, 0)
        self.Dscon3 = DSConv3x3(channel, channel)
        self.sig = nn.Sigmoid()
    def forward(self, r, d):
        rd = r + d
        full = self.Dscon3(rd)
        full_1 = self.conv13(self.conv1(rd))
        full_2 = self.conv31(self.conv1(rd))
        # print('full1', full_1.shape)
        # print('full2', full_2.shape)

        w_full12 = self.sig(full_1 * full_2)
        # print('w_full12', w_full12.shape)
        full_x = full * w_full12
        ayc_out = full + full_x
        return ayc_out

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6

class h_wish(nn.Module):
    def __init__(self, inplace=True):
        super(h_wish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class pp_upsample(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1),
            nn.BatchNorm2d(outc),
            nn.PReLU()
        )
    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)
class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x
class Sea_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5              # 开根号下1/2
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2d_BN(2 * self.dh, 2 * self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.dh, norm_cfg=norm_cfg)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # detail enhance
        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3,
                                                                                       2)  #####mean(dim=xx) 压缩某个维度
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)

        attn_row = torch.matmul(qrow, krow) * self.scale

        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)  # 参差结构+init put_v
        xx = self.proj(xx)
        xx = self.sigmoid(xx) * qkv
        return xx
class EnDecoderModel(nn.Module):
    def __init__(self, n_classes, backbone):
        super(EnDecoderModel, self).__init__()
        if backbone == 'segb4':
            self.backboner = mit_b4()
            self.backboned = mit_b4()
        elif backbone == 'segb2':
            self.backboner = mit_b2()
            self.backboned = mit_b2()
        #############################################
        self.con4 = convbnrelu(512, 320, k=1, s=1, p=0)
        self.con3 = convbnrelu(320, 128, k=1, s=1, p=0)
        self.con2 = convbnrelu(128, 64, 1, 1)
        self.c11 = convbnrelu(64, n_classes, 1, 1)


        self.rd_fusion34 = AyC_crossfusion(320)
        self.rd_fusion234 = AyC_crossfusion(128)
        self.rd_fusion1234 = AyC_crossfusion(64)

        self.f4_p = SalHead(320, n_classes)
        self.f3_p = SalHead(128, n_classes)
        self.f2_p = SalHead(64, n_classes)
        self.f2_edgep = SalHead(64, 1)
        self.f3_edgep = SalHead(128, 1)


        self.Decon_out1 = pp_upsample(64, 64)
        self.Decon_out2 = pp_upsample(64, 64)

        self.Decon_out320 = pp_upsample(320, 320)

        self.Decon_out128 = pp_upsample(128, 128)

        self.Decon_out64 = pp_upsample(64, 64)

        self.att1 = Sea_Attention(64, key_dim=8, num_heads=4, attn_ratio=2,
                                  activation=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True))
        self.att2 = Sea_Attention(128, key_dim=8, num_heads=8, attn_ratio=2,
                                  activation=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True))
        self.att3 = Sea_Attention(320, key_dim=16, num_heads=8, attn_ratio=2,
                                       activation=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True))
        self.att4 = Sea_Attention(512, key_dim=20, num_heads=8, attn_ratio=2,
                                       activation=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True))
        self.sig = nn.Sigmoid()
        self.con64 = nn.Conv2d(64, 64, 1, 1)
        self.con128 = nn.Conv2d(128, 128, 1, 1)
        # self.con192 = nn.Conv2d(192, 192, 1, 1)

    def forward(self, rgb, dep):

        features_rgb = self.backboner(rgb)
        features_dep = self.backboned(dep)
        features_rlist = features_rgb[0]
        features_dlist = features_dep[0]

        rf1 = features_rlist[0]
        rf2 = features_rlist[1]
        rf3 = features_rlist[2]
        rf4 = features_rlist[3]

        df1 = features_dlist[0]
        df2 = features_dlist[1]
        df3 = features_dlist[2]
        df4 = features_dlist[3]
        FD_pervise = []
        #############################################
        # fd1 = rf1 + df1
        # FD1 = self.att1(fd1)
        rf1a = self.att1(rf1)
        df1a = self.att1(df1)
        FD1 = rf1a + df1a
        #############################################
        # Edge_out = []
        # x_size1 = FD1.size()
        # FD1_canny = self.edge_canny(FD1, x_size1)
        # FD1_cannyout = FD1 + FD1_canny
        # FD1_cannyout = FD1 * self.sig(self.con64(FD1_cannyout))
        # Edge_out.append(self.f2_edgep(FD1_cannyout))
        #############################################
        # fd2 = rf2 + df2
        # FD2 = self.att2(fd2)
        rf2a = self.att2(rf2)
        df2a = self.att2(df2)
        FD2 = rf2a + df2a
        ##############################################
        # x_size2 = FD2.size()
        # FD2_canny = self.edge_canny(FD2, x_size2)
        # FD2_cannyout = FD2 + FD2_canny
        # FD2_cannyout = FD2 * self.sig(self.con128(FD2_cannyout))
        # Edge_out.append(self.f3_edgep(FD2_cannyout))
        ##############################################
        # pre_Edge = self.sig(self.con192(torch.cat((FD1_cannyout, self.Decon_out128(FD2_cannyout)), 1)))
        # Edge_out.append(self.fout_edgep(pre_Edge))
        ##############################################
        # fd3 = rf3 + df3
        # FD3 = self.att3(fd3)
        rf3a = self.att3(rf3)
        df3a = self.att3(df3)
        FD3 = rf3a + df3a
        #############################################
        #############################################
        # fd4 = rf4 + df4
        # FD4 = self.att4(fd4)
        rf4a = self.att4(rf4)
        df4a = self.att4(df4)
        FD4 = rf4a + df4a
        ##############################################
        ##############################################
        FD4 = self.con4(FD4)
        FD4_p = self.f4_p(FD4)
        FD_pervise.append(FD4_p)        # FD_p[0]_c:320->41

        FD4_2 = self.Decon_out320(FD4)

        FD34 = self.rd_fusion34(FD3, FD4_2)
        FD34 = self.con3(FD34)
        FD3_p = self.f3_p(FD34)
        FD_pervise.append(FD3_p)        # FD_p[1]_c:128->41
        FD34_2 = self.Decon_out128(FD34)

        # FD234 = rf2 + FD34_2
        # FD234 = FD2 + FD34_2
        FD234 = self.rd_fusion234(FD2, FD34_2)
        FD234 = self.con2(FD234)
        FD2_p = self.f2_p(FD234)
        FD_pervise.append(FD2_p)        # FD_p[2]_c:64->41
        FD234_2 = self.Decon_out64(FD234)


        out = self.rd_fusion1234(FD1, FD234_2)
        out = self.Decon_out1(out)
        out = self.Decon_out2(out)
        out = self.c11(out)

        # return out, FD_pervise, features_rlist, Edge_out, FD2_canny, FD2_cannyout
        return out, FD_pervise, features_rlist

    # def edge_canny(self, img, x_size):
    #     x_size = img.size()
    #     img_arr = img.cpu().detach().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
    #     canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
    #     for i in range(x_size[0]):
    #         canny[i] = cv2.Canny(img_arr[i], 10, 100)
    #     canny = torch.from_numpy(canny).cuda().float()
    #     return canny

    def load_pre(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.backboner.load_state_dict(new_state_dict3, strict=False)
        self.backboned.load_state_dict(new_state_dict3, strict=False)
        print('self.backboner loading', 'self.backboned loading')
        # print('self.backbone loading')

if __name__ == '__main__':
     import os
     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
     device = torch.device('cuda')
     rgb = torch.randn(1, 3, 480, 640).to(device)
     dep = torch.randn(1, 3, 480, 640).to(device)
     model = EnDecoderModel(n_classes=41, backbone='segb4').to(device)
     out = model(rgb, dep)
     # print('out--', out[0].shape)
     # for i in out:
     #     print('i',i.shape)
     print('out[0]输出结果：', out[0].shape)
     for i in out[3]:
        print('FDFD[3]输出结果：', i.shape)
     # print('out[2][0]', out[2][0].shape)
     #####################################################################################
     # print('****************************************')
     # print('参数量统计如下------------:')
     from model.toolbox.models.A1project2.FLOP import CalParams

     CalParams(model, rgb, dep)
     print('Total params % .2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
     ######################################################################################
        ##B1_shareparam###FLOPs: 130.470G
        ##################Params: 16.675M
        ##################Total params  16.69M
        # ##model2_segb4(FD=att_r+att_d)##################
        # [Statistics Information]
        # FLOPs: 129.454G
        # Params: 125.832M
        #  ####################
        # Total params  125.86M
