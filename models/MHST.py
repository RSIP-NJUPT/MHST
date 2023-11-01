import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.HSPT import HeadSelectPoolTransformer
from models.PyConv2D import get_pyconv


def img2seq(x):
    [b, c, h, w] = x.shape
    x = x.reshape((b, c, h * w))
    return x


def seq2img(x):
    [b, c, d] = x.shape
    p = int(d ** .5)
    x = x.reshape((b, c, p, p))
    return x


class LiDAR_Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(LiDAR_Encoder, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = get_pyconv(inplans=in_channels, planes=32, stride=1,
                                pyconv_kernels=[3, 5, 7, 9], out_planes_div=[4, 4, 4, 4], pyconv_groups=[1, 1, 1, 1])
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = get_pyconv(inplans=32, planes=out_channels, stride=1,
                                pyconv_kernels=[3, 5, 7, 9], out_planes_div=[4, 4, 4, 4], pyconv_groups=[1, 1, 1, 1])
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x23):
        x = self.conv1(x23)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class HSI_Encoder(nn.Module):
    def __init__(self, in_channels_3d=1, out_channels_3d=16,
                 in_depth_3d=144, out_channels_2d=64):
        super(HSI_Encoder, self).__init__()
        self.relu = nn.ReLU()

        # 3d
        self.conv1 = nn.Conv3d(in_channels=in_channels_3d, out_channels=out_channels_3d, kernel_size=(11, 3, 3),
                               stride=(3, 1, 1), padding=(5, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channels_3d)

        self.conv2_1 = nn.Conv3d(in_channels=out_channels_3d, out_channels=out_channels_3d // 4, kernel_size=(1, 1, 1),
                                 stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(in_channels=out_channels_3d, out_channels=out_channels_3d // 4, kernel_size=(3, 1, 1),
                                 stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(in_channels=out_channels_3d, out_channels=out_channels_3d // 4, kernel_size=(5, 1, 1),
                                 stride=(1, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(in_channels=out_channels_3d, out_channels=out_channels_3d // 4, kernel_size=(11, 1, 1),
                                 stride=(1, 1, 1), padding=(5, 0, 0))
        self.bn2 = nn.BatchNorm3d(out_channels_3d)

        self.conv3 = nn.Conv3d(in_channels=out_channels_3d, out_channels=out_channels_3d, kernel_size=(3, 3, 3),
                               stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(out_channels_3d)

        # 2d
        self.in_channels_2d = int((in_depth_3d + 2) / 3) * out_channels_3d
        self.conv4 = get_pyconv(inplans=self.in_channels_2d, planes=out_channels_2d, stride=1,
                                pyconv_kernels=[3, 5, 7, 9], out_planes_div=[4, 4, 4, 4], pyconv_groups=[1, 2, 4, 8])
        self.bn4 = nn.BatchNorm2d(out_channels_2d)

        self.conv5 = nn.Conv2d(in_channels=out_channels_2d, out_channels=out_channels_2d, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(out_channels_2d)

        self.pool = nn.MaxPool2d(2)

    def forward(self, hsi_img):
        # 3d
        x1 = self.conv1(hsi_img)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = torch.cat((self.conv2_1(x1), self.conv2_2(x1), self.conv2_3(x1), self.conv2_4(x1)), dim=1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        out_3d = self.relu(x3)

        # 2d
        x = rearrange(out_3d, 'b c h w y ->b (c h) w y')

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.pool(x)

        return x


class Classifier(nn.Module):
    def __init__(self, Classes):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = get_pyconv(inplans=64, planes=32, stride=1,
                                pyconv_kernels=[3, 5], out_planes_div=[2, 2], pyconv_groups=[2, 2])
        self.bn1 = nn.BatchNorm2d(32)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x_out = F.softmax(x, dim=1)

        return x_out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# MLP
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class MHST(nn.Module):
    def __init__(self, l1, l2, patch_size, num_patches, num_classes,
                 encoder_embed_dim, en_depth, en_heads,
                 mlp_dim, dim_head=16, dropout=0., emb_dropout=0.,
                 coefficient_hsi=0.5, coefficient_vit=0.5,
                 hsp_vit_depth=12, hsp_vit_num_heads=16,
                 vit_qkv_bias=True, use_head_select=True, head_tau=5,
                 mlp_ratio=4., attnproj_mlp_drop=0., attn_drop=0.):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.ada_en_depth = hsp_vit_depth

        self.coefficient_hsi = coefficient_hsi
        self.coefficient_lidar = 1 - self.coefficient_hsi
        self.coefficient_vit = coefficient_vit
        self.coefficient_cnn = 1 - self.coefficient_vit

        self.hsi_encoder = HSI_Encoder(in_channels_3d=1, in_depth_3d=l1, out_channels_3d=16, out_channels_2d=64)
        self.lidar_encoder = LiDAR_Encoder(in_channels=l2, out_channels=64)

        self.weight_hsi = torch.nn.Parameter(torch.Tensor([self.coefficient_hsi]))
        self.weight_lidar = torch.nn.Parameter(torch.Tensor([self.coefficient_lidar]))

        self.encoder_embedding = nn.Linear(((patch_size // 2) * 1) ** 2, self.patch_size ** 2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1, encoder_embed_dim))
        self.en_transformer = Transformer(encoder_embed_dim, en_depth, en_heads, dim_head, mlp_dim, dropout)

        self.HeadSelectViT = HeadSelectPoolTransformer(dim=encoder_embed_dim, depth=self.ada_en_depth,
                                                       num_heads=hsp_vit_num_heads, use_head_select=use_head_select,
                                                       qkv_bias=vit_qkv_bias, head_tau=head_tau,
                                                       mlp_ratio=mlp_ratio, drop=attnproj_mlp_drop, attn_drop=attn_drop)

        self.to_latent = nn.Identity()

        self.pyconv_classifier = Classifier(num_classes)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(encoder_embed_dim),
            nn.Linear(encoder_embed_dim, num_classes)
        )

        self.vit_cls_coefficient = torch.nn.Parameter(torch.Tensor([self.coefficient_vit]))
        self.cnn_cls_coefficient = torch.nn.Parameter(torch.Tensor([self.coefficient_cnn]))

    def encoder(self, x1, x2):
        x_hsi = self.hsi_encoder(x1)
        x_lidar = self.lidar_encoder(x2)
        x = self.weight_hsi * x_hsi + self.weight_lidar * x_lidar

        x = x.flatten(2)
        x_cnn = self.encoder_embedding(x)

        x_cnn = torch.einsum('nld->ndl', x_cnn)
        b, n, _ = x_cnn.shape
        x = x_cnn + self.encoder_pos_embed[:, 1:, :]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.encoder_pos_embed[:, :1]
        x = self.dropout(x)

        x = self.en_transformer(x, mask=None)
        x, hidden_list, head_select, select_logtis = self.HeadSelectViT(x)

        return x, x_cnn

    def classifier(self, x, x_cnn):
        x = self.to_latent(x[:, 0])

        x_cls1 = self.mlp_head(x)
        x_cls1 = F.softmax(x_cls1, dim=1)

        x_cnn = torch.einsum('ndl->nld', x_cnn)
        x_cnn = seq2img(x_cnn)
        x_cls2 = self.pyconv_classifier(x_cnn)

        x_cls = x_cls1 * self.vit_cls_coefficient + x_cls2 * self.cnn_cls_coefficient
        return x_cls

    def forward(self, img_hsi, img_lidar):
        x_vit, x_cnn = self.encoder(img_hsi, img_lidar)
        x_cls = self.classifier(x_vit, x_cnn)
        return x_cls
