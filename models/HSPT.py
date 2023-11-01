import torch
import torch.nn as nn
from timm.models.layers import DropPath
from models.Pooling import attn_pool


def _gumbel_sigmoid(logits, tau=1, hard=True, eps=1e-10, training=True, threshold=0.5):
    if training:
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else:
        y_soft = logits.sigmoid()

    if hard:
        y_hard = torch.zeros_like(logits,
                                  memory_format=torch.legacy_contiguous_format).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def get_select_policy(policy, ratio):
    random_p = torch.empty_like(policy).fill_(ratio).bernoulli() + policy * 0.0
    return random_p


class HeadSelectBlock(nn.Module):
    def __init__(self, dim_in, num_heads, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.mlp_head = nn.Linear(dim_in, num_heads, bias=bias)
        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold
        self.add_noise = True
        self.head_dim = dim_in // num_heads
        self.random_policy = False
        self.random_head = False
        self.random_head_ratio = 1.

    def forward(self, x):
        bsize = x.shape[0]
        logits = self.mlp_head(x)
        sample = _gumbel_sigmoid(logits, self.tau, self.is_hard,
                                 threshold=self.threshold, training=self.training)
        if self.random_policy or self.random_head:
            sample = get_select_policy(sample, self.random_head_ratio)
        sample = sample.unsqueeze(-1)

        width_select = sample.expand(-1, -1, self.head_dim)
        width_select = width_select.reshape(bsize, -1, 1)

        return sample, width_select, logits


class DynaLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_heads=6, bias=True, dyna_dim=[True, True], dyna_data=False):
        super(DynaLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features

        self.num_heads = num_heads
        self.width_mult = 1.
        self.dyna_dim = dyna_dim
        self.in_features = in_features
        self.out_features = out_features
        self.use_full_linear = False

        self.dyna_data = dyna_data
        self.count_flops = False

    def forward(self, input, width_select=None, width_specify=None):
        if self.use_full_linear:
            return super().forward(input)

        if self.count_flops:
            if width_select is not None:
                assert width_select.shape[0] == 1
                width_specify = int(width_select.sum().item())
                width_select = None

        if self.dyna_data and width_select is not None:
            assert input.dim() == 3
            assert width_select.dim() == 3
            assert width_select.shape[1] == 1 or width_select.shape[2] == 1

            if width_select.shape[1] == 1:
                input_mask = width_select
            else:
                input_mask = 1
            if width_select.shape[2] == 1:
                output_mask = width_select[..., 0].unsqueeze(1)
            else:
                output_mask = 1
            input = input * input_mask
            result = super().forward(input) * output_mask
            return result

        if width_select is not None:
            weight = self.weight * width_select
            b, n, c = input.shape
            input = input.transpose(1, 2).reshape(1, -1, n)
            weight = weight.view(-1, c, 1)
            if self.bias is None:
                bias = self.bias
            elif width_select.shape[-1] == 1:
                bias = self.bias * width_select.squeeze()
                bias = bias.view(-1)
            else:
                bias = self.bias.unsqueeze(0).expand(b, -1)
                bias = bias.reshape(-1)
            result = nn.functional.conv1d(input, weight, bias, groups=b)
            result = result.view(b, -1, n).transpose(1, 2)
            return result

        if width_specify is not None:
            if self.dyna_dim[0]:
                self.in_features = width_specify
            if self.dyna_dim[1]:
                self.out_features = width_specify

        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias

        return nn.functional.linear(input, weight, bias)


class PoolAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., dyna_data=False,
                 hw_shape=(8, 8), kernel_q=(3, 3), kernel_kv=(3, 3), stride_q=(1, 1), stride_kv=(1, 1),
                 norm_layer=nn.LayerNorm, has_cls_embed=True, residual_pooling=True):
        super().__init__()
        self.count_flops = False
        self.t_ratio = 1

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.query = DynaLinear(dim, dim, bias=qkv_bias, dyna_dim=[False, True], dyna_data=dyna_data)
        self.key = DynaLinear(dim, dim, bias=qkv_bias, dyna_dim=[False, True], dyna_data=dyna_data)
        self.value = DynaLinear(dim, dim, bias=qkv_bias, dyna_dim=[False, True], dyna_data=dyna_data)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = DynaLinear(dim, dim, dyna_dim=[True, True], dyna_data=dyna_data)
        self.proj_drop = nn.Dropout(proj_drop)

        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]
        dim_conv = self.head_dim
        self.hw_shape = hw_shape
        self.has_cls_embed = has_cls_embed
        self.residual_pooling = residual_pooling

        self.pool_q = (
            nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_q,
                stride=stride_q,
                padding=padding_q,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_q) > 0
            else None
        )
        self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None

        self.pool_k = (
            nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_kv) > 0
            else None
        )
        self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        self.pool_v = (
            nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_kv) > 0
            else None
        )
        self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None

    def forward(self, x, mask=None, head_select=None, value_mask_fill=-1e4,
                head_mask=None, width_select=None, width_specify=None, only_head_attn=False):
        B, N, C = x.shape
        if only_head_attn:
            assert head_select is not None
            width_select = None

        q = self.query(x,
                       width_select=width_select,
                       width_specify=width_specify).reshape(B, N, -1, C // self.num_heads).permute(0, 2, 1, 3)

        k = self.key(x,
                     width_select=width_select,
                     width_specify=width_specify).reshape(B, N, -1, C // self.num_heads).permute(0, 2, 1, 3)

        v = self.value(x,
                       width_select=width_select,
                       width_specify=width_specify).reshape(B, N, -1, C // self.num_heads).permute(0, 2, 1, 3)

        q, q_shape = attn_pool(
            q,
            self.pool_q,
            self.hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_q", None),
        )
        k, k_shape = attn_pool(
            k,
            self.pool_k,
            self.hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_k", None),
        )
        v, v_shape = attn_pool(
            v,
            self.pool_v,
            self.hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_v", None),
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.view(B, 1, N, 1).expand_as(attn)
            attn[~mask] = value_mask_fill

        attn = attn.softmax(dim=-1)
        if only_head_attn:
            head_select = head_select.view(*head_select.shape, *([1] * (4 - head_select.dim())))
            eye_mat = attn.new_zeros(attn.shape[-2:])
            eye_mat.fill_diagonal_(1).view(1, 1, *eye_mat.shape)  # (1,1,l,l)
            attn = attn * head_select + eye_mat * (1 - head_select)

        attn_origin = attn
        if head_mask is not None:
            attn = attn * head_mask

        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # res connection
        # ---------------------------------------------------------
        x = attn @ v
        if self.residual_pooling:
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, N, -1)
        # ---------------------------------------------------------

        if width_select is not None:
            width_select = width_select.transpose(-1, -2)
        x = self.proj(x, width_select, width_specify=width_specify)
        x = self.proj_drop(x)

        return x, attn_origin


class MlpBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 dyna_data=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.fc1 = DynaLinear(in_features, hidden_features, dyna_dim=[True, False], dyna_data=dyna_data)
        self.fc2 = DynaLinear(hidden_features, out_features, dyna_dim=[False, False], dyna_data=dyna_data)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None, width_select=None, width_specify=None):
        if mask is not None:
            assert mask.shape[:2] == x.shape[:2]
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            if mask.dtype != x.dtype:
                mask = mask.type_as(x)
        else:
            mask = x.new_ones(x.shape[:2]).unsqueeze(-1)
        x = self.fc1(x, width_select=width_select, width_specify=width_specify)
        x = x * mask
        x = self.act(x)
        x = self.drop(x)
        width_select = None
        x = self.fc2(x, width_select=width_select, width_specify=width_specify)
        x = x * mask
        x = self.drop(x)
        return x


class StepPoolViTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, use_head_select=True, head_tau=5, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 dyna_data=False, head_v2=False, only_head_attn=False, norm_policy=False,
                 hw_shape=(8, 8)
                 ):
        super().__init__()
        self.h_ratio, self.t_ratio = 1., 1.
        self.l_ratio = [1, 1]

        self.norm_policy = None
        if norm_policy and use_head_select:
            self.norm_policy = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.attn = PoolAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            dyna_data=dyna_data, hw_shape=hw_shape)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.ada_head_v2 = head_v2
        self.mlp = MlpBlock(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                            dyna_data=dyna_data)
        self.only_head_attn = only_head_attn and use_head_select
        self.head_select = None
        if use_head_select:
            self.head_select = HeadSelectBlock(dim_in=dim, num_heads=num_heads, tau=head_tau)

    def forward(self, x, mask=None, head_mask=None, width_specify=None):
        if self.norm_policy is not None:
            policy_token = self.norm_policy(x)[:, 0]
        else:
            policy_token = x[:, 0]

        if self.head_select is not None:
            head_select, width_select, head_logits = self.head_select(policy_token)
        else:
            head_select, width_select, head_logits = None, None, None

        if self.only_head_attn:
            assert head_select is not None
            width_select = None
        if width_select is not None:
            width_select_attn = width_select
            if self.ada_head_v2:
                bs = width_select.shape[0]
                width_select_mlp = width_select.expand(-1, -1, int(self.mlp_ratio)).reshape(bs, -1, 1)
            else:
                width_select_mlp = width_select.transpose(-1, -2)
        else:
            width_select_attn, width_select_mlp = [None] * 2

        attn_x, attn_origin = self.attn(self.norm1(x), mask=mask, head_mask=head_mask,
                                        width_select=width_select_attn,
                                        width_specify=width_specify, head_select=head_select,
                                        only_head_attn=self.only_head_attn)

        x = x + self.drop_path(attn_x)
        mlp_x = self.mlp(self.norm2(x), width_select=width_select_mlp, width_specify=width_specify)
        x = x + self.drop_path(mlp_x)

        return x, attn_origin, head_select, head_logits


class HeadSelectPoolTransformer(nn.Module):
    def __init__(self, dim, depth=12, num_heads=8, use_head_select=True, only_head_attn=False,
                 head_tau=5, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, dyna_data=False, ada_head_v2=False,
                 norm_policy=False, keep_layers=0, drop_path_rate=0.):
        super().__init__()
        self.norm = norm_layer(dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            StepPoolViTBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                head_tau=head_tau, drop=drop, attn_drop=attn_drop, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, dyna_data=dyna_data, head_v2=ada_head_v2,
                use_head_select=use_head_select and i >= keep_layers, norm_policy=norm_policy,
                only_head_attn=only_head_attn)
            for i in range(depth)])

    def forward_features(self, x):
        attn_list = []
        hidden_list = []
        head_select_list = []
        head_select_logits_list = []

        def filter_append(target_list, element):
            if element is not None:
                target_list.append(element)

        for blk in self.blocks:
            x, attn, this_head_select, this_head_select_logits = blk(x)
            attn_list.append(attn)
            hidden_list.append(x)
            filter_append(head_select_list, this_head_select)
            filter_append(head_select_logits_list, this_head_select_logits)

        def list2tensor(list_convert):
            if len(list_convert):
                result = torch.stack(list_convert, dim=1)
            else:
                result = None
            return result

        head_select = list2tensor(head_select_list)
        if head_select is not None:
            head_select = head_select.squeeze(-1)
        head_select_logits = list2tensor(head_select_logits_list)

        x = self.norm(x)
        return x, head_select, attn_list, hidden_list, dict(head_select_logits=head_select_logits)

    def forward(self, x, ret_attn_list=False):
        x, head_select, attn_list, hidden_list, select_logtis = self.forward_features(x)
        if ret_attn_list:
            return x, head_select, attn_list, hidden_list, select_logtis
        return x, hidden_list, head_select, select_logtis
