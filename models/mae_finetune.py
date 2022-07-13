# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from models.nextvlad import NeXtVLAD


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)

        # [3, B, head, Patch, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # [B C H W]->[B D H2 W2]->[B D P]->[B P D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        init_scale=0.0,
        pooling="nextvlad",
        mask_ratio=0.0,
        nextvlad_lamb=4,
        nextvlad_cluster=64,
        nextvlad_groups=8,
    ):
        super().__init__()
        self.num_classes = num_classes

        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        # patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # mask
        mask_ratio = mask_ratio
        num_patches = self.patch_embed.num_patches
        self.num_mask = int(mask_ratio * num_patches)
        self.patch_idx = [i for i in range(num_patches)]
        self.pooling = pooling
        if pooling == "nextvlad":
            self.before_norm = norm_layer(embed_dim)
            self.after_norm = norm_layer(embed_dim)
            self.nextvlad = NeXtVLAD(
                feature_size=embed_dim,
                max_frames=num_patches,
                nextvlad_cluster_size=nextvlad_cluster,
                lamb=nextvlad_lamb,
                groups=nextvlad_groups,
            )
            nextvlad_dim = (
                nextvlad_cluster * nextvlad_lamb * embed_dim // nextvlad_groups
            )
            self.dim_reduce = nn.Linear(nextvlad_dim, embed_dim)
            self.act = nn.ReLU()
        elif pooling == "mean":
            self.fc_norm = norm_layer(embed_dim)
        else:
            assert False

        # head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # pos embedding
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        # dropout rate
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )

        # initialize parameters
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)
        if pooling == "cls":
            trunc_normal_(self.cls_token, std=0.02)
        elif pooling == "transformer":
            trunc_normal_(self.trans_cls_token, std=0.02)
        trunc_normal_(self.head.weight, std=0.02)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "trans_cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def transformer_features(self, x):
        x = self.patch_embed(x)
        B, P, D = x.size()

        if self.pos_embed is not None:
            x = (
                x
                + self.pos_embed.expand(B, -1, -1)
                .type_as(x)
                .to(x.device)
                .clone()
                .detach()
            )
        x = self.pos_drop(x)

        # restore
        if self.pooling == "nextvlad":
            restore = torch.zeros_like(x)  # [B P D]

        # masking
        if self.training:
            num_mask = self.num_mask
        else:
            num_mask = 0
        mask = torch.zeros(B, P, dtype=bool, device=x.device)  # [B P]

        for row in mask:
            mask_idx = random.sample(self.patch_idx, num_mask)
            row[mask_idx] = 1
        x = x[~mask].reshape(B, -1, D)  # ~mask means visible

        # add cls token
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # forward
        for blk in self.blocks:
            x = blk(x)

        # nextvlad restore
        if self.pooling == "nextvlad":
            restore[~mask] = x.reshape(-1, D)  # [BP D]
            return restore, ~mask
        return x, None

    def forward_features(self, x):
        x, mask = self.transformer_features(x)

        # pooling
        if self.pooling == "nextvlad":
            x = self.before_norm(x)
            x = self.nextvlad(x, mask)
            x = self.dim_reduce(x)
            x = self.after_norm(x)
            x = self.act(x)
            return x
        elif self.pooling == "mean":
            return self.fc_norm(x.mean(1))
        else:
            assert False

    def forward(self, x):
        rep = self.forward_features(x)
        ce = self.head(rep)
        return ce


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
