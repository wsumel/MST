# Sheng Wang at Feb 22 2023

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter

###########
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

prj_path = osp.join('/home/wsl/mst-tiny')
add_path(prj_path)
#####################

from lib.models.mst.basevit import ViT


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module, r: int, alpha: int):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        x = self.w(x) + (self.alpha // self.r) * self.w_b(self.w_a(x))
        return x



class _LoRA_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
        alpha: int
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += (self.alpha // self.r) * new_q
        qkv[:, :, -self.dim :] += (self.alpha // self.r) * new_v
        return qkv


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, alpha: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT_timm, self).__init__()

        assert r > 0
        assert alpha > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        # dim = vit_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r,
                alpha
            )
        self.reset_parameters()
        self.lora_vit = vit_model
        self.proj_3d = nn.Linear(num_classes * 30, num_classes)
        # if num_classes > 0:
        #     self.lora_vit.reset_classifier(num_classes=num_classes)
            # self.lora_vit.head = nn.Linear(
            #     self.dim, num_classes)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)

    # def forward(self, x: Tensor) -> Tensor:
    #     x = rearrange(x, "b s c h w -> (b s) c h w", s=30)
    #     x = self.lora_vit(x)
    #     x = rearrange(x, "(b s) d -> b (s d)", s=30)
    #     x = self.proj_3d(x)
    #     return x
    
   


# 自定义ViT类，继承自timm_ViT
class CustomViT(timm_ViT):
    def __init__(self, *args, **kwargs):
        super(CustomViT, self).__init__(*args, **kwargs)
        # 存储每个块的特征
        self.block_features = []

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
            self.block_features.append(x.clone())
        
        return x

        # x = self.norm(x)
        # return x[:, 0]

    def forward(self, x):
        self.block_features.clear()  # 清空之前的特征
        x = self.forward_features(x)
        # x = self.pre_logits(x)
        return x

    def get_block_features(self):
        return self.block_features


if __name__ == "__main__":  # Debug
    
    # 加载原始权重文件
    state_dict = torch.load('/home/wsl/mst-tiny/pretrain/mae_tiny_distill_400e.pth.tar')['model']

    # 创建一个新的状态字典用于存储修正后的键名
    new_state_dict = {}

    # 映射关系：旧键 -> 新键
    key_mapping = {
        "module.model.cls_token": "cls_token",
        "module.model.pos_embed": "pos_embed",
        "module.model.patch_embed.proj.weight": "patch_embed.proj.weight",
        "module.model.patch_embed.proj.bias": "patch_embed.proj.bias",
    }

    # 添加Transformer块的映射关系
    for i in range(12):
        old_prefix = f"module.model.blocks.{i}"
        new_prefix = f"blocks.{i}"

        key_mapping[f"{old_prefix}.norm1.weight"] = f"{new_prefix}.norm1.weight"
        key_mapping[f"{old_prefix}.norm1.bias"] = f"{new_prefix}.norm1.bias"
        key_mapping[f"{old_prefix}.attn.qkv.weight"] = f"{new_prefix}.attn.qkv.weight"
        key_mapping[f"{old_prefix}.attn.qkv.bias"] = f"{new_prefix}.attn.qkv.bias"
        key_mapping[f"{old_prefix}.attn.proj.weight"] = f"{new_prefix}.attn.proj.weight"
        key_mapping[f"{old_prefix}.attn.proj.bias"] = f"{new_prefix}.attn.proj.bias"
        key_mapping[f"{old_prefix}.norm2.weight"] = f"{new_prefix}.norm2.weight"
        key_mapping[f"{old_prefix}.norm2.bias"] = f"{new_prefix}.norm2.bias"
        key_mapping[f"{old_prefix}.mlp.fc1.weight"] = f"{new_prefix}.mlp.fc1.weight"
        key_mapping[f"{old_prefix}.mlp.fc1.bias"] = f"{new_prefix}.mlp.fc1.bias"
        key_mapping[f"{old_prefix}.mlp.fc2.weight"] = f"{new_prefix}.mlp.fc2.weight"
        key_mapping[f"{old_prefix}.mlp.fc2.bias"] = f"{new_prefix}.mlp.fc2.bias"

    # 通用处理
    for old_key, new_key in key_mapping.items():
        if old_key in state_dict:
            new_state_dict[new_key] = state_dict[old_key]



    # # 添加剩余的层归一化和全连接层
    # new_state_dict['norm.weight'] = state_dict['module.model.norm.weight']
    # new_state_dict['norm.bias'] = state_dict['module.model.norm.bias']
    # # new_state_dict['head.weight'] = state_dict['module.model.decoder_norm.weight']
    # # new_state_dict['head.bias'] = state_dict['module.model.decoder_norm.bias']


    img = torch.randn(2*20, 3, 224, 224)
    # model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    
    # 初始化自定义的ViT模型
    model = CustomViT(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                    norm_layer=torch.nn.LayerNorm)

    # 初始化自定义的ViT模型
    model = CustomViT(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                    norm_layer=torch.nn.LayerNorm)

    # 将修正后的新状态字典加载到模型中
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    if missing_keys:
        print(f"缺失的键: {missing_keys}")
    if unexpected_keys:
        print(f"意外的键: {unexpected_keys}")

    # 打印模型结构以验证
    # print(model)

    # 测试输入
    x = torch.randn(1, 3, 224, 224)  # 示例输入张量



    lora_vit = LoRA_ViT_timm(vit_model=model, r=4, num_classes=10,alpha=4)
    pred = lora_vit(img)
    # pred = model(img)
    print(pred.shape)
