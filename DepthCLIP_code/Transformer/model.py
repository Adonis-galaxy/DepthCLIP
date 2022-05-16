import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F
import pdb
from .vision_transformer import *
def showpic(data,x,y):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(x,y))
    data1 = data.detach().cpu()
    for i in range(x*y):
        ax = fig.add_subplot(y, x, i+1)
        plt.axis('off')
        plt.imshow(data1[0,i])
    plt.subplots_adjust(wspace =0.1, hspace =0.1)
    plt.savefig('./show')


activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook
attention = {}
def get_attention(name, dataset):
    def hook(module, input, output):
        x = input[0]
        head = 1
        #head = module.num_heads
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, head, C // head).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0],qkv[1],qkv[2],)  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn
    return hook


class ASTransformer(nn.Module):
    def __init__(self,args, features=[96, 192, 384, 768],size=[384, 384],hooks=[2, 5, 8, 11],vit_features=768,use_readout="ignore",start_index=1,patch_embed=None):
        super().__init__()
        self.args = args
        self.model = VisionTransformer(
        img_size=size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),patch_embed=patch_embed)
        self.model.blocks[hooks[0]].register_forward_hook(get_activation("layer1"))
        self.model.blocks[hooks[1]].register_forward_hook(get_activation("layer2"))
        self.model.blocks[hooks[2]].register_forward_hook(get_activation("layer3"))
        self.model.blocks[hooks[3]].register_forward_hook(get_activation("layer4"))

        self.activations = activations

        self.model.blocks[hooks[3]].attn.register_forward_hook(get_attention("attn_4",args.dataset))
        self.attention = attention
        self.layer1_conv = nn.Conv2d(in_channels=vit_features,out_channels=features[0],kernel_size=1,stride=1,padding=0)
        self.layer1_up = nn.Upsample(scale_factor=8, mode='bicubic', align_corners=True)

        self.layer2_conv = nn.Conv2d(in_channels=vit_features,out_channels=features[1],kernel_size=1,stride=1,padding=0,)
        self.layer2_up = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)

        self.layer3_conv = nn.Conv2d(in_channels=vit_features,out_channels=features[2],kernel_size=1,stride=1,padding=0)
        self.layer3_up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

        self.layer4_conv = nn.Conv2d(in_channels=vit_features,out_channels=features[3],kernel_size=1,stride=1,padding=0,)
        self.patch_size = [16, 16]


    def get_feature(self, x, img):
        b, c, h, w = x.shape
        pos_embed = self.model.pos_embed
        B = x.shape[0]

        x, feature = self.model.patch_embed(x)

        cls_tokens = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed
        x = self.model.pos_drop(x)

        for i,blk in enumerate(self.model.blocks):
            if i in [2,5,8]:
                x,feature = blk(x,img[(i//3)])
            else:
                x = blk(x)

        x = self.model.norm(x)

        return x, feature

    def forward(self, x, img):
        b, c, h, w = x.shape
        _, feature = self.get_feature(x, img)

        layer_1 = self.activations["layer1"][0]
        feature1 = self.activations["layer1"][1]
        layer_2 = self.activations["layer2"][0]
        feature2 = self.activations["layer2"][1]
        layer_3 = self.activations["layer3"][0]
        feature3 = self.activations["layer3"][1]
        layer_4 = self.activations["layer4"]
        attention4 = self.attention['attn_4'][:,:,1:,1:]            
        layer_1 = layer_1[:,1:].transpose(1,2).contiguous()
        layer_2 = layer_2[:,1:].transpose(1,2).contiguous()
        layer_3 = layer_3[:,1:].transpose(1,2).contiguous()
        layer_4 = layer_4[:,1:].transpose(1,2).contiguous()

        unflatten = nn.Unflatten(2,torch.Size([h // self.patch_size[1],w // self.patch_size[0],]))

        layer_1 = self.layer1_up(self.layer1_conv(unflatten(layer_1)))
        layer_2 = self.layer2_up(self.layer2_conv(unflatten(layer_2)))
        layer_3 = self.layer3_up(self.layer3_conv(unflatten(layer_3)))
        layer_4 = self.layer4_conv(unflatten(layer_4))

        return [layer_1, layer_2, layer_3, layer_4],attention4,[feature1,feature2,feature3]


