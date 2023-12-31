import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, Conv2d, LayerNorm, Softmax
from torch.nn.modules.utils import _pair

import numpy as np
import copy
import math
from os.path import join as pjoin

import src.configs as configs


ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"
FC_0 = "MlpBlock_3/Dense_0/"
FC_1 = "MlpBlock_3/Dense_1/"
ATTENTION_NORM = "LayerNorm_0/"
MLP_NORM = "LayerNorm_2/"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def compute_mask(tensor, threshold):
    return (tensor.abs() > threshold).float()


class ChannelImportance(nn.Module):
    def __init__(self, num_channels):
        super(ChannelImportance, self).__init__()
        
        self.importance = nn.Parameter(torch.ones(num_channels))
        
    def forward(self, x):
        return x.mul(self.importance)


class Attention(nn.Module):
    def __init__(self, config, visualize, prune, ratio):
        super(Attention, self).__init__()
        self.visualize = visualize
        self.prune = prune
        self.ratio = ratio
        self.masks_channel = None
        self.masks_context = None
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.channel_importance = ChannelImportance(self.all_head_size)
        self.context_importance = ChannelImportance(self.all_head_size)
        
        self.query = Linear(config.hidden_size, self.all_head_size) # why to this dim
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        
        self.softmax = Softmax(dim=-1)
             
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states): # check shapes            
        hidden_states = self.channel_importance(hidden_states) # for pruning # it this part of the code needed during the inference?
        # if self.prune:
        #     if self.masks_channel is None:
        #         print("Computing mask for channels")
        #         scores = sorted(self.channel_importance.importance.detach())
        #         threshold_indx = int(self.ratio * len(scores))
        #         threshold = scores[threshold_indx]
        #         self.masks_channel = self.channel_importance.importance > threshold
        #     hidden_states = hidden_states * self.masks_channel
            
        if self.prune:
            if self.masks_channel is None:
                print("Computing mask for channels")
                threshold = torch.kthvalue(hidden_states.abs().view(-1), int(self.ratio * hidden_states.nelement()))[0]
                self.masks_channel = compute_mask(hidden_states, threshold)
            hidden_states = hidden_states * self.masks_channel
            
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_probs = self.softmax(attention_scores)
        
        weights = attention_probs if self.visualize else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        context_layer = self.channel_importance(context_layer) # for pruning
        
        # if self.prune:
        #     if self.masks_context is None:
        #         print("Computing mask for context")
        #         scores = sorted(self.context_importance.importance.detach())
        #         threshold_indx = int(self.ratio * len(scores))
        #         threshold = scores[threshold_indx]
        #         self.masks_context = self.context_importance.importance > threshold
        #     hidden_states = hidden_states * self.masks_context
            
        if self.prune:
            if self.masks_context is None:
                print("Computing mask for context")
                threshold = torch.kthvalue(context_layer.abs().view(-1), int(self.ratio * context_layer.nelement()))[0]
                self.masks_context = compute_mask(context_layer, threshold)
            context_layer = context_layer * self.masks_context
            
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
    

class Mlp(nn.Module):
    def __init__(self, config, prune, ratio):
        super(Mlp, self).__init__()
        self.prune = prune
        self.ratio = ratio
        self.masks_mlp1 = None
        self.masks_mlp2 = None
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])
        
        self.mlp_importance1 = ChannelImportance(config.hidden_size)
        self.mlp_importance2 = ChannelImportance(config.transformer["mlp_dim"])
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        
    def forward(self, x):
        x = self.mlp_importance1(x) # for pruning
        # if self.prune:
        #     if self.masks_mlp1 is None:
        #         print("Computing mask for FC1")
        #         scores = sorted(self.mlp_importance1.importance.detach())
        #         threshold_indx = int(self.ratio * len(scores))
        #         threshold = scores[threshold_indx]
        #         self.masks_mlp1 = self.mlp_importance1.importance > threshold
        #     x = x * self.masks_mlp1
            
        if self.prune:
            if self.masks_mlp1 is None:
                print("Computing mask for FC1")
                threshold = torch.kthvalue(x.abs().view(-1), int(self.ratio * x.nelement()))[0]
                self.masks_mlp1 = compute_mask(x, threshold)
            x = x * self.masks_mlp1
         
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.mlp_importance2(x) # for pruning
        # if self.prune:
        #     if self.masks_mlp2 is None:
        #         print("Computing mask for FC2")
        #         scores = sorted(self.mlp_importance2.importance.detach())
        #         threshold_indx = int(self.ratio * len(scores))
        #         threshold = scores[threshold_indx]
        #         self.masks_mlp2 = self.mlp_importance2.importance > threshold
        #     x = x * self.masks_mlp2
        
        if self.prune:
            if self.masks_mlp2 is None:
                print("Computing mask for FC2")
                threshold = torch.kthvalue(x.abs().view(-1), int(self.ratio * x.nelement()))[0]
                self.masks_mlp2 = compute_mask(x, threshold)
            x = x * self.masks_mlp2
        
        x = self.fc2(x)
        x = self.dropout(x)
        return x
            
            
class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)  
        
        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        self.dropout = Dropout(config.transformer["dropout_rate"])
        
    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
        

class Block(nn.Module):
    def __init__(self, config, visualize, prune, ratio):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config, prune, ratio)
        self.attn = Attention(config, visualize, prune, ratio)
        
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        
        return x, weights
    
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            

class Encoder(nn.Module):
    def __init__(self, config, visualize, prune, ratio, **block_kwargs):
        super(Encoder, self).__init__()
        self.visualize = visualize
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, visualize, prune, ratio, **block_kwargs)
            self.layer.append(copy.deepcopy(layer))
            
    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.visualize:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
    
    
class Transformer(nn.Module):
    def __init__(self, config, img_size, visualize, prune, ratio, quantize=False, half=False, **kwargs):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        config.quantize = quantize
        config.half = half
        self.encoder = Encoder(config, visualize, prune, ratio, **kwargs)
        
    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights
    

class ViT(nn.Module):
    def __init__(self, config, img_size=224, num_classes=10, visualize=False, prune=False, ratio=None, **kwargs):
        super(ViT, self).__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier
        
        self.transformer = Transformer(config, img_size, visualize, prune, ratio, **kwargs)
        self.head = Linear(config.hidden_size, num_classes)
        
    def forward(self, x):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])
        
        return logits, attn_weights
    
    def load_from(self, weights):
        with torch.no_grad():
            if self.num_classes == 1000:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())
            
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

    
CONFIGS = {
    'ViT-Ti_16': configs.get_ti16_config(),
}
