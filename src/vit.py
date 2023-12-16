import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 patch_size: int=16,
                 embedding_dim: int=768):
        super().__init__()
        
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        
    def forward(self, x):
        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)
    
    
class MSA(nn.Module):
    def __init__(self,
                 embedding_dim: int=192,
                 num_heads: int=3,
                 dropout: float=0.0):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.msa = nn.MultiheadAttention(embed_dim=embedding_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         batch_first=True)
        
    def forward(self, x):
        x = self.layer_norm(x)
        x, w = self.msa(query=x,
                        key=x,
                        value=x,
                        need_weights=True)
        
        return x
    
class MLP(nn.Module):
    def __init__(self,
                 embedding_dim: int=192,
                 mlp_size: int=768,
                 dropout: float=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self,
                 embedding_dim: int=192,
                 num_heads: int=3,
                 mlp_size: int=768,
                 mlp_dropout: float=0.1,
                 attn_dropout: float=0):
        super().__init__()
        
        self.msa = MSA(embedding_dim=embedding_dim,
                       num_heads=num_heads,
                       dropout=attn_dropout)
        
        self.mlp = MLP(embedding_dim=embedding_dim,
                       mlp_size=mlp_size,
                       dropout=mlp_dropout)
        
    def forward(self, x):
        x = self.msa(x) + x
        x = self.mlp(x) + x
        return x
    
class ViT(nn.Module):
    def __init__(self,
                 img_size: int=224,
                 in_channels: int=3,
                 patch_size: int=16,
                 layers: int=12,
                 embedding_dim: int=192,
                 mlp_size: int=768,
                 num_heads: int=3,
                 attn_dropout: float=0.0,
                 mlp_dropout: float=0.1,
                 emb_dropout: float=0.1,
                 num_classes: int=10):
        super().__init__()
        
        assert img_size % patch_size == 0, f'Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}.'
        
        self.num_patches = (img_size * img_size) // patch_size**2
        
        self.cls_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)
        self.pos_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        self.encoder = nn.Sequential(
            *[Encoder(embedding_dim=embedding_dim,
                      num_heads=num_heads,
                      mlp_size=mlp_size,
                      mlp_dropout=mlp_dropout)
              for _ in range(layers)]
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        cls_token = self.cls_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_embedding + x
        x = self.emb_dropout(x)
        x = self.encoder(x)
        x = self.classifier(x[:, 0]) # TODO mean??
        
        return x
        