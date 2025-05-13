import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        B = x.shape[0]
        # Project patches
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop_path = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TransformerMLP(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self‑attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_out)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N, D)
        # prepend class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # (B, D)
        logits = self.head(cls_out)
        return logits


class LitViT(pl.LightningModule):
    """PyTorch Lightning wrapper for VisionTransformer."""

    def __init__(self,  
                 learning_rate=1e-3,
                 weight_decay=1e-5,
                 momentum=0.95,
                 optimizer='AdamW',
                 **vit_kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.momentum = momentum

        if self.optimizer_name == 'Muon':
            self.automatic_optimization = False
        else:
            self.automatic_optimization = True

        self.save_hyperparameters()
        self.model = VisionTransformer(**vit_kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vit_kwargs['num_classes'])
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vit_kwargs['num_classes'])
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vit_kwargs['num_classes'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        
        if not self.automatic_optimization:
            optimizers = self.optimizers()
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
            for opt in optimizers:
                opt.zero_grad()
            self.manual_backward(loss)
            for opt in optimizers:
                opt.step()
                
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        self.test_acc(logits, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
        
        return loss

    def configure_optimizers(self):
        if self.optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            return optimizer
        elif self.optimizer_name == 'Muon':
            from muon import Muon
            
            muon_params = []
            adamw_params = []
            
            # Identify the head layer (classification layer)
            head_layer = self.model.head
            
            # Add head parameters to AdamW
            adamw_params.extend(list(head_layer.parameters()))
            
            # Process all other parameters
            for name, module in self.model.named_children():
                if module is not head_layer:
                    for p in module.parameters():
                        if p.ndim >= 2:
                            print(f"Muon param: {p.shape}")
                            muon_params.append(p)
                        else:
                            print(f"AdamW param: {p.shape}")
                            adamw_params.append(p)
            
            optimizers = []
            
            if muon_params:
                optimizers.append(Muon(muon_params, lr=self.learning_rate, momentum=self.momentum))
                
            if adamw_params:
                optimizers.append(torch.optim.AdamW(
                    adamw_params, 
                    lr=0.001, 
                    betas=(0.90, 0.95),
                    weight_decay=self.weight_decay
                ))
            
            if not optimizers:
                raise ValueError("No parameters were assigned to the optimizer")
            
            return optimizers
