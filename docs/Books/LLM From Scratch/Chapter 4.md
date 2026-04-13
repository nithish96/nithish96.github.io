# 4. Implementing a GPT Model from Scratch

## Overview

This chapter presents the implementation of a Generative Pretrained Transformer (GPT) model architecture designed for generating human-like text. The discussion builds on the concepts covered in earlier chapters, such as token embeddings, attention mechanisms, and transformer blocks, to assemble the GPT architecture. The chapter also outlines methods for scaling up the model and computing its parameter requirements.

This chapter covers 

1. Coding a GPT-like LLM architecture.
2. Stabilizing training with layer normalization.
3. Constructing transformer blocks with attention and feed-forward layers.
4. Computing the number of parameters and storage requirements of GPT models

---

## Key Components

### 4.1 Coding a GPT Architecture

- Generative Pretrained Transformers (GPTs) are models that predict the next token in a sequence based on prior tokens, operating in a unidirectional left-to-right manner.
- GPTs repeat core components (transformer blocks) to build deeper models.
- The architecture includes:
  - Input tokenization.
  - Token and positional embeddings.
  - Multi-head attention and feed-forward layers.
  - Final layer normalization and output projection.

### 4.2 Layer Normalization

- Training deep neural networks with many layers can sometimes prove challenging due to problems like vanishing or exploding gradients. 
- The main idea behind layer normalization is to adjust the activations of a neural network layer to have a mean of 0 and a variance of 1, also known as unit variance. 
- Layer normalization stabilizes neural network training by ensuring consistent activation distributions.
- In GPT, normalization is applied before multi-head attention and feed-forward layers (pre-layer normalization).
- This strategy improves convergence and training dynamics compared to post-layer normalization.

``` python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

### 4.3 Feed-Forward Networks with GELU Activations

- Feed-forward networks (FFNs) process the output of attention layers.
- Each FFN consists of two fully connected layers separated by a non-linear activation function, such as GELU (Gaussian Error Linear Unit) and SwiGLU (Swish-gated linear unit).
- GELU and SwiGLU enables smoother gradients than ReLU, enhancing training stability and performance.

``` python 
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
)
    def forward(self, x):
        return self.layers(x)

```
### 4.4 Shortcut Connections

- Shortcut connections help gradients flow across layers, mitigating the vanishing gradient problem in deep models.
- In GPT, shortcuts add input representations back to their transformed versions after attention and feed-forward processing.
- These connections improve the training of very deep architectures.

### 4.5 Transformer Blocks

- The transformer block integrates all core components:
  - Multi-head attention for contextual understanding.
  - Feed-forward layers for token-wise transformations.
  - Layer normalization for stability.
  - Shortcut connections for better gradient flow.
- Multiple transformer blocks are stacked to create the GPT model.

``` python 
from chapter03 import MultiHeadAttention
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut      # add original input
        shortcut = x         # shortcut for ff block 
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut      # add original input back 
        return x
```

### 4.6 Coding the GPT Model

- The GPT model uses token embeddings, positional embeddings, and dropout for regularization.
- The sequence of operations:
    1. Tokenized input is embedded into dense vectors.
    2. Positional embeddings are added to incorporate sequence order.
    3. Data passes through stacked transformer blocks.
    4. Final layer normalization prepares data for output.
    5. A linear projection maps outputs to vocabulary size for token prediction.

``` python 
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

### 4.7 Generating Text

- Text generation is an iterative process:
  1. Input tokens are processed by the GPT model.
  2. The model predicts the next token, appending it to the input sequence.
  3. The updated sequence is fed back into the model for the next prediction.
- This continues until a predefined stopping condition (e.g., end-of-text token).

``` python

def generate_text_simple(model, idx,                 
                         max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]    
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)    
        idx = torch.cat((idx, idx_next), dim=1)     
return idx

```

---

## Summary

Chapter 4 integrates previous concepts to assemble a complete GPT model. It emphasizes architectural design, stability techniques, and scalability considerations. The chapter concludes with the text generation process, preparing for training and evaluation in subsequent chapters.
