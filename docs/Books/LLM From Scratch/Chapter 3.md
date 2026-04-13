# 3. Coding Attention Mechanisms

## Overview

Chapter 3 delves into one of the core concepts underpinning modern large language models (LLMs): attention mechanisms. It explores the evolution from earlier architectures to attention-based models and systematically implements several variants of attention mechanisms.

This chapter covers:

1. The reasons for using attention mechanisms in neural networks
2. A basic self-attention framework, progressing to an enhanced self-attention mechanism
3. A causal attention module that allows LLMs to generate one token at a time
4. Masking randomly selected attention weights with dropout to reduce overfitting
5. Stacking multiple causal attention modules into a multi- head attention module

---

### 3.1 The Problem with Modeling Long Sequences

- Traditional architectures like encoder-decoder RNNs face challenges:
    - They compress the entire input sequence into a single hidden state.
    - The decoder then takes in this hidden state to produce the output
    - Long-range dependencies are lost, leading to poor performance on lengthy sequences.
- Attention mechanisms were developed to address this by allowing models to focus on specific parts of the input dynamically during processing.

### 3.2 Capturing Data Dependencies with Attention

- One major shortcoming in this approach is that the RNN must remember the entire encoded input in a single hidden state before passing it to the decoder. 
- The first major improvement was the Bahdanau attention mechanism (2014), which enabled selective access to input tokens during decoding.
- This inspired the self-attention mechanism of the transformer architecture, enabling LLMs to weigh all input tokens when computing each token's representation.

### 3.3 Attending to Different Parts of the Input with Self-Attention

#### 3.3.1 Simplified Self-Attention

- In its basic form, self-attention computes a context vector for each token in the input sequence, incorporating information from all tokens.
- The mechanism computes attention weights based on the relevance between tokens.

#### Example:
  - Input: `x(1), x(2), ..., x(T)`
  -  When computing the context vector `z(2)`,  attention weights are calculated with respect to input element `x(2)` and all other inputs. 

#### Implementation:
1. Compute attention scores as dot products between token embeddings.
2. Normalize scores using softmax to get attention weights.
3. Compute context vectors as weighted sums of token embeddings.

``` python 
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
```
### 3.4 Implementing Self-Attention with Trainable Weights

- Adds trainable weight matrices:
    - Queries (`Wq`), Keys (`Wk`), and Values (`Wv`).
- The steps involve:
  1. Projecting token embeddings into query, key, and value spaces.
  2. Computing attention scores as dot products of queries and keys.
  3. Normalizing scores with softmax and scaling.
  4. Computing weighted sums of value vectors to get context vectors.

#### Compact Implementation:
- nn.Linear has an optimized weight initialization scheme, contributing to more stable and effective model training.

``` python 
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
```

### 3.5 Hiding Future Words with Causal Attention

- For tasks like text generation, models must only attend to previous tokens to avoid peeking into the future.
- Causal masking ensures that attention weights for future tokens are set to zero.
- This is achieved by:
  - Applying an upper triangular mask to the attention score matrix.
  - Normalizing non-masked scores.

- Implementation

``` python 

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)            
        self.register_buffer(
           'mask',
           torch.triu(torch.ones(context_length, context_length),
           diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(                                     
          self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
          attn_weights = torch.softmax(
              attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
```
### 3.6 Extending Single-Head Attention to Multi-Head Attention

- Multi-head attention runs multiple attention mechanisms in parallel, allowing the model to capture diverse relationships.
- Each head processes a separate subspace of the input data.
- Outputs from all heads are concatenated and linearly transformed.


#### Implementation:
- A wrapper class stacks multiple attention heads and combines their outputs.

- Implementation 
``` python 
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                 d_in, d_out, context_length, dropout, qkv_bias
             )
             for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```
---

## Summary

Chapter 3 provides a detailed exploration of attention mechanisms, starting with basic concepts and progressively adding complexity. By the end, readers will understand and implement:

- Self-attention with and without trainable weights.
- The rationale and coding of causal attention.
- Multi-head attention for parallel processing of input features.

These mechanisms form the backbone of transformer architectures like GPT, enabling efficient handling of long-range dependencies and parallelization for scalability.

