
Attention mechanisms in neural networks allow models to focus on relevant parts of the input data when making predictions. They have been widely used in various tasks, including natural language processing, computer vision, and sequence-to-sequence tasks. Here's an explanation of attention in neural networks along with some code examples:

## Dot Product Attention

One of the simplest forms of attention is dot product attention. In this mechanism, the attention score between a query vector q and a key vector k is computed as the dot product of the two vectors. The attention score is then normalized using a softmax function to obtain attention weights. Finally, the weighted sum of the value vectors vv is computed using the attention weights.

``` py 
import torch
import torch.nn.functional as F

def dot_product_attention(query, key, value):
    # Compute attention scores
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Apply softmax to obtain attention weights
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    # Compute weighted sum of value vectors
    attention_output = torch.matmul(attention_weights, value)
    
    return attention_output, attention_weights

```

## Self-Attention

Self-attention is a mechanism used in the Transformer architecture that allows the model to weigh the importance of different words in a sequence when encoding or decoding. It enables the model to focus on relevant parts of the input sequence and has been crucial in achieving state-of-the-art performance in various natural language processing tasks.

- **Query, Key, and Value:** In self-attention, each word in the input sequence is represented by three vectors: a query vector, a key vector, and a value vector. These vectors are linear transformations of the input word embeddings.
    
- **Attention Scores:** For each word in the input sequence, self-attention computes attention scores that measure the relevance of that word to every other word in the sequence. These scores are computed based on the similarity between the query vector of the current word and the key vectors of all other words.
    
- **Attention Weights:** The attention scores are normalized using a softmax function to obtain attention weights. These weights determine how much each word in the sequence contributes to the representation of the current word.
    
- **Weighted Sum:** The final representation of each word is computed as a weighted sum of the value vectors of all words in the sequence, where the weights are the attention weights.

``` py 
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # Compute query, key, and value vectors
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute weighted sum of value vectors
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output, attention_weights
```


## Multi head Attention 

Multi-head self-attention is an extension of the self-attention mechanism used in neural networks, particularly in models like the Transformer. It enables the model to capture different aspects of the input sequence simultaneously by computing multiple attention heads in parallel. Each attention head learns different representations of the input sequence, allowing the model to attend to different parts of the sequence and capture different patterns or relationships.

- **Query, Key, and Value:** Like in regular self-attention, each word in the input sequence is represented by three vectors: a query vector, a key vector, and a value vector. These vectors are linear transformations of the input embeddings.
    
- **Multiple Heads:** In multi-head self-attention, the model computes multiple sets of query, key, and value vectors, known as attention heads. Each attention head is capable of learning different relationships between words in the sequence.
    
- **Parallel Computation:** The attention heads are computed in parallel, allowing the model to capture different aspects of the input sequence simultaneously. After computing the attention scores for each attention head, the outputs are concatenated and linearly transformed to obtain the final output.

``` py 
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.linear_out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        # Split the embedding into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute weighted sum of value vectors
        attention_output = torch.matmul(attention_weights, value)
        
        # Concatenate attention heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, embed_dim)
        
        # Apply linear transformation for output
        attention_output = self.linear_out(attention_output)
        
        return attention_output, attention_weights

```

## Cross Attention 

Cross-attention, also known as encoder-decoder attention, is a type of attention mechanism used in the Transformer architecture for sequence-to-sequence tasks, such as machine translation. Unlike self-attention, which focuses on relationships within a single input sequence, cross-attention allows the decoder to attend to different parts of the encoder's input sequence when generating the output sequence.


- **Query, Key, and Value:** In cross-attention, the query vectors are generated from the decoder's hidden states, while the key and value vectors are generated from the encoder's hidden states. This allows the decoder to attend to different parts of the encoder's input sequence when generating each token in the output sequence.
    
- **Computing Attention Scores:** The attention scores are computed as the dot product of the query vectors (decoder) with the key vectors (encoder). This measures the similarity between the decoder's current state and each token in the encoder's input sequence.
    
- **Applying Softmax:** Softmax is applied to the attention scores to obtain attention weights, which determine how much each token in the encoder's input sequence contributes to the generation of the current token in the decoder's output sequence.
    
- **Weighted Sum:** The final representation of each token in the decoder's output sequence is computed as a weighted sum of the encoder's value vectors, where the weights are the attention weights.

``` py 
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, decoder_hidden, encoder_hidden):
        # Compute query vectors from decoder's hidden states
        query = self.query_linear(decoder_hidden)
        
        # Compute key and value vectors from encoder's hidden states
        key = self.key_linear(encoder_hidden)
        value = self.value_linear(encoder_hidden)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute weighted sum of value vectors
        context_vector = torch.matmul(attention_weights, value)
        
        return context_vector, attention_weights

```
## Conclusion 

These are some examples of attention mechanisms in neural networks. They allow models to focus on relevant parts of the input data and have been instrumental in achieving state-of-the-art performance in various machine learning tasks.


## References 

1. [Understanding Deep Learning](https://udlbook.github.io/udlbook/)
2.  [Transformer Primer ](https://aman.ai/primers/ai/transformers/)