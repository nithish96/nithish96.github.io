# 2. Working with Text Data

## Overview

This chapter dives into the essential steps required to prepare text data for training large language models (LLMs). It explains processes for converting raw text into a structured format suitable for machine learning models, including tokenization, embedding creation, and data sampling.

The main topics covered are:

1. Preparing Text for Training
2. Converting Tokens into IDs
3. Adding Special Context Tokens
4. Data Sampling with a Sliding Window
5. Creating Token Embeddings
6. Encoding Word Positions

---

### **2.1 Understanding Word Embeddings**

- Neural networks cannot process raw categorical text directly, necessitating the transformation of words into numerical vectors.
- Embeddings are dense vector representations of categorical data, allowing for efficient learning and generalization.
- Techniques such as Word2Vec train models to generate embeddings by analyzing the context in which words appear.
- Word embeddings cluster semantically similar words in the same region of the vector space, enhancing interpretability.
- Word embeddings with higher dimensionality might capture more nuanced relationships but at the cost of computational efficiency.
- LLMs commonly produce their own embeddings that are part of the input layer and are updated during training. 
- GPT-2 models use an embedding size of 768 dimensions. The largest GPT-3 model uses an embedding size of 12,288 dimensions.

### 2.2 Tokenizing Text

- Tokenization is the process of breaking down text into smaller units (tokens) such as words, subwords, or punctuation.
- Example of tokenization:
  - Input: `"Hello, world!"`
  - Tokens: `['Hello', ',', 'world', '!']`
- Libraries like SpaCy, NLTK, or manual regular expressions can perform tokenization, with different levels of customization and accuracy.

### 2.3 Converting Tokens into Token IDs

- A vocabulary is created by assigning each token a unique numerical ID.
- Example:
  - Vocabulary: `{'Hello': 1, ',': 2, 'world': 3, '!': 4}`
  - Tokenized Input: `[1, 2, 3, 4]`
- These token IDs form the bridge between categorical text and numerical processing.

### 2.4 Adding Special Context Tokens

- Special tokens handle specific requirements in LLMs:
  - `<|unk|>`: Represents unknown or out-of-vocabulary words.
  - `<|endoftext|>`: Marks the boundary between independent text samples.
- These additions improve model robustness by ensuring proper handling of edge cases.
- Example implementation of tokenizer would be as follows 

``` python
class SimpleTokenizerV2:
   def __init__(self, vocab):
      self.str_to_int = vocab
      self.int_to_str = { i:s for s,i in vocab.items()}
   
   def encode(self, text):
      preprocessed = re.split(r'([,.:;?_!"()\']|--|\\s)', text)
      preprocessed = [
         item.strip() for item in preprocessed if item.strip()
      ]
      preprocessed = [item if item in self.str_to_int            
                     else "<|unk|>" for item in preprocessed]
      ids = [self.str_to_int[s] for s in preprocessed]
      return ids
   
   def decode(self, ids):
      text = " ".join([self.int_to_str[i] for i in ids])
      text = re.sub(r'\s+([,.:;?!"()\\'])', r'\\1', text)    
      return text 
```

### 2.5 Byte Pair Encoding (BPE)

- BPE is a method for splitting rare or unseen words into subwords, reducing the size of the vocabulary while improving generalization.
- Example:
  - Word: `"unimaginable"`
  - Subwords: `['un', 'imagin', 'able']`
- This technique ensures efficient encoding of diverse linguistic structures.

### 2.6 Data Sampling with a Sliding Window

- A sliding window generates overlapping input-target sequences, enabling models to capture continuous context efficiently.
  - Example:
    - Input Sequence: `['The', 'cat', 'sat']`
    - Target Sequence: `['cat', 'sat', 'on']`
- This approach ensures robust learning from limited datasets.

- Example Implementation 

``` python 
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
   def __init__(self, txt, tokenizer, max_length, stride):
      self.input_ids = []
      self.target_ids = []
      token_ids = tokenizer.encode(txt)    
      for i in range(0, len(token_ids) - max_length, stride):     
         input_chunk = token_ids[i:i + max_length]
         target_chunk = token_ids[i + 1: i + max_length + 1]
         self.input_ids.append(torch.tensor(input_chunk))
         self.target_ids.append(torch.tensor(target_chunk))
   
   def __len__(self):    
      return len(self.input_ids)

   def __getitem__(self, idx):         
      return self.input_ids[idx], self.target_ids[idx]
```
### 2.7 Creating Token Embeddings

- Token Embeddings map token IDs to high-dimensional dense vectors optimized during training.
- Each token has a unique vector representation that encodes semantic and syntactic information, essential for deep learning tasks.

### 2.8 Encoding Word Positions

- Positional encoding integrates sequential information into embeddings, enabling models to understand the order of tokens.
- We can use two broad categories of position- aware embeddings: relative positional embeddings and absolute positional embeddings.
- Methods like sinusoidal encodings are commonly used, ensuring unique positional information for each token.

---

## Summary

Chapter 2 provides an in-depth exploration of preparing text data for training large language models. From tokenization and vocabulary creation to embedding generation and positional encoding, this chapter equips readers with the foundational tools for processing text data. By meticulously structuring text input, these methods enable LLMs to learn complex linguistic relationships, paving the way for effective training and generalization.


--- 

## References 

- [tiktoken](https://github.com/openai/tiktoken)
- [Byte-Pair Encoding tokenization
](https://huggingface.co/learn/nlp-course/en/chapter6/5)