# 5. Pretraining on Unlabeled Data

## Overview

Chapter 5 focuses on the foundational process of pretraining a large language model (LLM) like GPT. Pretraining involves teaching the model to predict the next word in a sequence using large, unlabeled datasets. This stage helps the model learn general language patterns, syntax, and semantics, forming the foundation for downstream tasks.

This chapter covers 

1. Computing training and validation losses.
2. Implementing pretraining procedures.
3. Saving and loading model weights.

---

### 5.1 Evaluating Generative Text Models

#### 5.1.1 Using GPT for Text Generation

- Text generation is the primary task of GPT during pretraining.
- The model generates text by iteratively predicting the next token based on the input sequence.
- Evaluation involves:
    1. Providing a prompt (e.g., "Once upon a time...").
    2. Measuring the quality and coherence of the generated output.

#### 5.1.2 Calculating Text Generation Loss

- Loss measures how well the model predicts the next word in a sequence.
- Loss functions such as cross-entropy are used, where lower loss indicates better performance.
- Perplexity is a measure often used alongside cross entropy loss to evaluate the performance of models in tasks like language modeling

#### 5.1.3 Calculating Training and Validation Losses

- Training loss: Assesses the model's performance on the training dataset.
- Validation loss: Measures the model's ability to generalize to unseen data.
- Monitoring these losses helps identify overfitting or underfitting.

``` python 
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)         #1
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
)
return loss
```

### 5.2 Training an LLM

- Training involves:
    1. Tokenizing the input dataset.
    2. Feeding tokenized data into the model.
    3. Computing loss and updating weights via backpropagation.
- Optimization techniques such as Adam or AdamW are employed for stable training.

``` python 
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []    
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):    
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()   
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()                     
            optimizer.step()                    
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:    
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
)
        generate_and_print_sample(                      
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen
```

### 5.3 Decoding Strategies to Control Randomness

- Decoding strategies affect the quality and diversity of generated text:

#### 5.3.1 Temperature Scaling

- Controls randomness by scaling the logits before applying softmax.
- Higher temperatures increase diversity but may reduce coherence.
- Lower temperatures produce more deterministic outputs.

#### 5.3.2 Top-k Sampling

- Limits predictions to the top-k most probable tokens, improving coherence.
- Reduces the likelihood of low-probability tokens disrupting the output.

#### 5.3.3 Modifying the Text Generation Function

- Custom implementations can combine strategies like top-k sampling and temperature scaling for fine-grained control.

``` python 
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
  for _ in range(max_new_tokens):
      idx_cond = idx[:, -context_size:]
      with torch.no_grad():
          logits = model(idx_cond)
      logits = logits[:, -1, :]
      if top_k is not None:                     
          top_logits, _ = torch.topk(logits, top_k)
          min_val = top_logits[:, -1]
          logits = torch.where(
              logits < min_val,
              torch.tensor(float('-inf')).to(logits.device),
              logits
          )
      if temperature > 0.0:                  
          logits = logits / temperature
          probs = torch.softmax(logits, dim=-1)
          idx_next = torch.multinomial(probs, num_samples=1)
      else:    
          idx_next = torch.argmax(logits, dim=-1, keepdim=True)
      if idx_next == eos_id:              
          break
      idx = torch.cat((idx, idx_next), dim=1)
  return idx
```
### 5.4 Saving and Loading Model Weights in PyTorch

- Saving weights allows for resuming training or deploying the model:
  1. Save weights using `torch.save`.
  2. Load weights with `torch.load`.
- Models can also load pretrained weights from publicly available sources to save computation costs.

---

## Summary

Chapter 5 provides a comprehensive guide to pretraining LLMs on unlabeled data. By focusing on efficient loss computation, decoding strategies, and weight management, it equips readers to train robust models capable of generalizing across diverse language tasks. The foundation established here enables fine-tuning and application in real-world scenarios.

