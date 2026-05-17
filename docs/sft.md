# Supervised Fine-Tuning (SFT)

> *The base model can talk. The problem is it talks however it wants. SFT teaches it when to start and when to stop.*

---

## The Problem SFT Solves

After pre-training, AlmondGPT can generate coherent text. But if you give it the prompt `"Tell me about yourself"`, it won't answer the question — it'll just generate random text that matches the distribution of TinyStories.

```
Prompt  : "Tell me about yourself"
Base    : "One day a little girl found a dog. The dog was very happy..."
```

The model doesn't know it's supposed to **answer** the prompt. It only knows how to **continue** text.

SFT teaches the format: when the user speaks, when the model should respond, and when to stop.

---

## Dataset Format

I manually created an instruction-response dataset — question and answer pairs about a specific context.

Format used:

```
<|user|> [question or instruction] <|assistant|> [answer] <|endoftext|>
```

Example sample:

```
<|user|> Tell a story about a fast train. <|assistant|> A girl accidentally 
knocked over a rock and fell fast on the cushion floor... <|endoftext|>
```

**Why this format matters:**

The model learns that after `<|user|>` comes context it needs to pay attention to, and after `<|assistant|>` it's its turn to generate. `<|endoftext|>` teaches the model when to stop.

These special tokens already existed in the vocab from tokenizer training — they have embedding rows in the matrix, but their weights were never updated since they never appeared in TinyStories. SFT is the first stage where they get "activated."

---

## Loss Function — Cross Entropy with Ignore Index

One important difference in SFT compared to pre-training: **padding tokens must not be included in the loss calculation.**

```python
targets[targets == pad_token_id] = ignore_index  # ignore_index = -100
loss = F.cross_entropy(logits, targets)  # PyTorch automatically ignores -100
```

Why? Because if padding is included in the loss, the model will learn to generate padding tokens — and that's not what we want.

---

## Training Config

```yaml
num_epochs    : 15   (SFT doesn't need many epochs)
learning_rate : 5e-5  (much smaller than pre-training 1e-3)
batch_size    : 64
max_length    : 128
optimizer     : AdamW
```

**Why a small learning rate?**

The model already has knowledge from pre-training. If the LR is too large, fine-tuning will overwrite what was already learned — this is called **catastrophic forgetting**. A small LR ensures we adjust the model's behavior, not reset it from scratch.

---

## Loss Curve

![SFT Loss](assets/sft_loss.png)

| Iteration | Loss |
|-----------|------|
| 1 | 4.20 |
| 5 | ~3.80 |
| 10 | ~3.60 |
| 15 | 3.41 |

SFT loss starts lower than pre-training because the model already has language understanding — it's not learning from zero, just adjusting behavior.

---

## Output Comparison

> Prompt: *"Tell a story about a fast train."*

**Base model:**
```
Once upon a time there was a train that went very fast.
The train was red and big. One day...
(keeps generating without any instruction-following structure)
```

**SFT model:**
```
A girl accidentally knocked over a rock and fell fast on the cushion floor.
The girl cried. Her mom tried to stop her arm to say roll. She told her
to stop crying and be brave.
```

The difference is subtle but real: the SFT model starts **responding** to the prompt, not just continuing text. The format is beginning to take shape.

---

## Limitations of SFT in AlmondGPT

Honestly — the output is still far from a capable chatbot. A few reasons:

- The instruction dataset is very small (~50-100 pairs). The model doesn't have enough examples to generalize well.
- Base model is only 10-15M parameters — limited representational capacity.
- TinyStories as pre-training data makes the model "think" in the context of children's stories.

**But this isn't a failure.** The goal of SFT here isn't to build a good chatbot. The goal is to prove that **instruction-following format can be taught** to a model through supervised learning. And it worked.

---

## What I Learned

SFT is surprisingly straightforward to implement — it's basically standard fine-tuning with an instruction-formatted dataset. The tricky parts were:

1. **Collate function** — correct padding, shifting targets by one, ignore index for padding tokens
2. **Learning rate** — too large means catastrophic forgetting, too small means no visible change
3. **Special token handling** in the tokenizer — make sure `<|user|>` and `<|assistant|>` encode as a single token ID, not split into individual bytes

Next step: DPO — teaching the model not just how to answer, but **how to answer in the way I prefer**.
