# Pre-Training — AlmondGPT Base Model

> *I wrote these notes for myself. If you're reading this and confused, you're probably not future me.*

---

## Why Build from Scratch?

I could've grabbed GPT-2 from HuggingFace, fine-tuned it, and called it a day. But that wouldn't have taught me why attention works, why RMSNorm is better, or why sampling has no gradient.

So I built everything from zero. Every line of code written by hand. Every bug debugged myself. Turns out that's exactly what made everything stick.

---

## Dataset

**TinyStories** — 50,000 rows of short children's stories.

Why TinyStories? Because the language structure is simple and consistent — great for a small model learning language from scratch. The model doesn't need to understand the world yet, just sentence patterns.

Target vocab size: 16,000 → actual after BPE early-stop: **5,848 tokens**

BPE stopped early because remaining token pair frequencies dropped below threshold 5 — meaning a 50k-row corpus isn't diverse enough to justify 16,000 vocab entries. The YAML config auto-updates after tokenizer training completes.

---

## Tokenizer — BPE from Scratch

I implemented Byte Pair Encoding from scratch, not using tiktoken or HuggingFace tokenizers.

**Why byte-level?**

So there's no such thing as an "ungenerable" token. If a new word isn't in the vocab, the model can assemble it byte by byte. No unknown tokens.

**BPE mechanics:**

```
1. Encode all text into bytes (0-255)
2. Count frequency of all adjacent byte pairs
3. Merge the most frequent pair → becomes a new token
4. Repeat until vocab_size is reached or frequency < threshold
```

This differs from WordPiece in BERT which merges by probability `P(AB) / P(A)*P(B)`. BPE is purely by frequency — simpler and better suited for GPT-style autoregressive models.

**Special tokens** are registered manually, skipping the merge process entirely:

```
ID 256: <|endoftext|>
ID 257: <|pad|>
ID 258: <|user|>
ID 259: <|assistant|>
ID 260: <|startoftext|>
```

---

## Architecture — Why Not Vanilla GPT-2?

Original GPT-2 came out in 2019. A lot has improved since then. I went straight for modern architecture choices rather than replicating something old.

---

### RMSNorm, not LayerNorm

LayerNorm formula:
```
y = (x - mean(x)) / sqrt(var(x) + ε) * γ + β
```

RMSNorm formula:
```
y = x / sqrt(mean(x²) + ε) * γ
```

**My intuition:** subtracting the mean isn't that useful because models learn from variation, not from the mean itself. Dropping it makes computation cheaper and results are empirically comparable. LLaMA, Mistral — they all use this.

---

### Grouped Query Attention (GQA), not Multi-Head Attention

Standard MHA: every head has its own Q, K, V. With 8 heads, that's 8 KV sets to store in the KV cache during inference.

GQA: multiple query heads share one KV set.

```
My config: n_q_head=8, n_kv_heads=4
→ Query heads 1-2 share KV group 0
→ Query heads 3-4 share KV group 1
→ and so on
```

**My intuition:** MHA is expensive at inference because the KV cache has to store K and V for every single head. GQA reduces the KV cache memory footprint without much quality loss. LLaMA-2 and Mistral both use this.

---

### Rotary Positional Encoding (RoPE), not Learned PE

Learned Positional Encoding (LPE) has a problem: across different batches, token positions can differ. The model has to re-learn relative positions between tokens constantly → slow training. At inference, if it encounters an unseen position → it just guesses randomly.

**My intuition on RoPE:** RoPE uses trigonometry. Token positions are represented as rotation angles θ. What matters isn't the absolute position — it's the relative distance between tokens. Tokens at positions 1 and 3 have a distance of 2, and that's consistent across the entire context window.

```
q_rotated = q * cos(θ) + rotate_half(q) * sin(θ)
```

Because it's grounded in mathematical rotation, positions are consistent and don't change across batches. The model learns faster. At unseen positions during inference, the model can extrapolate because the mathematical pattern holds.

---

### SwiGLU, not ReLU/GELU FFN

Standard FFN:
```
output = Linear(ReLU(Linear(x)))
```

SwiGLU:
```
output = Linear(SiLU(Linear(x)) * Linear(x))
```

**My intuition:** there are two paths — one goes through SiLU (which gives a little tolerance to negative values, similar to GELU), the other passes through directly. Both paths get multiplied. High values get amplified. Values pushed toward zero die out. This captures important non-linearities better than ReLU which hard-kills everything negative.

ReLU is a strict bouncer — anything negative gets kicked out immediately. SwiGLU is a smarter bouncer — it checks the context first.

---

### KV Cache

Without KV cache, every new generated token requires recomputing all K and V from every previous token. That's O(n²) per token.

With KV cache, K and V from previous tokens are stored. A new token only needs to compute its own K and V, then append to the cache.

```python
if use_cache:
    if self.cache_key is None:
        self.cache_key, self.cache_value = K_rope, V
    else:
        self.cache_key = torch.cat([self.cache_key, K_rope], dim=2)
        self.cache_value = torch.cat([self.cache_value, V], dim=2)
```

One gotcha I discovered: the KV cache must be cleared at the start of every new generation. If not, output duplicates — tokens from the previous generation are still sitting in the cache.

---

## Training Config

```yaml
embedding_dim : 256
n_head        : 8  (n_q_head=8, n_kv_heads=4)
n_blocks      : 6
block_size    : 128
batch_size    : 32
learning_rate : 0.001
optimizer     : AdamW + Cosine Annealing LR
max_iters     : 5000
```

---

## Loss Curve

![Pre-Training Loss](assets/pretrain_loss.png)

| Step | Loss |
|------|------|
| 0 | 8.83 |
| 500 | 4.22 |
| 1000 | 3.75 |
| 2000 | 3.37 |
| 3000 | 3.16 |
| 4000 | 3.05 |
| 5000 | 2.97 |

The initial loss ~8.83 is expected — close to `ln(vocab_size) = ln(5848) ≈ 8.67`. This means random weights haven't learned anything yet and the output distribution is nearly uniform.

Loss drops quickly in the first 500 iterations as the model picks up the most common token frequencies. After that it slows down as the model starts learning more complex patterns.

---

## Output Sample

```
Prompt  : "Once upon a time"
Output  : "Once upon a time there was a little girl who lived in a big house.
           One day she found a small dog. The dog was very happy.
           They played together every day. The end.<|endoftext|>"
```

The model can already generate stories with basic structure — an opening, a simple plot, and an ending. Occasional weird tokens (`pmiddm`, `thumbant`) still appear — expected at this vocab size.

---

## What I Learned

The biggest takeaway from pre-training: **loss numbers alone don't tell the full story**. A loss of 2.97 doesn't say much until you look at actual output — whether sentence structure exists, whether the model generates `<|endoftext|>` in the right place, whether there's any coherence at all.

Next step: SFT — teaching the model when to speak and when to stop, in the context of instructions.
