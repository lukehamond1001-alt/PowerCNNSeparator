# PowerCNN Separator

A novel character-level word boundary detector based on **architecturally-imposed sparsity** — the same no-bias, no-embedding design philosophy as the original Powernet project, applied to the problem of segmenting a continuous character stream into words.

---

## The Discovery

Standard tokenizers (BPE, WordPiece) are statistical — they learn subword merges from frequency tables. This architecture takes a different approach: **8 parallel sliding-kernel scanners** scan the character stream and vote on where word boundaries should be, using only one-hot character inputs with no learned embeddings and no bias terms.

The result is a sparse, interpretable model where every neuron has a clear role: each of the 2,040 hidden neurons represents one specific **combination of scanner firings at one specific window slot**.

---

## Architecture: ScannerChunker

```
Input: 12-character window (spaces stripped from PTB text)
Task:  predict split / no-split at each of the 11 inner boundaries

(B, V=49, 12)
    ↓  pad 3 zeros each side, unfold(size=8, step=1)
(B, 49, 11, 8)         — 11 boundary positions × 8 window slots
    ↓  8 scanner weights + ReLU  (no bias — SparseMap rule)
(B, 8, 11, 8)          — 8 scanners × 11 boundaries × 8 slots
    ↓  regroup slot-first
(B, 64, 11)            — [8 scanner values @ slot 0 | … | 8 values @ slot 7]
    ↓  Conv1d(64, 2040, kernel=1, groups=8, no bias) + ReLU
(B, 2040, 11)          — 255 combination neurons per slot × 8 slots
    ↓  Conv1d(2040, 2, kernel=1, bias)
(B, 2, 11)             — split / no-split logit at each boundary
```

### Why this grouping matters

The 2,040 hidden neurons are **grouped by slot**: slot 0's 255 neurons only see which of the 8 scanners fired at slot 0; slot 1's 255 neurons only see slot 1; and so on. This means:

- **255 = 2^8 − 1**: one neuron per non-empty combination of the 8 scanner signals at that position
- No neuron is ever redundant — each one represents a unique scanner-firing pattern
- The final 2-neuron output then cross-references all 8 slots to make the split decision

### Scanner mechanics

Each scanner is a `(vocab_size, 8_slots)` weight matrix. It slides along the character stream one position at a time — the weights that hit slot 1 on step 1 hit slot 2 on step 2, etc. (shared-kernel behavior of Conv1d). No bias means: if none of the 8 scanners fire at a boundary, the model defaults to **split** (the safe choice).

### Window geometry

```
12-char input:   [c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11]
Padded to 18:    [0  0  0  c0 c1 ...              c11 0  0  0]
Unfold(8, step=1) → 11 windows of 8 chars each:
  window 0:  [0  0  0  c0 | c1 c2 c3 c4]   boundary between c0 and c1
  window 5:  [c2 c3 c4 c5 | c6 c7 c8 c9]   boundary between c5 and c6
  window 10: [c7 c8 c9 c10| c11 0  0  0]   boundary between c10 and c11

Slots 0-3: left of split   Slots 4-7: right of split
Max context: 4 characters on each side of every boundary
```

---

## Design Principles (SparseMap Rules)

Inherited from the Powernet project:

1. **No embedding tables** — one-hot inputs directly into Conv1d weights
2. **No bias on scanner or hidden layers** — absent character patterns produce exactly zero activation, enforcing architectural sparsity
3. **ReLU throughout** — negative activations die, creating hard zeros rather than soft suppressions
4. Bias is only allowed on the final output layer (the decision layer may need a threshold offset)

---

## Results

All results on Penn Treebank (PTB) validation set.

### Training on all words (no filter)

| Config | Hidden neurons | Params | Best val F1 |
|---|---|---|---|
| 8 scanners | 2,040 | 23,538 | 69.4% |
| 12 scanners | 32,760 | 463,346 | 70.3% |

### Training only on windows where every word ≤ 6 characters

Removing windows contaminated by words longer than the scanner's 4-char context (which just add noise) gave a clean **+6.5 point gain** with zero architectural changes:

| Config | Filter | Best val F1 |
|---|---|---|
| 8 scanners | none | 69.4% |
| 8 scanners | max word ≤ 6 chars | **75.9%** |

### F1 by word length (8-scanner model, all-words training)

| Both adjacent words | F1 |
|---|---|
| = 1 char | **99.9%** |
| ≤ 2 chars | 83.7% |
| ≤ 3 chars | 75.2% |
| ≤ 4 chars | 71.8% |
| ≤ 6 chars | 72.7% |
| ALL | 69.4% |

Nearly perfect on single-character words, degrading as words exceed the scanner's context window.

---

## Files

| File | Purpose |
|---|---|
| `segdata.py` | Dataset: strips spaces from PTB, records boundary labels, builds sliding windows, `make_scanner_batch()` for one-hot unfolding |
| `scanner.py` | `ScannerChunker` model, training loop (Adam + ReduceLROnPlateau), Precision/Recall/F1 metrics |
| `analyze_by_wordlen.py` | Post-hoc analysis: F1 broken down by adjacent word length |
| `checkpoints/ScannerChunker_best.pt` | Best 8-scanner checkpoint (all-words training, 69.4% F1) |
| `checkpoints/8scanner_w6/` | Best 8-scanner checkpoint (≤6-char word filter, 75.9% F1) |
| `checkpoints/12scanner/` | Best 12-scanner checkpoint (all-words training, 70.3% F1) |

---

## Usage

```bash
# Quick test — 5 epochs, 100k samples, 8 scanners
python scanner.py

# Full PTB run, save best checkpoint
python scanner.py --epochs 30 --max_samples 0 --save checkpoints/run1

# Train only on short-word windows (≤6 chars per word)
python scanner.py --epochs 30 --max_samples 0 --max_word_len 6 --save checkpoints/run_w6

# 12-scanner model
python scanner.py --scanners 12 --epochs 20 --max_samples 500000 --save checkpoints/run12

# Analyse F1 by word length
python analyze_by_wordlen.py
```

---

## Known Ceiling and Next Steps

**Current ceiling: ~76% F1 on short words, ~70% overall.**

The gap comes from two sources:

1. **Window too narrow** — 4-char context can't see beyond the immediate characters. A word like `into` looks identical to `in to` without wider context. Widening to ~20 chars (10 each side) would cover most English word endings.

2. **Spaces stripped** — the model has to infer boundaries purely from character patterns. If spaces are left in the character stream, the space character becomes a direct high-weight signal and the model should jump to ~90%+ with the same architecture.

### Roadmap

- [ ] Keep spaces in the stream as explicit boundary markers
- [ ] Widen scanner window to ~20 chars (10 each side)
- [ ] 16 scanners → 2^16-1 = 65,535 combination neurons per slot
- [ ] Expected ceiling with spaces + wider window: ~97% F1 for English

---

## Architecture Scaling

| Scanners (N) | Hidden neurons (8 × (2^N−1)) | Total params | Expected use case |
|---|---|---|---|
| 8 | 2,040 | ~24k | Quick experiments |
| 12 | 32,760 | ~463k | Better English coverage |
| 16 | 524,280 | ~7.2M | Multi-language, wider vocab |
| 32 | ~34B | impractical | Theoretical limit |

Practical sweet spot for English with spaces: **16 scanners, 20-char window**.

---

## Relation to Powernet

Both projects share the same hypothesis: **architectural sparsity through no-bias ReLU on one-hot inputs produces cleaner, more interpretable representations than learned embeddings**. Powernet applies this to language modeling (predicting the next word); PowerCNN Separator applies it to character-level boundary detection. The scanner's slot-grouped combination layer is a direct extension of Powernet's no-embed linear layer to a 2D (slot × scanner) structured space.
