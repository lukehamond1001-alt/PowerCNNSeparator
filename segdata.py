"""
segdata.py — Word-boundary detection dataset for the PowerCNN ScannerChunker.

Strips spaces from PTB text, records their positions as split labels, and
produces sliding (12-char window, 11-label) pairs.

  label[b] = 1  →  split between window char b and char b+1 (word boundary)
  label[b] = 0  →  keep together (within a word)

One-hot encoding and per-boundary unfolding happen per-batch via
make_scanner_batch(), so the dataset stores compact integer IDs.
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CACHE_DIR    = os.path.join(os.path.dirname(__file__), ".ptb_cache")
WINDOW_LEN   = 12
N_BOUNDARIES = WINDOW_LEN - 1   # 11
N_SLOTS      = 8                # scanner window width


# --------------------------------------------------------------------------- #
# PTB loading                                                                  #
# --------------------------------------------------------------------------- #

def _load_cached(split: str) -> str:
    path = os.path.join(CACHE_DIR, f"ptb.{split}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python experiment.py` first to download PTB."
        )
    with open(path) as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Vocab                                                                        #
# --------------------------------------------------------------------------- #

def build_char_vocab(text: str) -> tuple[dict, dict]:
    """
    Vocab built from non-space characters only.
    Index 0 is reserved for padding (edge-padding in make_scanner_batch).
    """
    chars   = sorted(set(c for c in text if c not in (' ', '\n')))
    vocab   = {'<pad>': 0}
    vocab.update({c: i + 1 for i, c in enumerate(chars)})
    idx2char = {v: k for k, v in vocab.items()}
    return vocab, idx2char


# --------------------------------------------------------------------------- #
# Stripping and labelling                                                      #
# --------------------------------------------------------------------------- #

def _strip_and_label(
    text: str,
    vocab: dict,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Strip spaces/newlines from text and record where word boundaries were.

    Returns:
      ids        — non-space char IDs (length L)
      boundaries — 0/1 flags, length L-1
                   boundaries[i] = 1  iff there was whitespace between
                   ids[i] and ids[i+1] in the original text
      left_lens  — length of the word to the LEFT  of each boundary (length L-1)
      right_lens — length of the word to the RIGHT of each boundary (length L-1)
    """
    words = text.replace('\n', ' ').split()

    ids        = []
    word_idxs  = []
    for wi, w in enumerate(words):
        for c in w:
            ids.append(vocab.get(c, 0))
            word_idxs.append(wi)

    word_lens  = [len(w) for w in words]
    boundaries = [1 if word_idxs[i] != word_idxs[i + 1] else 0
                  for i in range(len(ids) - 1)]
    left_lens  = [word_lens[word_idxs[i]]     for i in range(len(ids) - 1)]
    right_lens = [word_lens[word_idxs[i + 1]] for i in range(len(ids) - 1)]

    return ids, boundaries, left_lens, right_lens


# --------------------------------------------------------------------------- #
# Sliding windows                                                              #
# --------------------------------------------------------------------------- #

def make_windows(
    ids: list[int],
    boundaries: list[int],
    window_len: int    = WINDOW_LEN,
    max_word_len: int  = 0,
    left_lens: list[int]  | None = None,
    right_lens: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      x : (N, window_len)     int64   — char ID windows
      y : (N, window_len-1)   float32 — boundary label windows

    If max_word_len > 0, windows where ANY word at ANY boundary
    exceeds max_word_len characters are dropped.  This removes
    windows contaminated by long words that the scanner can't fully see.
    """
    ids_t = torch.tensor(ids,        dtype=torch.long)
    bnd_t = torch.tensor(boundaries, dtype=torch.float32)
    x = ids_t.unfold(0, window_len,     1)   # (N, 12)
    y = bnd_t.unfold(0, window_len - 1, 1)   # (N, 11)

    if max_word_len > 0 and left_lens is not None and right_lens is not None:
        lw_t   = torch.tensor(left_lens,  dtype=torch.long)
        rw_t   = torch.tensor(right_lens, dtype=torch.long)
        maxw_t = torch.max(lw_t, rw_t)                        # (L-1,)
        # For window i: boundaries i … i+(window_len-2) must all be ≤ max_word_len
        maxw_win = maxw_t.unfold(0, window_len - 1, 1)        # (N, 11)
        keep     = maxw_win.amax(dim=1) <= max_word_len        # (N,)
        x, y     = x[keep], y[keep]

    return x, y


def make_scanner_batch(context_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Convert integer char windows to the unfolded one-hot tensor for ScannerChunker.

    context_ids : (B, 12)   int64
    Returns     : (B, vocab_size, 11, 8)   float32
                  [batch, vocab, n_boundaries, n_slots]

    Steps:
      1. one-hot on CPU → (B, 12, V) → permute → (B, V, 12)
      2. pad 3 zeros on each side   → (B, V, 18)
      3. unfold(size=8, step=1)     → (B, V, 11, 8)

    scatter_ is always built on CPU (unsupported on MPS), then moved to device.
    """
    device  = context_ids.device
    ids_cpu = context_ids.cpu()
    B, C    = ids_cpu.shape                                    # C == 12

    one_hot = torch.zeros(B, C, vocab_size, dtype=torch.float32)
    one_hot.scatter_(2, ids_cpu.unsqueeze(2), 1.0)             # (B, 12, V)
    x = one_hot.permute(0, 2, 1).contiguous()                  # (B, V, 12)
    x = F.pad(x, (3, 3))                                       # (B, V, 18)
    x = x.unfold(2, 8, 1)                                      # (B, V, 11, 8)
    return x.to(device)


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

class BoundaryDataset(Dataset):
    """Pairs of (12-char integer window, 11 binary boundary labels)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x   # (N, 12)  int64
        self.y = y   # (N, 11)  float32

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# --------------------------------------------------------------------------- #
# Public loader                                                                #
# --------------------------------------------------------------------------- #

def load_ptb_boundaries(
    window_len: int   = WINDOW_LEN,
    max_train: int    = 0,
    max_val: int      = 0,
    max_word_len: int = 0,
):
    """
    Returns:
      train_ds, val_ds, test_ds  — BoundaryDataset instances
      vocab, idx2char
      pos_weight  — n_neg / n_pos in train set (for weighted CrossEntropyLoss)

    max_word_len : if > 0, drop any window where a word longer than this
                  appears anywhere in the window.  Use 6 to keep only windows
                  the 8-wide scanner can fully cover (4 chars each side).
    """
    train_text = _load_cached("train")
    val_text   = _load_cached("valid")
    test_text  = _load_cached("test")

    vocab, idx2char = build_char_vocab(train_text)

    def make_ds(text, cap):
        ids, bnd, lw, rw = _strip_and_label(text, vocab)
        x, y = make_windows(ids, bnd, window_len,
                             max_word_len=max_word_len,
                             left_lens=lw, right_lens=rw)
        if cap > 0:
            x, y = x[:cap], y[:cap]
        return BoundaryDataset(x, y)

    train_ds = make_ds(train_text, max_train)
    val_ds   = make_ds(val_text,   max_val)
    test_ds  = make_ds(test_text,  0)

    n_pos      = float(train_ds.y.sum())
    n_neg      = float(train_ds.y.numel() - n_pos)
    pos_weight = n_neg / max(n_pos, 1.0)

    wlen_tag = f"  (max word len ≤ {max_word_len})" if max_word_len > 0 else ""
    print(f"Char vocab size   : {len(vocab)}  (index 0 = padding)")
    print(f"Train samples     : {len(train_ds):,}"
          + (f"  (capped at {max_train:,})" if max_train > 0 else "")
          + wlen_tag)
    print(f"Val   samples     : {len(val_ds):,}"
          + (f"  (capped at {max_val:,})" if max_val > 0 else "")
          + wlen_tag)
    print(f"Test  samples     : {len(test_ds):,}" + wlen_tag)
    print(f"Label balance     : {n_pos / (n_pos + n_neg) * 100:.1f}% split  "
          f"(pos_weight = {pos_weight:.2f})")

    return train_ds, val_ds, test_ds, vocab, idx2char, pos_weight


if __name__ == "__main__":
    load_ptb_boundaries()
