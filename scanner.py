"""
scanner.py — PowerCNN ScannerChunker

Architecture
------------
Input: 12-char window (spaces stripped from PTB text).
Task:  predict split/no-split at each of the 11 inner boundaries.

Data flow (N = n_scanners, H = 2^N − 1 combos per slot):
  (B, V, 12)
      ↓  pad ±3, unfold(size=8, step=1)
  (B, V, 11, 8)          — 11 boundary positions × 8 window slots
      ↓  scanner weights + ReLU  (no bias — SparseMap rule)
  (B, N, 11, 8)          — N scanners × 11 boundaries × 8 slots
      ↓  regroup slot-first
  (B, N*8, 11)           — [N scanner vals @ slot 0 | … | N vals @ slot 7]
      ↓  Conv1d(N*8, 8*H, k=1, groups=8, no bias) + ReLU
  (B, 8*H, 11)           — H combo neurons per slot × 8 slots
      ↓  Conv1d(8*H, 2, k=1, bias)
  (B, 2, 11)             — split / no-split logits at each boundary

Default config (--scanners 8):
  N=8  → H=255   → 8×255 =  2,040 hidden neurons   ~   23k params
12-scanner config (--scanners 12):
  N=12 → H=4095  → 8×4095 = 32,760 hidden neurons  ~  463k params

Class labels:  0 = no-split (keep together)   1 = split (word boundary)
Default:       split (class 1) when all scanners are silent at a boundary.
Metrics:       Precision, Recall, F1 computed on the split (positive) class.

Usage
-----
  python scanner.py                                  # 5 epochs, 100k samples, 8 scanners
  python scanner.py --scanners 12 --max_samples 0    # 12-scanner full PTB run
  python scanner.py --epochs 20 --max_samples 0 --save checkpoints
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from segdata import load_ptb_boundaries, make_scanner_batch


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #

class ScannerChunker(nn.Module):
    """
    N parallel scanners, each producing a per-slot value (not a summed scalar).

    scanner_w[n, v, k]  — weight for scanner n, vocabulary item v, window slot k.
    At boundary b, slot k, scanner n:
        value = ReLU( Σ_v  scanner_w[n, v, k] × onehot[v] )
              = ReLU( scanner_w[n, char_id, k] )   (one-hot lookup, no bias)

    H = 2^N − 1 combination neurons per slot, grouped so that slot k's H
    neurons see only the N scanner values at slot k.  Total = 8 × H hidden neurons.

    2 output neurons see all 8×H hidden neurons.
    """

    N_SLOTS = 8   # scanner window width — fixed by geometry (4 chars each side)

    def __init__(self, vocab_size: int, n_scanners: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_scanners = n_scanners
        self.n_hidden   = (2 ** n_scanners) - 1   # combos per slot
        self.n_total    = self.N_SLOTS * self.n_hidden

        # Scanner weights — (n_scanners, vocab_size, N_SLOTS)
        # Each (scanner, slot) pair is a V → 1 projection; fan_in = vocab_size.
        # No bias: absent characters stay zero (SparseMap rule).
        self.scanner_w = nn.Parameter(
            torch.empty(n_scanners, vocab_size, self.N_SLOTS)
        )
        bound = 1.0 / math.sqrt(vocab_size)
        nn.init.uniform_(self.scanner_w, -bound, bound)

        # Hidden: 8 independent N→H linear blocks (one per slot), no bias.
        # groups=8 maps in_channels[0:N]→out[0:H], [N:2N]→[H:2H], etc.
        self.hidden = nn.Conv1d(
            n_scanners * self.N_SLOTS,    # N*8 in-channels  (N per group)
            self.n_total,                  # 8*H out-channels  (H per group)
            kernel_size=1,
            bias=False,
            groups=self.N_SLOTS,
        )

        # Output: 8*H → 2, with bias (cross-slot decision layer).
        self.out = nn.Conv1d(self.n_total, 2, kernel_size=1, bias=True)

    def forward(self, x_unfolded: torch.Tensor) -> torch.Tensor:
        """
        x_unfolded : (B, V, T, K)  — from make_scanner_batch
                     T = n_boundaries = 11,  K = n_slots = 8
        Returns logits : (B, 2, T)
        """
        B, V, T, K = x_unfolded.shape
        N = self.n_scanners

        # --- Scanner layer ---------------------------------------------------
        # x_unfolded (B, V, T, K) → permute(0,3,2,1) → (B, K, T, V)
        # scanner_w  (N, V, K)    → permute(2,1,0)    → (K, V, N)
        # matmul (B, K, T, V) @ (K, V, N) → (B, K, T, N)
        # permute(0,3,2,1) → (B, N, T, K)
        x_t = x_unfolded.permute(0, 3, 2, 1)          # (B, K, T, V)
        w_t = self.scanner_w.permute(2, 1, 0)          # (K, V, N)
        h   = torch.matmul(x_t, w_t)                   # (B, K, T, N)
        h   = h.permute(0, 3, 2, 1).contiguous()       # (B, N, T, K)
        h   = F.relu(h)

        # --- Regroup slot-first for grouped Conv1d ---------------------------
        # (B, N, T, K) → permute(0,3,1,2) → (B, K, N, T)
        # reshape → (B, K*N, T)
        h = h.permute(0, 3, 1, 2).reshape(B, K * N, T)

        # --- Hidden layer (8 independent slot blocks) ------------------------
        h = F.relu(self.hidden(h))    # (B, 8*H, T)

        # --- Output ----------------------------------------------------------
        return self.out(h)            # (B, 2, T)


# --------------------------------------------------------------------------- #
# Sparsity diagnostics                                                         #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def sparsity_stats(model: ScannerChunker) -> dict:
    """
    Fraction of scanner_w (scanner, slot) pairs whose weights are all ≤ 0
    across the entire vocabulary — approximates dead scanner-slot units.
    """
    w = model.scanner_w   # (N, V, K)
    dead = (w <= 0).all(dim=1).float().mean().item()
    return {"scanner_dead_frac": dead}


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    model: ScannerChunker,
    loader: DataLoader,
    device: torch.device,
    vocab_size: int,
    class_weight: torch.Tensor | None = None,
) -> tuple[float, float, float, float]:
    """
    Returns (avg_loss, precision, recall, f1).
    Positive class = 1 (split / word boundary).
    """
    model.eval()
    weight    = class_weight.to(device) if class_weight is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)

    total_loss   = 0.0
    total_tokens = 0
    tp = fp = fn = 0

    for context, labels in loader:
        context = context.to(device)
        labels  = labels.to(device)

        x_unf   = make_scanner_batch(context, vocab_size)   # (B, V, 11, 8)
        logits  = model(x_unf)                              # (B, 2, 11)

        B, _, T     = logits.shape
        flat_logits = logits.permute(0, 2, 1).reshape(-1, 2)   # (B*T, 2)
        flat_labels = labels.long().reshape(-1)                 # (B*T,)

        total_loss   += criterion(flat_logits, flat_labels).item() * flat_labels.size(0)
        total_tokens += flat_labels.size(0)

        pred  = flat_logits.argmax(dim=1)   # 1 = predicted split
        tp   += int(((pred == 1) & (flat_labels == 1)).sum())
        fp   += int(((pred == 1) & (flat_labels == 0)).sum())
        fn   += int(((pred == 0) & (flat_labels == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    model.train()
    return total_loss / max(total_tokens, 1), precision, recall, f1


# --------------------------------------------------------------------------- #
# Training loop                                                                #
# --------------------------------------------------------------------------- #

def train_model(
    model: ScannerChunker,
    train_ds,
    val_ds,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    class_weight: torch.Tensor | None = None,
    save_dir: str | None = None,
):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    weight    = class_weight.to(device) if class_weight is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Halve LR whenever val F1 fails to improve for 3 consecutive epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    model.to(device)

    params   = sum(p.numel() for p in model.parameters())
    best_f1  = 0.0

    print(f"\n{'='*72}")
    print(f"  ScannerChunker   ({model.n_scanners} scanners → {model.n_hidden} combos/slot"
          f" → {model.n_total:,} hidden   |   {params:,} params)")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {epochs}   Batch: {batch_size}   LR: {lr}  (ReduceLROnPlateau ×0.5, patience=3)")
    if class_weight is not None:
        print(f"  Class weight [no-split, split] : {class_weight.tolist()}")
    print(f"{'='*72}")
    header = f"  {'Ep':>3}  {'TrainLoss':>10}  {'ValLoss':>8}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'LR':>8}  {'Time':>6}"
    print(header)
    print(f"  {'-'*68}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0         = time.time()
        total_loss = 0.0
        total_tok  = 0

        for context, labels in train_loader:
            context = context.to(device)
            labels  = labels.to(device)

            x_unf  = make_scanner_batch(context, model.vocab_size)
            logits = model(x_unf)                              # (B, 2, T)

            B, _, T     = logits.shape
            flat_logits = logits.permute(0, 2, 1).reshape(-1, 2)
            flat_labels = labels.long().reshape(-1)

            loss = criterion(flat_logits, flat_labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * flat_labels.size(0)
            total_tok  += flat_labels.size(0)

        train_loss              = total_loss / total_tok
        val_loss, prec, rec, f1 = evaluate(
            model, val_loader, device, model.vocab_size, class_weight
        )
        elapsed  = time.time() - t0
        cur_lr   = optimizer.param_groups[0]['lr']

        scheduler.step(f1)

        marker = " *" if f1 > best_f1 else "  "
        print(
            f"  {epoch:>3}  {train_loss:>10.4f}  {val_loss:>8.4f}"
            f"  {prec*100:>6.1f}%  {rec*100:>6.1f}%  {f1*100:>6.1f}%"
            f"  {cur_lr:.2e}  {elapsed:>5.1f}s{marker}"
        )

        if f1 > best_f1:
            best_f1 = f1
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                ckpt = os.path.join(save_dir, "ScannerChunker_best.pt")
                torch.save(model.state_dict(), ckpt)

    stats = sparsity_stats(model)
    print(f"\n  Best val F1 : {best_f1 * 100:.1f}%")
    print(f"  Scanner dead-weight frac : {stats['scanner_dead_frac']:.3f}")
    if save_dir and best_f1 > 0:
        print(f"  Best checkpoint saved → {os.path.join(save_dir, 'ScannerChunker_best.pt')}")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train the PowerCNN ScannerChunker.")
    parser.add_argument("--epochs",      type=int,   default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch",       type=int,   default=512,
                        help="Batch size (default: 512)")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--scanners",     type=int,   default=8,
                        help="Number of parallel scanners; hidden layer = 8 × (2^N − 1) (default: 8)")
    parser.add_argument("--max_samples", type=int,   default=100_000,
                        help="Cap training windows; 0 = full PTB (default: 100000)")
    parser.add_argument("--max_word_len",type=int,   default=0,
                        help="Drop windows containing any word longer than this (0 = keep all, default: 0)")
    parser.add_argument("--save",        type=str,   default=None,
                        help="Directory to save best checkpoint (optional)")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print("\nLoading boundary detection data...")
    max_val = max(args.max_samples // 5, 1) if args.max_samples > 0 else 0
    train_ds, val_ds, _, vocab, _, pos_weight = load_ptb_boundaries(
        max_train=args.max_samples,
        max_val=max_val,
        max_word_len=args.max_word_len,
    )
    vocab_size   = len(vocab)            # includes padding at index 0
    class_weight = torch.tensor([1.0, pos_weight])

    model = ScannerChunker(vocab_size, n_scanners=args.scanners)
    train_model(
        model, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=device,
        class_weight=class_weight,
        save_dir=args.save,
    )


if __name__ == "__main__":
    main()
