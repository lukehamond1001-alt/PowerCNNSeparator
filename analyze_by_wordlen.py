"""
Breaks down ScannerChunker F1 by the length of the longest word
adjacent to each boundary.  This answers: does the model approach
100% on short words that fit fully inside the 4-char scanner context?
"""
import torch
from segdata import make_scanner_batch, _load_cached, build_char_vocab
from scanner import ScannerChunker

train_text = _load_cached("train")
val_text   = _load_cached("valid")
vocab, _   = build_char_vocab(train_text)
vocab_size  = len(vocab)

# ------------------------------------------------------------------ #
# Build val char stream with per-boundary word-length metadata        #
# ------------------------------------------------------------------ #
def strip_with_meta(text, vocab):
    words = text.replace("\n", " ").split()
    chars = []
    for wi, w in enumerate(words):
        for c in w:
            chars.append((vocab.get(c, 0), wi))
    ids   = torch.tensor([x[0] for x in chars], dtype=torch.long)
    widxs = torch.tensor([x[1] for x in chars], dtype=torch.long)
    wlens = torch.tensor([len(w) for w in words], dtype=torch.long)
    bnds  = (widxs[1:] != widxs[:-1]).long()   # 1 = word boundary
    lw    = wlens[widxs[:-1]]                   # left-word length at each boundary
    rw    = wlens[widxs[1:]]                    # right-word length at each boundary
    return ids, bnds, lw, rw

print("Building val metadata...", flush=True)
ids, bnds, lws, rws = strip_with_meta(val_text, vocab)
WINDOW = 12
N = len(ids) - WINDOW + 1

# Unfold into windows — all (N, 11) tensors stay on CPU
ctx_all  = ids.unfold(0, WINDOW,     1)   # (N, 12)
lbl_all  = bnds.unfold(0, WINDOW-1, 1)   # (N, 11)
lw_all   = lws.unfold(0,  WINDOW-1, 1)   # (N, 11)
rw_all   = rws.unfold(0,  WINDOW-1, 1)   # (N, 11)
maxw_all = torch.max(lw_all, rw_all)      # (N, 11)

# ------------------------------------------------------------------ #
# Load model                                                          #
# ------------------------------------------------------------------ #
device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

model = ScannerChunker(vocab_size, n_scanners=8)
model.load_state_dict(torch.load("checkpoints/ScannerChunker_best.pt",
                                  map_location="cpu", weights_only=True))
model.to(device).eval()
print(f"Model loaded  ({sum(p.numel() for p in model.parameters()):,} params)  device={device}")

# ------------------------------------------------------------------ #
# Inference                                                           #
# ------------------------------------------------------------------ #
BATCH    = 4096
pred_all = []
print(f"Running inference on {N:,} windows...", flush=True)
with torch.no_grad():
    for i in range(0, N, BATCH):
        ctx = ctx_all[i : i + BATCH].contiguous()   # (b, 12) on CPU
        # make_scanner_batch builds one-hot on CPU then moves to device
        x   = make_scanner_batch(ctx.to(device), vocab_size)   # (b, V, 11, 8)
        pred_all.append(model(x).argmax(dim=1).cpu())           # (b, 11)
        if i % 100_000 == 0:
            print(f"  {i:,} / {N:,}", flush=True)

preds  = torch.cat(pred_all).reshape(-1)   # (N*11,)
labels = lbl_all.contiguous().reshape(-1)
maxw   = maxw_all.contiguous().reshape(-1)
print("Done.\n")

# ------------------------------------------------------------------ #
# Report                                                              #
# ------------------------------------------------------------------ #
print(f"  {'Max word len':>14}  {'# splits':>9}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'no-split acc':>12}")
print("  " + "-" * 68)

buckets = [
    (1,  "= 1 char"),
    (2,  "<= 2 chars"),
    (3,  "<= 3 chars"),
    (4,  "<= 4 chars"),   # fully inside scanner window
    (5,  "<= 5 chars"),
    (6,  "<= 6 chars"),
    (8,  "<= 8 chars"),
    (12, "<= 12 chars"),
    (999,"ALL"),
]
for max_len, label in buckets:
    mask = maxw <= max_len
    if mask.sum() == 0:
        continue
    p  = preds[mask]
    l  = labels[mask]
    tp = int(((p == 1) & (l == 1)).sum())
    fp = int(((p == 1) & (l == 0)).sum())
    fn = int(((p == 0) & (l == 1)).sum())
    tn = int(((p == 0) & (l == 0)).sum())
    prec   = tp / (tp + fp + 1e-8)
    rec    = tp / (tp + fn + 1e-8)
    f1     = 2 * prec * rec / (prec + rec + 1e-8)
    ns_acc = tn / (tn + fp + 1e-8)
    print(f"  {label:>14}  {int(l.sum()):>9,}  {prec*100:>6.1f}%  {rec*100:>6.1f}%"
          f"  {f1*100:>6.1f}%  {ns_acc*100:>10.1f}%")
