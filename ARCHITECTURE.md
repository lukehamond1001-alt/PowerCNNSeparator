# PowerCNN Separator — Architecture Equations

## Notation

| Symbol | Meaning |
|---|---|
| $B$ | batch size |
| $V = 49$ | vocabulary size (unique characters + padding index 0) |
| $N = 8$ | number of scanners |
| $K = 8$ | window slots (slots 0–3 left of split, slots 4–7 right) |
| $T = 11$ | boundary positions per 12-character window |
| $H = 2^N - 1 = 255$ | combination neurons per slot |

---

## Step 0 — Input Encoding

A 12-character window $c_0, c_1, \ldots, c_{11}$ where each $c_i \in \{0, \ldots, V-1\}$ is an integer character ID.

Pad 3 zeros on each side to produce an 18-element extended sequence $\tilde{c}$:

$$\tilde{c}_i = \begin{cases} 0 & i < 3 \text{ or } i > 14 \\ c_{i-3} & \text{otherwise} \end{cases}$$

For boundary $t \in \{0,\ldots,10\}$ and slot $k \in \{0,\ldots,7\}$, the input is the one-hot vector:

$$\mathbf{x}^{(t,k)} = \text{onehot}(\tilde{c}_{t+k}) \in \{0,1\}^V$$

---

## Step 1 — Scanner Layer

**Weight tensor** $\mathbf{W} \in \mathbb{R}^{N \times V \times K}$, no bias.

For scanner $n$, boundary $t$, slot $k$:

$$s_{n,t,k} = \text{ReLU}\!\left(\sum_{v=1}^{V} W_{n,v,k} \cdot x^{(t,k)}_v\right)$$

Because $\mathbf{x}^{(t,k)}$ is one-hot at index $\tilde{c}_{t+k}$, this simplifies to a **direct character lookup**:

$$\boxed{s_{n,t,k} = \text{ReLU}\!\left(W_{n,\,\tilde{c}_{t+k},\,k}\right)}$$

**SparseMap rule in equation form:** the scanner either fires on the exact character present at that slot, or outputs exactly zero.  No soft mixing between characters.  No bias means zero cannot be shifted — silence is structural, not learned.

Scanner output for all boundaries and slots:

$$\mathbf{S}^{(t)} \in \mathbb{R}^{N \times K}, \qquad S^{(t)}_{n,k} = s_{n,t,k}$$

---

## Step 2 — Slot-Grouped Combination Layer

Each slot $k$ has its own independent weight matrix $\mathbf{U}_k \in \mathbb{R}^{H \times N}$, no bias.

The $N$ scanner values at slot $k$ and boundary $t$ feed into $H = 255$ combination neurons:

$$\mathbf{h}^{(t)}_k = \text{ReLU}\!\left(\mathbf{U}_k \begin{bmatrix} s_{1,t,k} \\ \vdots \\ s_{N,t,k} \end{bmatrix}\right) \in \mathbb{R}^H$$

**Why 255?** With $N = 8$ scanners, there are $2^8 = 256$ possible firing patterns.  The all-zero pattern (no scanner fires) carries no information and maps to zero by construction.  The remaining $2^N - 1 = 255$ non-empty patterns each get one neuron.

The full hidden vector concatenates all $K = 8$ slots:

$$\boxed{\mathbf{h}^{(t)} = \begin{bmatrix} \mathbf{h}^{(t)}_1 \\ \vdots \\ \mathbf{h}^{(t)}_K \end{bmatrix} \in \mathbb{R}^{K \cdot H} = \mathbb{R}^{2040}}$$

Slot $k$'s neurons see **only** the scanner values at slot $k$ — never at other slots.  Cross-slot reasoning is deferred to the output layer.

---

## Step 3 — Output Layer

**Weight matrix** $\mathbf{V} \in \mathbb{R}^{2 \times KH}$, bias $\mathbf{b} \in \mathbb{R}^2$  (bias permitted here — the decision threshold may need an offset).

$$\mathbf{y}^{(t)} = \mathbf{V}\,\mathbf{h}^{(t)} + \mathbf{b} \in \mathbb{R}^2$$

Prediction at boundary $t$:

$$\hat{y}^{(t)} = \arg\max\left(\mathbf{y}^{(t)}\right) \in \{0 = \text{no-split},\; 1 = \text{split}\}$$

Default behavior: when all scanners are silent ($\mathbf{S}^{(t)} = \mathbf{0}$), the hidden layer is zero, and the output defaults to **split** — the safe choice.

---

## Step 4 — Training Loss

Weighted cross-entropy to correct the ~4:1 class imbalance (no-split : split):

$$w_0 = 1, \qquad w_1 = \frac{n_{\text{no-split}}}{n_{\text{split}}} \approx 3.7$$

$$\mathcal{L} = -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\; w_{l^{(b,t)}} \log \frac{\exp(y^{(b,t)}_{l^{(b,t)}})}{\displaystyle\sum_{j=0}^{1} \exp(y^{(b,t)}_j)}$$

where $l^{(b,t)} \in \{0,1\}$ is the ground-truth label for batch item $b$ at boundary $t$.

---

## Full Forward Pass (one boundary)

Substituting Steps 0–3, the complete computation from raw characters to logit for boundary $t$:

$$\mathbf{y}^{(t)} = \mathbf{V} \begin{bmatrix} \text{ReLU}\!\left(\mathbf{U}_1\;\text{ReLU}(\mathbf{W}_{:,\,\tilde{c}_{t},\;1})\right) \\ \text{ReLU}\!\left(\mathbf{U}_2\;\text{ReLU}(\mathbf{W}_{:,\,\tilde{c}_{t+1},\;2})\right) \\ \vdots \\ \text{ReLU}\!\left(\mathbf{U}_K\;\text{ReLU}(\mathbf{W}_{:,\,\tilde{c}_{t+K-1},\;K})\right) \end{bmatrix} + \mathbf{b}$$

where $\mathbf{W}_{:,\,\tilde{c}_{t+k},\,k} \in \mathbb{R}^N$ is the column of scanner weights for whichever character appears at slot $k$.

---

## Parameter Count

| Layer | Tensor | Count |
|---|---|---|
| Scanner | $\mathbf{W} \in \mathbb{R}^{N \times V \times K}$ | $8 \times 49 \times 8 = \mathbf{3{,}136}$ |
| Hidden (per slot) | $\mathbf{U}_k \in \mathbb{R}^{H \times N}$, $k=1\ldots K$ | $8 \times (255 \times 8) = \mathbf{16{,}320}$ |
| Output | $\mathbf{V} \in \mathbb{R}^{2 \times KH}$, $\mathbf{b} \in \mathbb{R}^2$ | $2 \times 2040 + 2 = \mathbf{4{,}082}$ |
| **Total** | | **23,538** |
