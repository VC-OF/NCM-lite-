# Research Notes: Stanford Next-Concept Modeling (NCM)

**Document Status:** Grounded Source Notes (Verified against Full Paper HTML & Official Codebase)  
**Author:** Agent 00 (Research Architect)  
**Verification Date:** 18 November 2026  

---

## 1. Primary Sources Inspected & Stored Locally

The following primary sources were fetched in full, verified, and are mirrored locally under `docs/research/sources/`:

1. **Full arXiv Paper (HTML & LaTeX Text)**:
   - Source URL: `https://arxiv.org/html/2610.14162`
   - Identifier: arXiv:2610.14162v1 (cs.CL, submitted 09 Oct 2026)
   - Title: *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*
   - Authors: Hermann Kumbong, Ishan S. Khare, Iddah Mlauzi, Andrew Woen, Charles Ding, E. Kelly Buchanan, Simran Arora, Azalia Mirhoseini, Christopher Ré (Stanford University & Together AI)
   - Local Copy: `docs/research/sources/arxiv_2610.14162.html` (1.86 MB)
2. **Official Repository Structure & Files**:
   - Source: `https://github.com/HazyResearch/ncm`
   - Files fetched and inspected:
     - `README.md` (`docs/research/sources/github_ncm_README.md`)
     - `models/vqvae.py` (`docs/research/sources/models_vqvae.py`)
     - `models/quant.py` (`docs/research/sources/models_quant.py`)
     - `models/next_concept.py` (`docs/research/sources/models_next_concept.py`)
     - `config/eval/tinystories.yaml` (`docs/research/sources/tinystories_eval_config.yaml`)
     - `config/eval/sudoku-easy.yaml`
3. **Official Blog Post**:
   - `https://hazyresearch.stanford.edu/blog/2026-10-12-ncm`

---

## 2. Answers to the 4 Core Verification Questions

### Q1: Does generation really take $K$ parallel passes, one per level?
- **Status:** `[PAPER]` & `[OFFICIAL CODE]` (Fully Verified)
- **Paper Text (Section 3.4, Algorithm 2)**:
  ```text
  Algorithm 2 NCM generation
  1: Frozen tokenizer, generator p_theta
  2: \hat{z}^(0) <- 0
  3: for k = 1, ..., K do
  4:   Resize \hat{z}^(k-1) to l_k positions
  5:   Project the resized vectors to the hidden width
  6:   Predict code distributions in one forward pass
  7:   Sample all l_k codes r_k^i in parallel
  8:   \hat{z}^(k) <- \hat{z}^(k-1) + R_k(e_k(r_k))
  9: end for
  10: Decode \hat{z}^(K) to token distributions
  11: Select output tokens by argmax
  ```
- **Code Reference**: `models/next_concept.py`, lines 660–685 (`generate()` method).
- **Finding**: Generation takes **exactly $K$ sequential passes of the generator, plus 1 final tokenizer decode pass**. In pass $k$, all $\ell_k$ discrete code positions are sampled simultaneously from a single transformer forward pass.

---

### Q2: How are the scales and downsampling defined?
- **Status:** `[PAPER]` & `[OFFICIAL CODE]` (Fully Verified)
- **Paper Text (Section 3.3, Equations 7 & 8)**:
  - Sequence token length is $L$.
  - Discrete codes exist at $K$ levels with lengths $\ell_1 < \ell_2 < \dots < \ell_K = L$.
  - Downsampling operator $\text{Down}_k(z^{(k-1)}) \in \mathbb{R}^{\ell_k \times d}$: downsamples the full-length residual $z^{(k-1)}$ to resolution $\ell_k$.
  - Reconstruction operator $R_k(e_k(r_k)) \in \mathbb{R}^{L \times d}$: maps code vectors of length $\ell_k$ back to full length $L$.
- **Code Reference**: `models/quant.py`:
  - `Down_k`: lines 310–330, implemented via 1D area pooling:  
    `F.interpolate(z.transpose(-1, -2), size=(ell_k,), mode="area").transpose(-1, -2)`
  - `R_k` (Upsampling): lines 420–435, implemented via 1D linear interpolation:  
    `F.interpolate(q.transpose(-1, -2), size=(L,), mode="linear").transpose(-1, -2)` followed by optional residual convolution `Phi`.
  - Levels schedule in official configs (`config/eval/tinystories.yaml`):  
    `manual_levels: [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]` (12 levels).

---

### Q3: Is the loss per level, and how does level dropout work?
- **Status:** `[PAPER]` & `[OFFICIAL CODE]` (Fully Verified)
- **Stage 1 Tokenizer Loss & Nested Dropout (Paper Section 3.3, Eq. 9 & `models/quant.py` line 262)**:
  - During training, nested dropout randomly samples a level cutoff $k \in \{1, \dots, K\}$ with dropout probability (e.g. 0.5), discarding finer levels $j > k$.
  - The cumulative reconstruction at that cutoff is $\hat{z}^{(k)} = \sum_{j=1}^k R_j(e_j(r_j))$.
  - Tokenizer loss is:
    $$\mathcal{L}_{\text{tok}} = -\frac{1}{L}\sum_{i=1}^L \log p_\psi(x^i \mid \hat{z}^{(k)}) + \beta \text{MSE}(z, \text{sg}[\hat{z}^{(k)}]) + \text{MSE}(\text{sg}[z], \hat{z}^{(k)})$$
  - Codebook updates: In official code (`models/quant.py` lines 297–304), codebooks are updated via **EMA (Exponential Moving Average, decay 0.95–0.99)** rather than gradient descent on codebook vectors. Gradients pass to the encoder via Straight-Through Estimator (STE).
- **Stage 2 Generator Loss (Paper Section 3.4, Eq. 10)**:
  $$\mathcal{L}_{\text{NCM}} = -\sum_{k=1}^K w_k \sum_{i=1}^{\ell_k} \log p_\theta(r_k^i \mid \tilde{z}^{(k-1)}), \quad w_k = \frac{\ell_k^{-\alpha}}{\sum_{j=1}^K \ell_j^{1-\alpha}}$$
  - Conditioned on corrupted cumulative reconstructions $\tilde{z}^{(k-1)}$ to prevent exposure bias.

---

### Q4: Is the transformer block-causal in the way the spec describes?
- **Status:** `[PAPER]` & `[OFFICIAL CODE]` (Fully Verified)
- **Paper Text (Section 3, Figure 4b caption & Section 3.4)**:
  - *"In the unconditional setting, block-diagonal attention restricts each input block to itself. During training, all $K$ levels are predicted in one forward pass... During generation, levels are predicted sequentially from coarse to fine."*
- **Code Reference**: `models/next_concept.py` line 77 (`attn_pattern = 'block_diagonal'`) and lines 507–510 (`gen_mask[:, :, :P, P:] = -torch.inf`).
- **Finding**: Inside level $k$, attention is bidirectional across all positions within that level. Across levels, attention is causal: level $k$ attends only to the prefix and previous levels $1, \dots, k-1$.

---

## 3. Official Repository Component Audit

| Component | Paper & Code Status | NCM-Lite Prototype Decision |
|---|---|---|
| **Tokenizer Architecture** | `enc_layers=0`, `dec_layers=1` (`models/vqvae.py` line 41). Encoder is word token embedding + positional embedding; decoder is 1 Transformer block. | `prototype adaptation`: We use byte-level embedding ($V=256$) + 1 residual conv/linear block, decoder is linear projection or 1 lightweight block. |
| **Quantization** | `MultiscaleResidualVectorQuantizer` with EMA codebooks (`models/quant.py` lines 228–305). | `paper-faithful`: Residual vector quantization with EMA updates or standard VQ loss with STE. |
| **Downsampling / Upsampling** | Area-pooling (`Down_k`) and linear interpolation (`R_k`) (`models/quant.py` lines 310–330, 420–435). | `paper-faithful`: Factor-of-2 / 1D pooling downsampling and repeat-interleave or linear upsampling. |
| **Generator Transformer** | Standard Transformer with block-diagonal attention, level embeddings (`wle`), and tied codebook heads (`models/next_concept.py`). | `paper-faithful`: Transformer predictor with block-causal attention and level indicators. |
| **Generator Input Conditioning** | Resized cumulative continuous reconstruction $\hat{z}^{(k-1)}$ projected through linear layer `wte` (`models/next_concept.py` lines 664–670). | `paper-faithful`: Generator takes projected cumulative reconstruction $\hat{z}^{(k-1)}$ as input rather than raw integer indices. |
| **Generation Loop** | Exactly $K$ parallel forward passes (`models/next_concept.py` lines 660–685). | `paper-faithful`: Strictly $K$ parallel generator passes followed by 1 decoder pass. |
