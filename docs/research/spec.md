# NCM-Lite Architecture & Technical Specification (`spec-v1`)

**Document Status:** Complete Architecture Specification (Source-Verified against arXiv:2610.14162 & Official Codebase)  
**Version:** 1.0.0-revised (Candidate for `spec-v1` tag)  
**Author:** Agent 00 (Research Architect)  
**Target Hardware:** Consumer Laptop (x86-64 CPU, 13.3 GB usable RAM, no discrete GPU required)

---

## 1. Scope and Non-Goals

### Project Goals
1. Implement a self-contained, CPU-trainable prototype faithful to the core mathematical principles of Stanford's Next-Concept Modeling (NCM) paper (arXiv:2610.14162).
2. Validate the central two-stage architecture:
   - **Stage 1:** Multiscale Residual Vector-Quantized (RVQ) Tokenizer with nested level dropout and cumulative reconstruction.
   - **Stage 2:** Frozen-tokenizer Coarse-to-Fine Next-Concept Transformer Predictor operating via block-causal attention.
3. Validate parallel level-wise generation (all codes in a level predicted simultaneously in a single forward pass; sequence finishes in exactly $K$ passes).
4. Maintain strict modularity across four implementation domains: Tokenizer, Model, Training, and Evaluation.
5. Provide comprehensive synthetic smoke tests and verifiable CPU execution within an 8.0 GB RAM envelope.

### Explicit Non-Goals
1. We are **not** reproducing Stanford's 100M+ parameter models, GPU cluster training, or web-scale training runs.
2. We are **not** building a general-purpose chat LLM.
3. We do **not** use external black-box tokenizers (like HuggingFace GPT-2 BPE) as a substitute for multiscale RVQ concept tokenization.
4. We do **not** use ordinary left-to-right token generation or continuous latent diffusion.

---

## 2. Mathematical Architecture

```mermaid
flowchart TD
    subgraph Stage1["Stage 1: Multiscale RVQ Tokenizer (arXiv:2610.14162 Sec 3.3)"]
        X["Input Text Tokens X [B, L]"] --> Enc["Encoder (Embedding + Positional Lookup)"]
        Enc --> Z["Full Latent z in R^[B, L, D_tok]"]
        Z --> Down1["Down_1 (Area-Pool to ell_1)"]
        Down1 --> U1["u_1 [B, ell_1, D_tok]"]
        U1 --> Q1["Quantize to Codebook V_1"]
        Q1 --> R1["Reconstruction Operator R_1 (Linear-Up to L)"]
        R1 --> Zhat1["Cumulative z_hat^(1) [B, L, D_tok]"]
        Zhat1 --> Res2["Residual z^(1) = z - z_hat^(1)"]
        Res2 --> Down2["Down_2 (Area-Pool to ell_2)"]
        Down2 --> U2["u_2 [B, ell_2, D_tok]"]
        U2 --> Q2["Quantize to Codebook V_2"]
        Q2 --> R2["Reconstruction Operator R_2 (Linear-Up to L)"]
        R2 --> Zhat2["Cumulative z_hat^(2) [B, L, D_tok]"]
        Zhat2 --> Res3["Residual z^(2) = z - z_hat^(2)"]
        Res3 --> Q3["Quantize to Codebook V_3"]
        Q3 --> Zhat3["Cumulative z_hat^(K) [B, L, D_tok]"]
        Zhat3 --> Dec["Decoder (1 Block / Linear Projection)"]
        Dec --> X_hat["Reconstructed Token Logits [B, L, V_vocab]"]
    end

    subgraph Stage2["Stage 2: Next-Concept Transformer (arXiv:2610.14162 Sec 3.4)"]
        Zhat_prev["Resized Cumulative Reconstruction z_hat^(k-1) [B, ell_k, D_tok]"] --> Proj["Linear Projection wte + Level/Pos Embeddings"]
        Proj --> Trans["Next-Concept Transformer (Block-Diagonal / Block-Causal Mask)"]
        Trans --> Logits["Logits for Level k [B, ell_k, |V_k|]"]
        Logits --> ParallelSample["Parallel Code Sampling r_k in {1, ..., |V_k|}^ell_k"]
        ParallelSample --> Update["Update z_hat^(k) = z_hat^(k-1) + R_k(e_k(r_k))"]
    end
```

---

## 3. Dimensional Parameters & Hardware Budget

### Prototype Architecture Parameters

| Parameter | Symbol | Primary Spec (11.9M Params) | Fallback Spec (3.1M Params) | Paper Source & Notes |
|---|---|---|---|---|
| Vocabulary Size | $V_{\text{vocab}}$ | 256 (byte-level) | 256 (byte-level) | `[PROTOTYPE DECISION]`: Self-contained UTF-8 byte vocabulary |
| Target Length | $L$ | 256 | 128 | Fixed token sequence length |
| Multiscale Levels | $K$ | 3 | 3 | `[PAPER]`: Coarse to fine ($K=3$ for prototype; paper uses $8$–$32$) |
| Sequence Lengths | $\ell_k$ | $[64, 128, 256]$ | $[32, 64, 128]$ | `[PAPER]` Eq. 3: $\ell_1 < \ell_2 < \dots < \ell_K = L$ |
| Codebook Size | $|\mathcal{V}_k|$ | 256 per level | 256 per level | `[PAPER]` Table 5: distinct codebook per level |
| Code/Latent Dim | $D_{\text{tok}}$ | 64 | 64 | `[OFFICIAL CODE]` `embed_dim` in `models/vqvae.py` |
| Generator Layers | $N_{\text{layers}}$ | 6 | 4 | `[PROTOTYPE DECISION]`: Transformer blocks for predictor |
| Generator Hidden Dim | $D_{\text{gen}}$ | 384 | 192 | `[PROTOTYPE DECISION]`: Scaled for CPU memory |
| Generator Heads | $N_{\text{heads}}$ | 6 | 4 | Head dimension: 64 |
| Generator MLP Dim | $D_{\text{mlp}}$ | 1536 ($4 \times D$) | 768 ($4 \times D$) | Standard 4x expansion |
| Commitment Beta | $\beta$ | 0.25 | 0.25 | `[PAPER]` Eq. 9: $\beta = 0.25$ |
| Nested Dropout Prob | $p_{\text{drop}}$ | 0.5 | 0.5 | `[PAPER]` Table 5: 0.5 |

### Memory Budget & Parameter Breakdown (Primary Model)

1. **Parameter Count Calculation:**
   - Tokenizer Model:
     - Embedding ($256 \times 64$): $16.4\text{K}$
     - 3 Codebooks ($3 \times 256 \times 64$): $49.1\text{K}$
     - Decoder (1 Transformer Block or Linear Head, $64 \times 256$): $\approx 150\text{K}$
     - **Tokenizer Total:** $\approx 0.25\text{M} - 0.63\text{M}$ parameters.
   - Generator Transformer:
     - Projection `wte` ($64 \times 384$): $24.5\text{K}$
     - Position + Level Embeddings: $448 \times 384 + 3 \times 384 \approx 173\text{K}$
     - 6 Transformer Layers ($6 \times [4 \times 384^2 + 2 \times 384 \times 1536]$): $\approx 10.63\text{M}$
     - Prediction Heads ($3 \times 384 \times 256$): $\approx 295\text{K}$
     - **Generator Total:** $\approx 11.2\text{M}$ parameters.
   - **Combined Total:** $\approx 11.8\text{M}$ parameters (matches 10–15M ceiling).

2. **Process RAM Footprint (CPU, Batch Size = 2):**
   - Model weights (FP32): $12\text{M} \times 4\text{ bytes} \approx 48\text{ MB}$.
   - Gradients: $48\text{ MB}$.
   - AdamW states: $96\text{ MB}$.
   - Activations: $\approx 50\text{ MB}$.
   - Python runtime + PyTorch overhead: $\approx 350\text{ MB}$.
   - **Total Estimated Training Memory:** $\approx 600\text{ MB} - 1.2\text{ GB}$.
   - **Conclusion:** Safe for 13.3 GB system RAM and 8.0 GB process ceiling.

---

## 4. Stage 1: Multiscale RVQ Tokenizer Details

### Downsampling & Upsampling Operators (`[PAPER]` Section 3.3 & `[OFFICIAL CODE]`)
- For residual $z^{(k-1)} \in \mathbb{R}^{B \times L \times D_{\text{tok}}}$:
  $$\text{Down}_k(z^{(k-1)}) = \text{AreaPool1D}(z^{(k-1)}, \text{target\_size}=\ell_k) \in \mathbb{R}^{B \times \ell_k \times D_{\text{tok}}}$$
- For code vectors $q_k \in \mathbb{R}^{B \times \ell_k \times D_{\text{tok}}}$:
  $$R_k(q_k) = \text{LinearInterpolate1D}(q_k, \text{target\_size}=L) \in \mathbb{R}^{B \times L \times D_{\text{tok}}}$$
- Cumulative reconstruction update:
  $$\hat{z}^{(k)} = \hat{z}^{(k-1)} + R_k(q_k), \quad z^{(k)} = z - \hat{z}^{(k)}$$

### Nested Dropout (`[PAPER]` Section 3.3, Eq. 9)
- During training, sample a level cutoff $k \in \{1, \dots, K\}$ with probability $p=0.5$.
- Compute reconstruction loss against $\hat{z}^{(k)}$ to force coarse levels to carry self-sufficient semantics:
  $$\mathcal{L}_{\text{tok}} = -\frac{1}{L}\sum_{i=1}^L \log p_\psi(x^i \mid \hat{z}^{(k)}) + \beta \text{MSE}(z, \text{sg}[\hat{z}^{(k)}]) + \text{MSE}(\text{sg}[z], \hat{z}^{(k)})$$

---

## 5. Stage 2: Next-Concept Transformer Details

### Input Conditioning (`[OFFICIAL CODE]` `models/next_concept.py` line 665)
- Input at level $k$ is the resized cumulative reconstruction $\hat{z}^{(k-1)}$:
  $$x_k = \text{LinearProj}(\text{Down}_k(\hat{z}^{(k-1)})) + \mathbf{E}_{\text{level}}[k] + \mathbf{E}_{\text{pos}} \in \mathbb{R}^{B \times \ell_k \times D_{\text{gen}}}$$

### Block-Diagonal / Block-Causal Masking (`[PAPER]` Figure 4b & `[OFFICIAL CODE]`)
- During training: all $K$ levels are concatenated: $L_{\text{total}} = \sum_{k=1}^K \ell_k = 448$.
- Mask $M_{i, j} = 0.0$ if $k(j) \le k(i)$; $M_{i, j} = -\infty$ if $k(j) > k(i)$.
- Positions within the same level attend bidirectionally; across levels, attention is strictly causal.

### Loss Function (`[PAPER]` Eq. 10)
$$\mathcal{L}_{\text{NCM}} = -\sum_{k=1}^K w_k \sum_{i=1}^{\ell_k} \log p_\theta(r_k^i \mid \tilde{z}^{(k-1)}), \quad w_k = \frac{\ell_k^{-\alpha}}{\sum_{j=1}^K \ell_j^{1-\alpha}}$$
- $\alpha = 0$ gives equal position weight; $\alpha = 1$ gives equal level weight.

---

## 6. Inference Sequence (Algorithm) (`[PAPER]` Section 3.4, Algorithm 2)

```text
Algorithm 2: NCM-Lite Coarse-to-Fine Parallel Inference
Input: Start token / condition prefix (optional), Temperature tau, Top-k
Output: Reconstructed UTF-8 text string

1. Initialize cumulative reconstruction: z_hat = zeros(B, L, D_tok)
2. For k = 1 to K (K = 3):
   a. Resize z_hat to ell_k positions: u = Down_k(z_hat)
   b. Project: h_in = LinearProj(u) + E_level[k] + E_pos
   c. Run ONE forward pass of NextConceptTransformer
   d. Extract logits for level k: Z_k in R^[B x ell_k x |V_k|]
   e. In parallel for all i in {1, ..., ell_k}:
      Sample r_{k, i} ~ Softmax(Z_{k, i} / tau) (with optional top-k truncation)
   f. Lookup codebook vectors: q_k = lookup(C_k, r_k)
   g. Update cumulative reconstruction: z_hat = z_hat + R_k(q_k)
3. Pass final cumulative reconstruction z_hat through frozen Stage 1 Decoder:
   logits = Decoder(z_hat) in R^[B x L x 256]
   token_ids = argmax(logits, dim=-1)
4. Convert token_ids bytes to UTF-8 string (errors='replace')
5. Return generated text
```

**Total forward passes:** Exactly 3 generator forward passes + 1 decoder forward pass.
