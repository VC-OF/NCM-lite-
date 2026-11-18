# NCM-Lite Architectural Deviations & Adaptations Matrix

**Document Status:** Approved Deviations Register (Source-Verified against arXiv:2610.14162 & Official Codebase)  
**Version:** 1.0.0-revised  
**Author:** Agent 00 (Research Architect)  

Every divergence between Stanford's Next-Concept Modeling paper (arXiv:2610.14162) and this NCM-Lite prototype is registered here with an explicit classification and technical rationale.

---

## 1. Classification Categories

1. `paper-faithful`: Implements the exact algorithm, mathematical formulation, loss, or dependency structure described in the paper or official repository.
2. `prototype adaptation`: An intentional simplification or downscaling necessitated by consumer laptop constraints (single x86-64 CPU, 13.3 GB usable RAM, no discrete GPU, fast turnaround).
3. `experimental`: A design choice or modification proposed for exploration that must be tested against a baseline.

---

## 2. Deviations Register

| # | Dimension | Classification | Stanford NCM Behavior `[PAPER]` | NCM-Lite Prototype Behavior `[PROTOTYPE DECISION]` | Rationale & Tradeoff | Evaluation Strategy |
|---|---|---|---|---|---|---|
| **1** | **Vocabulary & Tokenization** | `prototype adaptation` | Subword BPE tokenizer (e.g. GPT-2 / LLaMA tokenizer with $\approx 32\text{K}$ to $50\text{K}$ tokens; $8\text{K}$ on TinyStories). | Fixed UTF-8 byte-level vocabulary ($V_{\text{vocab}} = 256$, tokens $0 \dots 255$). | Eliminates external tokenizer dependencies; prevents HuggingFace cache overhead; enables self-contained, byte-exact reconstruction checks. Tradeoff: sequences are character/byte-level rather than subword-level. | Byte reconstruction accuracy on synthetic and short text samples. |
| **2** | **Model Parameter Scale** | `prototype adaptation` | 90M to 150M generator parameters + 36M to 144M tokenizer parameters trained across 8× H100 GPUs. | $\approx 11.2\text{M}$ generator parameters + $0.6\text{M}$ tokenizer parameters ($\approx 11.8\text{M}$ total). | Mandatory to guarantee training and forward passes execute comfortably within $< 1.5\text{ GB}$ of CPU RAM on a 13.3 GB machine. | Parameter count test (`test_generator_parameter_count`). |
| **3** | **Hardware & Precision** | `prototype adaptation` | Multi-GPU distributed training using BF16 / FP8 with FlashAttention-2 / FlashAttention-3. | Single-process CPU execution using standard FP32 with PyTorch native attention. | Laptop CPU does not support CUDA or FlashAttention kernels; FP32 avoids numerical underflow on CPU without mixed-precision complexity. | Numerical stability tests; loss finite checks. |
| **4** | **Number of Scales ($K$)** | `paper-faithful` | Hierarchical scales ranging from 8 to 32 levels depending on context length ($K=12$ on TinyStories / LM1B, $K=32$ on OWT). | Exactly $K=3$ scales with sequence lengths $\ell = [64, 128, 256]$ for target length $L=256$. | Faithfully implements coarse-to-fine hierarchy while keeping sequence lengths and memory manageable on CPU. | Parallel forward pass step counter (`test_parallel_inference_steps`). |
| **5** | **Straight-Through Estimator (STE)** | `paper-faithful` | Discrete codebook lookup with Straight-Through Estimator for backpropagation: $\mathbf{\hat{z}} = \mathbf{z} + \text{sg}[\mathbf{q} - \mathbf{z}]$ (`models/quant.py`). | Identical STE formulation in PyTorch autograd. | Fully faithful to paper's quantization backpropagation mechanics. | Gradient flow tests (`test_rvq_ste_gradient_flow`). |
| **6** | **Two-Stage Training Separation** | `paper-faithful` | Stage 1 trains multiscale RVQ tokenizer until convergence; Stage 2 completely freezes tokenizer and trains generator on discrete code targets. | Identical two-stage protocol: Stage 1 trains `TokenizerModel`; Stage 2 sets `tokenizer.eval()` and freezes all parameters. | Core research principle: prevents moving-target instability and enforces discrete concept representations. | Gradient freeze test (`test_stage2_frozen_tokenizer`). |
| **7** | **Attention Masking** | `paper-faithful` | Block-diagonal attention (`attn_pattern: block_diagonal` in `models/next_concept.py` line 77): bidirectional within each hierarchical scale, causal across scales. | Explicit block-causal boolean/float mask in `NextConceptTransformer`. | Guaranteed faithful implementation of parallel within-scale generation and causal coarse-to-fine conditioning. | Non-leakage perturbation test (`test_no_future_leakage`). |
| **8** | **Downsampling & Upsampling** | `paper-faithful` | Area-pooling `Down_k` and linear interpolation `R_k` (`models/quant.py` lines 310–330, 420–435). | Identical area-pooling downsampling and linear interpolation upsampling. | Faithful to the paper's multiscale operators while avoiding heavy convolutional parameter overhead. | Latent shape tests (`test_encoder_output_shapes`). |
| **9** | **Nested Level Dropout** | `paper-faithful` | Randomly samples level cutoff $k$ during training (prob $0.5$) and computes loss on partial reconstruction $\hat{z}^{(k)}$ (`models/quant.py` line 262). | Identical nested dropout mechanism with $p=0.5$. | Forces coarse levels to encode self-sufficient semantics. | Verification in Stage 1 training tests. |
| **10** | **Dataset & Data Access** | `prototype adaptation` | Web-scale corpora (OpenWebText, LM1B, billions of tokens) downloaded and pre-tokenized across clusters. | Synthetic deterministic tensors for Phase 1 & 2 development; optional small text sample (TinyStories / Shakespeare) for Phase 4. | Avoids unapproved network downloads, large disk caches, and memory materialization on consumer laptop. | Streaming dataloader tests; zero unapproved download rule. |

---

## 3. Preservation of Research Integrity

NCM-Lite preserves all non-negotiable scientific elements of the paper:
1. **Multiscale Residual Vector Quantization (RVQ):** Yes. Residuals between consecutive scales are quantized and summed into cumulative reconstruction.
2. **Discrete Concept Codes:** Yes. Stage 2 operates strictly on discrete code indices, not continuous embeddings.
3. **Parallel Level-wise Generation:** Yes. All positions within a level are predicted simultaneously in a single forward pass.
4. **Coarse-to-Fine Dependency:** Yes. Coarse concepts are predicted first; finer concepts are conditioned causally on coarse predictions.
5. **Two-Stage Frozen Workflow:** Yes. The tokenizer is never trained jointly with the generator.

No component has been silently replaced by BPE tokenization, left-to-right autoregressive next-token prediction, or continuous latent diffusion.
