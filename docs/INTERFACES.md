# NCM-Lite Interface Contracts (`spec-v1`)

**Status:** Approved Shared Contract (Verified against arXiv:2610.14162 & Official Codebase)  
**Version:** 1.0.0-revised  
**Framework:** PyTorch `>=2.2.0`  
**Initial Device Target:** `cpu`  
**Default Tensor Dtype:** `torch.float32` for activations/weights; `torch.int64` for discrete indices.

---

## 1. Global Symbolic Dimensions

| Symbol | Description | Prototype Value (Primary) | Prototype Value (Fallback) | Paper Term |
|---|---|---|---|---|
| $B$ | Batch Size | 1 or 2 (CPU training) | 1 | Global batch size |
| $L$ | Full Sequence Length | 256 | 128 | $L$ in paper |
| $K$ | Number of Hierarchical Scales | 3 | 3 | $K$ in paper |
| $\ell_1, \ell_2, \ell_3$ | Per-Scale Sequence Lengths | 64, 128, 256 | 32, 64, 128 | $\ell_k$ in paper |
| $V_{\text{vocab}}$ | Vocabulary Size (Byte-level) | 256 | 256 | $|\mathcal{T}|$ in paper |
| $|\mathcal{V}_k|$ | Codebook Size per Scale | 256 | 256 | $|\mathcal{V}_k|$ in paper |
| $D_{\text{tok}}$ | Tokenizer Latent Dimension | 64 | 64 | $d$ in paper / `embed_dim` |
| $D_{\text{gen}}$ | Generator Transformer Dimension | 384 | 192 | `n_embd` in paper |
| $N_{\text{heads}}$ | Generator Attention Heads | 6 | 4 | `n_head` in paper |
| $N_{\text{layers}}$ | Generator Transformer Layers | 6 | 4 | `n_layer` in paper |

---

## 2. Stage 1 — Tokenizer & Residual VQ (Owner: Agent 01)

### 2.1 `MultiscaleEncoder`
```python
class MultiscaleEncoder(nn.Module):
    """Encodes byte tokens into initial continuous latent tensor z of length L."""
    def __init__(self, vocab_size: int = 256, embed_dim: int = 64):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor of shape [B, L], dtype torch.int64 (byte tokens in [0, 255]).
        Returns:
            z: torch.Tensor of shape [B, L, embed_dim], dtype torch.float32.
        """
```

### 2.2 `MultiscaleResidualVectorQuantizer`
```python
class MultiscaleResidualVectorQuantizer(nn.Module):
    """
    Performs multiscale residual vector quantization across K scales.
    Implements downsampling Down_k, codebook lookup, reconstruction operator R_k,
    and cumulative reconstruction z_hat^(k) = z_hat^(k-1) + R_k(q_k).
    """
    def __init__(
        self,
        num_levels: int = 3,
        levels: list[int] = [64, 128, 256],
        codebook_size: int = 256,
        embed_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.95,
        nested_drop_prob: float = 0.5
    ):
        ...

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        """
        Args:
            z: continuous latent tensor from encoder of shape [B, L, embed_dim].
        Returns:
            z_hat: final cumulative reconstruction tensor of shape [B, L, embed_dim].
            code_indices: list of K discrete tensors [r_1, ..., r_K],
                          where r_k has shape [B, ell_k], dtype torch.int64.
            losses: dict containing:
                - 'recon_loss': None (computed by outer model)
                - 'commitment_loss': scalar MSE(z, sg[z_hat])
                - 'quantization_loss': scalar MSE(sg[z], z_hat)
                - 'total_vq_loss': scalar total VQ loss
        """

    def get_next_autoregressive_input(
        self,
        level_idx: int,
        z_hat: torch.Tensor,
        z_q_level: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates cumulative reconstruction z_hat with new level's quantized vectors,
        and returns updated z_hat and resized input for next level.
        Matches official implementation in models/quant.py.
        """

    def idx_to_z_q(self, code_indices: list[torch.Tensor]) -> torch.Tensor:
        """Dequantizes discrete code hierarchy to full continuous reconstruction z_hat."""
```

### 2.3 `MultiscaleDecoder`
```python
class MultiscaleDecoder(nn.Module):
    """Reconstructs token logits from final cumulative continuous latent z_hat."""
    def __init__(self, embed_dim: int = 64, vocab_size: int = 256):
        ...

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_hat: torch.Tensor of shape [B, L, embed_dim].
        Returns:
            logits: torch.Tensor of shape [B, L, vocab_size], dtype torch.float32.
        """
```

### 2.4 `TokenizerModel`
```python
class TokenizerModel(nn.Module):
    """Composed Stage 1 Autoencoder containing Encoder, RVQ, and Decoder."""
    def __init__(self, config: dict):
        ...

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        """
        Args:
            x: torch.Tensor of shape [B, L], dtype torch.int64.
        Returns:
            token_logits: torch.Tensor of shape [B, L, 256].
            code_indices: list of K discrete index tensors [r_1, ..., r_K].
            losses: dict with 'recon_loss', 'vq_loss', 'total_loss'.
        """

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns target discrete code indices [r_1, ..., r_K]."""

    @torch.no_grad()
    def decode(self, code_indices: list[torch.Tensor]) -> torch.Tensor:
        """Maps discrete code indices back to token logits [B, L, 256]."""
```

---

## 3. Stage 2 — Next-Concept Predictor (Owner: Agent 02)

### 3.1 `NextConceptTransformer`
```python
class NextConceptTransformer(nn.Module):
    """
    Coarse-to-fine hierarchical transformer generator.
    Implements block-diagonal attention (bidirectional within level, causal across levels).
    """
    def __init__(self, config: dict, tokenizer: TokenizerModel):
        ...

    def forward(
        self,
        target_codes: list[torch.Tensor],
        corruption_rate: float = 0.0
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        """
        Single-pass training step over all K levels (Algorithm 1 in paper).
        Args:
            target_codes: list of ground-truth discrete codes [r_1, ..., r_K].
            corruption_rate: probability of replacing conditioning codes.
        Returns:
            level_logits: list of K tensors [Z_1, ..., Z_K], where
                          Z_k has shape [B, ell_k, |V_k|].
            loss_dict: dict with 'loss_level_1', ..., 'total_ncm_loss'.
        """

    @torch.no_grad()
    def generate(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> list[torch.Tensor]:
        """
        Inference step: Generates K levels in exactly K sequential passes (Algorithm 2).
        Returns:
            generated_codes: list of K tensors [r_1, ..., r_K].
        """
```

---

## 4. Training & Data Pipeline (Owner: Agent 03)

```python
class ByteTextDataset(torch.utils.data.Dataset):
    """Streams or chunks UTF-8 text into fixed-length byte index tensors [L]."""

class SyntheticTextDataset(torch.utils.data.Dataset):
    """Deterministic synthetic dataset for shape, boundary, and memory smoke testing."""

class TokenizerTrainer:
    """Orchestrates Stage 1 optimization."""
    def train_step(self, batch: torch.Tensor) -> dict[str, float]: ...

class GeneratorTrainer:
    """Orchestrates Stage 2 optimization with frozen TokenizerModel."""
    def train_step(self, batch: torch.Tensor) -> dict[str, float]: ...

class CheckpointManager:
    """Atomic saving and loading of model, optimizer, scheduler, and RNG state."""
    @staticmethod
    def save(state: dict, filepath: str) -> None: ...
    @staticmethod
    def load(filepath: str, device: str = "cpu") -> dict: ...
```

---

## 5. Inference & Evaluation (Owner: Agent 04)

```python
def generate_hierarchical_codes(
    generator: NextConceptTransformer,
    temperature: float = 1.0,
    top_k: int = 0
) -> list[torch.Tensor]:
    """Runs exactly K parallel forward passes to generate all concept codes."""

def decode_hierarchical_codes(
    tokenizer: TokenizerModel,
    generated_codes: list[torch.Tensor]
) -> str:
    """Decodes hierarchical codes to UTF-8 text via frozen tokenizer decoder."""

def evaluate_reconstruction(
    tokenizer: TokenizerModel,
    dataloader: torch.utils.data.DataLoader
) -> dict[str, float]:
    """Calculates byte reconstruction accuracy and codebook utilization / perplexity."""

def evaluate_generation(
    generator: NextConceptTransformer,
    tokenizer: TokenizerModel,
    num_samples: int = 5
) -> dict[str, Any]:
    """Measures latency, memory usage, and samples from end-to-end generation."""
```
