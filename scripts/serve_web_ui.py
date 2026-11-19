"""
NCM-Lite Interactive Studio Web Server.
Serves a responsive single-page browser studio for real-time coarse-to-fine
text generation and prompt continuation using standard Python libraries.
"""

import os
import sys
import time
import json
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import torch

from ncm.tokenizer.model import TokenizerModel
from ncm.models.next_concept import NextConceptTransformer
from ncm.training.checkpoint import CheckpointManager
from ncm.inference.generate import (
    generate_hierarchical_progression,
    generate_prompt_continuation,
    decode_latent_to_text,
    _sample_from_logits
)


class NCMStudioServer:
    def __init__(self, base_dir: str, device: str = "cpu"):
        self.base_dir = base_dir
        self.device = device
        self.levels = [64, 128, 256]
        self.seq_len = 256

        print("[*] Initializing NCM-Lite models for interactive studio...")
        self.tokenizer = TokenizerModel(
            vocab_size=256,
            embed_dim=64,
            max_seq_len=self.seq_len,
            num_levels=3,
            levels=self.levels,
            codebook_size=256,
            commitment_cost=0.25,
            decay=0.95,
            nested_drop_prob=0.0
        ).to(device)

        self.generator = NextConceptTransformer(
            num_levels=3,
            levels=self.levels,
            codebook_size=256,
            latent_dim=64,
            embed_dim=384,
            num_layers=6,
            num_heads=6,
            intermediate_dim=1536,
            dropout=0.0,
            loss_alpha=0.5
        ).to(device)

        gen_ckpt = os.path.join(base_dir, "checkpoints", "ncm_real_text_model.pt")
        tok_ckpt = os.path.join(base_dir, "checkpoints", "ncm_real_text_tokenizer.pt")

        if os.path.exists(gen_ckpt):
            print(f"[+] Loaded Generator from: {gen_ckpt}")
            CheckpointManager.load(gen_ckpt, model=self.generator, device=device)
        if os.path.exists(tok_ckpt):
            print(f"[+] Loaded Tokenizer from: {tok_ckpt}")
            CheckpointManager.load(tok_ckpt, model=self.tokenizer, device=device)

        self.generator.eval()
        self.tokenizer.eval()
        print("[✓] NCM-Lite Studio models loaded and ready.")

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.92
    ) -> dict:
        t0 = time.perf_counter()

        prompt = prompt if prompt else ""

        if not prompt:
            # Unconditional coarse-to-fine progression
            _, _, texts = generate_hierarchical_progression(
                generator=self.generator,
                tokenizer=self.tokenizer,
                batch_size=1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=self.device
            )
            l1_text = texts[0][0]
            l2_text = texts[1][0]
            l3_text = texts[2][0]
        else:
            # Prompt-conditioned coarse-to-fine progression
            L = self.tokenizer.vq.target_len
            prompt_bytes = list(prompt.encode("utf-8", errors="replace"))
            P = min(len(prompt_bytes), L - 16)
            padded = prompt_bytes[:P] + [0] * (L - P)
            x_prompt = torch.tensor([padded], dtype=torch.long, device=self.device)
            prompt_codes = self.tokenizer.encode(x_prompt)

            z_hat = torch.zeros(1, L, self.tokenizer.vq.embed_dim, device=self.device)
            prev_u: list = []
            progress_texts: list = []

            for k in range(self.generator.num_levels):
                ell_k = self.generator.levels[k]
                P_k = max(1, min(ell_k - 1, round(P * (ell_k / L))))
                u_k = self.tokenizer.vq._down_to_level(z_hat, ell_k)
                logits_k = self.generator.predict_single_level(u_k=u_k, level_idx=k, all_prev_u=prev_u)

                r_k_pred = _sample_from_logits(logits_k, temperature=temperature, top_k=top_k, top_p=top_p)
                r_k = torch.cat([prompt_codes[k][:, :P_k], r_k_pred[:, P_k:]], dim=1)

                q_k = self.tokenizer.vq.embeddings[k](r_k)
                r_full = self.tokenizer.vq._up_to_full(q_k, L)
                z_hat = z_hat + r_full
                prev_u.append(u_k)

                # Decode progressive snapshot with anchored prompt
                raw_snapshot = decode_latent_to_text(self.tokenizer, z_hat)[0]
                snapshot_text = prompt + raw_snapshot[P:]
                progress_texts.append(snapshot_text)

            l1_text = progress_texts[0]
            l2_text = progress_texts[1]
            l3_text = progress_texts[2]

        elapsed = time.perf_counter() - t0

        return {
            "prompt": prompt,
            "latency_seconds": elapsed,
            "throughput_bytes_per_sec": float(self.seq_len / max(1e-6, elapsed)),
            "level_1_text": l1_text,
            "level_2_text": l2_text,
            "level_3_text": l3_text,
            "final_text": l3_text
        }


def make_request_handler(studio: NCMStudioServer, html_path: str):
    class StudioHTTPRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            elif self.path == "/api/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = {
                    "status": "healthy",
                    "model": "NCM-Lite (11.3M)",
                    "device": studio.device,
                    "levels": studio.levels
                }
                self.wfile.write(json.dumps(response).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/api/generate":
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8")
                try:
                    payload = json.loads(body) if body else {}
                except Exception:
                    payload = {}

                prompt = payload.get("prompt", "")
                temperature = float(payload.get("temperature", 0.8))
                top_k = int(payload.get("top_k", 40))
                top_p = float(payload.get("top_p", 0.92))

                result = studio.generate(
                    prompt=prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Clean logging
            sys.stdout.write(f"[{self.log_date_time_string()}] {format % args}\n")

    return StudioHTTPRequestHandler


def run_server(port: int = 8000):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base_dir, "src", "ncm", "web", "index.html")

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"Studio HTML not found at: {html_path}")

    studio = NCMStudioServer(base_dir=base_dir, device="cpu")
    handler = make_request_handler(studio, html_path)

    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print("\n" + "=" * 70)
    print(f"NCM-LITE INTERACTIVE STUDIO RUNNING")
    print(f"URL: http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop the server.")
    print("=" * 70 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping NCM-Lite Studio Server...")
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NCM-Lite Interactive Studio Web Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    args = parser.parse_args()
    run_server(port=args.port)
