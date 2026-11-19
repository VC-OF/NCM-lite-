"""
Live Interactive NCM Studio Test Script.
Runs the exact interactive server generation engine across multiple test prompts,
tracks latency and throughput, and inspects the live coarse-to-fine outputs.
"""

import os
import sys
import time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from scripts.serve_web_ui import NCMStudioServer


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    studio = NCMStudioServer(base_dir=base_dir, device="cpu")

    test_cases = [
        {
            "category": "In-Domain Story Prompt #1",
            "prompt": "Once upon a time, Lily found a ",
            "temp": 0.8, "top_k": 40, "top_p": 0.92
        },
        {
            "category": "In-Domain Story Prompt #2",
            "prompt": "Tim and his dog Spot ran to the ",
            "temp": 0.8, "top_k": 40, "top_p": 0.92
        },
        {
            "category": "Out-of-Domain Knowledge Prompt",
            "prompt": "what is english",
            "temp": 0.8, "top_k": 40, "top_p": 0.92
        },
        {
            "category": "Unconditional Generation (Blank Prompt)",
            "prompt": "",
            "temp": 0.8, "top_k": 40, "top_p": 0.92
        },
        {
            "category": "Greedy Sampling Check (T=0.0)",
            "prompt": "One sunny morning, the little ",
            "temp": 0.0, "top_k": 0, "top_p": 1.0
        }
    ]

    print("\n" + "=" * 76)
    print("LIVE INTERACTIVE NCM-LITE ENGINE BENCHMARK")
    print("=" * 76)

    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/5] TEST CATEGORY: {tc['category']}")
        print(f"  • Prompt: {repr(tc['prompt'])}")
        print(f"  • Parameters: T={tc['temp']}, top_k={tc['top_k']}, top_p={tc['top_p']}")

        res = studio.generate(
            prompt=tc["prompt"],
            temperature=tc["temp"],
            top_k=tc["top_k"],
            top_p=tc["top_p"]
        )

        lat_ms = res["latency_seconds"] * 1000
        thr_bps = res["throughput_bytes_per_sec"]
        print(f"  • Latency: {lat_ms:.1f} ms | Throughput: {thr_bps:.0f} bytes/sec (3 parallel passes)")

        print("  ┌── [Pass 1/3: Level 1 (ell_1 = 64 tokens)] ──────────────────────────────")
        l1_clean = res["level_1_text"][:140].replace("\n", " ")
        print(f"  │ \"{l1_clean}...\"")

        print("  ├── [Pass 2/3: Level 2 (ell_2 = 128 tokens)] ─────────────────────────────")
        l2_clean = res["level_2_text"][:140].replace("\n", " ")
        print(f"  │ \"{l2_clean}...\"")

        print("  └── [Pass 3/3: Level 3 (ell_3 = 256 tokens) — Final Result] ─────────────")
        l3_clean = res["level_3_text"][:180].replace("\n", " ")
        print(f"    \"{l3_clean}...\"")

    print("\n" + "=" * 76)
    print("ALL INTERACTIVE BENCHMARKS COMPLETED")
    print("=" * 76)


if __name__ == "__main__":
    main()
