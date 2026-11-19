"""
Prepare expanded TinyStories dataset for NCM-Lite training.
Downloads a clean 512 KB chunk of authentic TinyStories data.
"""

import os
import urllib.request

def prepare_data(target_size_bytes: int = 524288) -> str:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(root_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    output_path = os.path.join(raw_dir, "tinystories_expanded.txt")

    print(f"[1/3] Downloading {target_size_bytes // 1024} KB slice of TinyStories-valid.txt...")
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    headers = {"Range": f"bytes=0-{target_size_bytes - 1}"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=15) as response:
        raw_bytes = response.read()

    print(f"[2/3] Decoding and normalizing text ({len(raw_bytes)} bytes downloaded)...")
    text = raw_bytes.decode("utf-8", errors="ignore")
    # Clean redundant whitespace/newlines while preserving story breaks
    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    char_count = len(clean_text)
    word_count = len(clean_text.split())
    # Estimate chunks with length 256 and stride 64
    estimated_chunks = max(0, (char_count - 256) // 64 + 1)

    print(f"[3/3] Successfully saved to: {output_path}")
    print(f"  • Total characters: {char_count:,}")
    print(f"  • Total words:      {word_count:,}")
    print(f"  • Training chunks:  {estimated_chunks:,} (L=256, stride=64)")
    return output_path

if __name__ == "__main__":
    prepare_data()
