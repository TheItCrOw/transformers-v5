"""Load saved embeddings and compare them via cosine similarity."""

import sys
from datetime import datetime
from pathlib import Path
from itertools import combinations

import numpy as np
from safetensors.numpy import load_file

EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
RESULTS_DIR = Path(__file__).parent / "results"


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (n, dim) arrays."""
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b)


def load_all_embeddings() -> dict[str, np.ndarray]:
    """Load all .safetensors files from the embeddings directory."""
    results = {}
    for path in sorted(EMBEDDINGS_DIR.glob("*.safetensors")):
        data = load_file(str(path))
        results[path.stem] = data["embeddings"]
    return results


def compare():
    if not EMBEDDINGS_DIR.exists():
        print("No embeddings directory found. Run embed_sentences.py first.")
        sys.exit(1)

    embeddings = load_all_embeddings()
    if len(embeddings) < 2:
        print(f"Found {len(embeddings)} embedding file(s), need at least 2 to compare.")
        sys.exit(1)

    # Load sentences for display
    sentences_path = EMBEDDINGS_DIR / "sentences.txt"
    sentences = (
        sentences_path.read_text(encoding="utf-8").splitlines()
        if sentences_path.exists()
        else [f"sentence_{i}" for i in range(len(next(iter(embeddings.values()))))]
    )

    lines = []
    lines.append(f"Loaded embeddings: {list(embeddings.keys())}")
    lines.append("")

    for name_a, name_b in combinations(embeddings.keys(), 2):
        emb_a = embeddings[name_a]
        emb_b = embeddings[name_b]
        similarities = cosine_similarity_matrix(emb_a, emb_b)

        lines.append(f"=== {name_a} vs {name_b} ===")
        lines.append(f"  Mean cosine similarity: {similarities.mean():.8f}")
        lines.append(f"  Min cosine similarity:  {similarities.min():.8f}")
        lines.append(f"  Max cosine similarity:  {similarities.max():.8f}")
        lines.append("")

        # Per-sentence breakdown
        for i, (sim, sent) in enumerate(zip(similarities, sentences)):
            flag = "" if sim > 0.9999 else " <-- DRIFT"
            lines.append(f"  [{i}] {sim:.8f}  {sent[:70]}{flag}")
        lines.append("")

    output = "\n".join(lines)
    print(output)

    # Save to file
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"comparison_{timestamp}.txt"
    result_path.write_text(output, encoding="utf-8")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    compare()
