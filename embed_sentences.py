"""Embed German sentences with both backends and save results."""

import os
import argparse
import numpy as np
from pathlib import Path
from safetensors.numpy import save_file

from embedding_model import EmbeddingModel

SENTENCES = [
    "Die Künstliche Intelligenz verändert unsere Arbeitswelt grundlegend.",
    "Im Schwarzwald gibt es viele schöne Wanderwege entlang der Täler.",
    "Der Kunde hat sich über die verspätete Lieferung beschwert.",
    "Maschinelles Lernen ermöglicht die automatische Erkennung von Mustern in Daten.",
    "Das neue Gesetz zur Datenschutzgrundverordnung tritt nächsten Monat in Kraft.",
    "Die Entwicklung von Elektrofahrzeugen schreitet in Deutschland schnell voran.",
    "Unser Team arbeitet an einer skalierbaren Microservice-Architektur.",
    "Die Inflationsrate ist im vergangenen Quartal unerwartet gestiegen.",
    "Transformer-Modelle haben die natürliche Sprachverarbeitung revolutioniert.",
    "Der Patient wurde erfolgreich mit einer neuartigen Therapie behandelt.",
]

EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"


def run(version_tag: str):
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    for backend in ("sentence_transformers", "transformers"):
        print(f"Embedding with backend={backend} ...")
        model = EmbeddingModel(backend=backend)
        embeddings = model.embed(SENTENCES)

        filename = f"{version_tag}_{backend}.safetensors"
        path = EMBEDDINGS_DIR / filename
        save_file({"embeddings": embeddings}, str(path))
        print(f"  Saved {path}  shape={embeddings.shape}")

    # Also save the sentences for reference
    sentences_path = EMBEDDINGS_DIR / "sentences.txt"
    if not sentences_path.exists():
        sentences_path.write_text("\n".join(SENTENCES), encoding="utf-8")
        print(f"  Saved {sentences_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version-tag",
        default="v5x",
        help="Tag for this run, e.g. 'v5x' or 'v4x' (used in filenames)",
    )
    args = parser.parse_args()
    run(args.version_tag)
