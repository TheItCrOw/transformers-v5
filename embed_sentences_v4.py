"""Embed sentences using v4x libraries (no v5x workarounds needed)."""

import numpy as np
import torch
from pathlib import Path
from safetensors.numpy import save_file
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

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

MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v2.0"
EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"


def run():
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # sentence-transformers backend
    print("Embedding with backend=sentence_transformers ...")
    st_model = SentenceTransformer(
        MODEL_NAME, trust_remote_code=True,
        config_kwargs={"use_memory_efficient_attention": False},
    )
    embeddings = st_model.encode(SENTENCES, normalize_embeddings=True)
    path = EMBEDDINGS_DIR / "v4x_sentence_transformers.safetensors"
    save_file({"embeddings": embeddings}, str(path))
    print(f"  Saved {path}  shape={embeddings.shape}")

    # raw transformers backend
    print("Embedding with backend=transformers ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.use_memory_efficient_attention = False
    model = AutoModel.from_pretrained(
        MODEL_NAME, config=config, trust_remote_code=True
    )
    model.eval()
    tokens = tokenizer(SENTENCES, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    embeddings = output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).numpy()
    path = EMBEDDINGS_DIR / "v4x_transformers.safetensors"
    save_file({"embeddings": embeddings}, str(path))
    print(f"  Saved {path}  shape={embeddings.shape}")


if __name__ == "__main__":
    run()
