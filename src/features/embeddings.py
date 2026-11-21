"""Dense representation: Sentence-BERT embeddings (+ caching and 2-D visualisation).

all-MiniLM-L6-v2 maps a whole sentence to a 384-d vector such that sentences with similar
MEANING are close (cosine similarity), even with no shared words - e.g. "internet dead in
room" and "wifi not working in hostel".  Unlike TF-IDF it captures synonyms and context,
but individual dimensions are not interpretable.

Embeddings are computed on LIGHTLY cleaned text (text_light) and cached to
artifacts/embeddings/*.npy, keyed by a hash of the texts so a changed dataset is detected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.utils import ensure_dir, get_logger

log = get_logger("embeddings")


def device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class SbertEncoder:
    """Lazy wrapper around SentenceTransformer."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from src.utils import MODEL_LOAD_LOCK

            # Lock: transformers' from_pretrained temporarily makes newly created layers "meta"
            # (weightless) process-wide; building another model on a parallel thread (e.g. a
            # Streamlit rerun) at that moment yields "Cannot copy out of meta tensor".
            with MODEL_LOAD_LOCK:
                if self._model is None:
                    self._model = self._load()
        return self._model

    def _load(self):
        from sentence_transformers import SentenceTransformer

        dev = device()
        try:
            # load on CPU (weights fully materialised), then move to the GPU
            model = SentenceTransformer(self.model_name, device="cpu", model_kwargs={"low_cpu_mem_usage": False})
        except NotImplementedError as e:  # meta-tensor error despite the lock -> retry once
            log.warning("SBERT load hit %s - retrying", e)
            model = SentenceTransformer(self.model_name, device="cpu", model_kwargs={"low_cpu_mem_usage": False})
        return model.to(dev)

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        # normalize_embeddings=True -> unit vectors, so dot product == cosine similarity
        return self.model.encode(list(texts), batch_size=self.batch_size, show_progress_bar=show_progress,
                                 normalize_embeddings=True, convert_to_numpy=True)


def _hash(texts: list[str], model_name: str) -> str:
    h = hashlib.md5(model_name.encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
    return h.hexdigest()


def cached_embeddings(texts: list[str], name: str, encoder: SbertEncoder, cache_dir: str | Path) -> np.ndarray:
    """Return embeddings from cache if the texts/model are unchanged, else compute and save."""
    cache_dir = ensure_dir(cache_dir)
    slug = encoder.model_name.split("/")[-1]
    npy, meta = cache_dir / f"{name}_{slug}.npy", cache_dir / f"{name}_{slug}.json"
    key = _hash(texts, encoder.model_name)
    if npy.exists() and meta.exists() and json.loads(meta.read_text())["hash"] == key:
        log.info("Loaded cached embeddings %s", npy.name)
        return np.load(npy)
    log.info("Encoding %d texts for '%s' on %s ...", len(texts), name, device())
    X = encoder.encode(texts, show_progress=True)
    np.save(npy, X)
    meta.write_text(json.dumps({"hash": key, "model": encoder.model_name, "shape": list(X.shape)}))
    return X


def plot_embeddings_2d(X: np.ndarray, y: list[str], out_path: str | Path, title: str = "SBERT embeddings", seed: int = 42) -> str:
    """Project to 2-D with UMAP (falls back to t-SNE) and save a scatter plot coloured by class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import umap

        coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(X)
        method = "UMAP"
    except Exception as e:  # umap missing or numba problem
        from sklearn.manifold import TSNE

        log.warning("UMAP unavailable (%s) - using t-SNE", e)
        coords = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30).fit_transform(X)
        method = "t-SNE"
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for lab in sorted(set(y)):
        m = y == lab
        ax.scatter(coords[m, 0], coords[m, 1], s=7, alpha=0.65, label=lab)
    ax.set_title(f"{title} ({method} projection)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return method
