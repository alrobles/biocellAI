"""Models: shared cell encoder + heads for the two M1 arms.

Design (ADR-002): identical capacity in both arms — only the training
objective differs (supervised vs text-contrastive).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CellEncoder(nn.Module):
    """MLP encoder: expression vector (HVGs) -> embedding."""

    def __init__(self, n_input: int, hidden: tuple[int, ...] = (256, 128), embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_input
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, embed_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextEncoder:
    """Frozen text encoder with embedding cache.

    Loads a SentenceTransformer when the repo ships an ST config; otherwise
    falls back to a plain HF encoder + mean pooling (e.g. SapBERT, PubMedBERT).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self._cache: dict[str, np.ndarray] = {}
        self._st = None
        self._tok = None
        self._hf = None
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(model_name, device=device)
            for p in self._st.parameters():
                p.requires_grad_(False)
            self._st.eval()
            get_dim = getattr(self._st, "get_embedding_dimension", None) or getattr(
                self._st, "get_sentence_embedding_dimension"
            )
            self.dim = int(get_dim())
        except Exception:
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(model_name)
            self._hf = AutoModel.from_pretrained(model_name).to(device)
            for p in self._hf.parameters():
                p.requires_grad_(False)
            self._hf.eval()
            self.dim = int(self._hf.config.hidden_size)

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        if self._st is not None:
            return self._st.encode(
                texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
            )
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self._hf(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            emb = F.normalize(emb, dim=-1)
            outs.append(emb.cpu().numpy())
        return np.concatenate(outs, axis=0)

    def embed(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            embs = self._encode(missing, batch_size)
            for t, e in zip(missing, embs):
                self._cache[t] = e
        return torch.from_numpy(np.stack([self._cache[t] for t in texts])).float()


def info_nce(cell_emb: torch.Tensor, text_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric CLIP loss. cell_emb: [B, D] learnable; text_emb: [B, T] frozen."""
    c = F.normalize(cell_emb, dim=-1)
    t = F.normalize(text_emb, dim=-1)
    logits = c @ t.T / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
