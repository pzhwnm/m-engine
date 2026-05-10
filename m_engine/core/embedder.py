"""
Embedder: 文本嵌入模块。
封装 sentence-transformers 和 OpenAI Embeddings 两种后端，
为 FactBus 检索提供统一的向量化接口。
"""

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """文本嵌入器。

    尝试顺序：
    1. 如果指定了 model_name 且 sentence-transformers 可用，使用本地模型
    2. 如果设置了 OPENAI_API_KEY，使用 OpenAI text-embedding-3-small
    3. 回退到简单的词袋哈希嵌入（仅保证维度一致，语义差）
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dim: int = 256,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.dim = dim
        self._model = None
        self._backend = None  # "local", "openai", "hash"

    def _ensure_model(self):
        if self._backend is not None:
            return
        # 尝试本地 sentence-transformers
        if self.model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._backend = "local"
                self.dim = self._model.get_sentence_embedding_dimension()
                logger.info("Using local embedding model: %s (dim=%d)",
                            self.model_name, self.dim)
                return
            except ImportError:
                logger.warning("sentence-transformers not installed, trying OpenAI")
            except Exception as e:
                logger.warning("Local model load failed: %s", e)

        # 尝试 OpenAI 兼容 embeddings（含 Ollama）
        if self.api_key or self.base_url:
            self._backend = "openai"
            logger.info("Using OpenAI-compatible embeddings (dim=%d)", self.dim)
            return

        # 回退到哈希嵌入
        self._backend = "hash"
        logger.warning("No embedding backend available, using hash-based fallback")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """将文本列表编码为向量列表。"""
        self._ensure_model()

        if self._backend == "local":
            return self._model.encode(texts, normalize_embeddings=True).tolist()

        elif self._backend == "openai":
            return self._encode_openai(texts)

        else:
            return [self._hash_embed(t) for t in texts]

    def encode_single(self, text: str) -> List[float]:
        return self.encode([text])[0]

    def _encode_openai(self, texts: List[str]) -> List[List[float]]:
        # Ollama native embedding API
        if self.base_url and "11434" in str(self.base_url):
            return self._encode_ollama(texts)

        from openai import OpenAI
        client = OpenAI(api_key=self.api_key or "ollama", base_url=self.base_url)
        emb_model = self.model_name or "text-embedding-3-small"
        try:
            response = client.embeddings.create(
                model=emb_model,
                input=texts,
            )
            embeddings = [d.embedding for d in response.data]
            # 规范化
            result = []
            for emb in embeddings:
                arr = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                result.append(arr.tolist())
            return result
        except Exception as e:
            logger.error("OpenAI embedding failed: %s, falling back to hash", e)
            return [self._hash_embed(t) for t in texts]

    def _encode_ollama(self, texts: List[str]) -> List[List[float]]:
        """Ollama 原生 /api/embeddings 端点。"""
        import json as _json
        import urllib.request

        emb_model = self.model_name or "nomic-embed-text"
        url = str(self.base_url).replace("/v1", "") + "/api/embeddings"
        if not url.startswith("http"):
            url = "http://localhost:11434/api/embeddings"

        result = []
        for text in texts:
            body = _json.dumps({"model": emb_model, "prompt": text}).encode("utf-8")
            req = urllib.request.Request(url, data=body,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = _json.loads(resp.read().decode("utf-8"))
                    emb = data.get("embedding", [])
                    arr = np.array(emb, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    result.append(arr.tolist())
            except Exception as e:
                logger.error("Ollama embedding failed: %s, using hash", e)
                result.append(self._hash_embed(text))
        return result

    def _hash_embed(self, text: str) -> List[float]:
        """基于字符 n-gram 哈希的简单嵌入。一致性优于语义准确度。"""
        vec = np.zeros(self.dim, dtype=np.float32)
        # 2-gram 和 3-gram 哈希
        for n in (2, 3):
            for i in range(len(text) - n + 1):
                h = hash(text[i:i + n]) % self.dim
                vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
