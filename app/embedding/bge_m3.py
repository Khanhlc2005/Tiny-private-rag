from typing import TypedDict

from app.core.config import settings


class EmbeddedText(TypedDict):
    dense: list[float] # 1024 chiều
    sparse_indices: list[int] # id của token
    sparse_values: list[float] # trọng số của token


class BGEM3Embedder:
    def __init__(self):
        self.model_name = settings.embedding_model
        self.device = settings.embedding_device
        self.batch_size = settings.embedding_batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self.model_name,
                devices =[self.device],
                use_fp16=(self.device != "cpu"),
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[EmbeddedText]:
        if not texts:
            return []

        output = self._load().encode(
            texts,
            batch_size=self.batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        results: list[EmbeddedText] = []
        for dense_vec, weights in zip(output["dense_vecs"], output["lexical_weights"]):
            results.append(
                {
                    "dense": dense_vec.tolist(),
                    "sparse_indices": [int(token_id) for token_id in weights],
                    "sparse_values": [float(weight) for weight in weights.values()],
                }
            )
        return results

    def embed_query(self, text: str) -> EmbeddedText:
        return self.embed_documents([text])[0]