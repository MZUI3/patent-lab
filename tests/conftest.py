# 테스트 공용 도구: 실제 Chroma(임시 폴더) + 결정적인 가짜 임베딩
# 권한 필터는 벡터 DB의 where 문법에 기대므로, 가짜 저장소가 아니라 실제 Chroma로 확인한다.
import hashlib
import math

import chromadb
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class CharEmbeddings(Embeddings):
    """글자 2-gram을 해시해 세는 64차원 벡터. 같은 단어가 많이 겹치면 가깝다 — 네트워크·모델 없이 돈다."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 64
        t = "".join(text.split())
        for i in range(len(t) - 1):
            v[int(hashlib.md5(t[i:i + 2].encode()).hexdigest(), 16) % 64] += 1
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def chroma_store(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))

    def make(name: str = "internal_docs") -> Chroma:
        return Chroma(collection_name=name, embedding_function=CharEmbeddings(), client=client,
                      collection_metadata={"hnsw:space": "cosine"})
    return make
