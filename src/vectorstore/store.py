# src/vectorstore/store.py
# 컬렉션 생성·조회 (Vector Store 영역). 저장 위치와 임베딩 모델은 config에서 읽는다.
import ctypes
import os
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.config import CHROMA_DIR, EMBEDDING_MODEL, EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MAX_TOKENS
from src.env import openai_kwargs


def chroma_path() -> str:
    """Chroma를 열 경로. Chroma(Rust)는 Windows에서 한글 등 비ASCII 문자가 든 경로의 HNSW 인덱스 파일을 읽지 못해
    컬렉션이 커져 인덱스가 디스크에 쓰인 뒤 "Error loading hnsw index"가 난다. 그래서 ASCII 경로로 연다:
    절대 경로 → 현재 폴더 기준 상대 경로 → Windows 짧은 경로(8.3) 순으로 시도한다."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [str(CHROMA_DIR)]
    try:
        candidates.append(os.path.relpath(CHROMA_DIR))
    except ValueError:   # 다른 드라이브
        pass
    if os.name == "nt":
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(str(CHROMA_DIR), buf, 1024):
            candidates.append(buf.value)
    for path in candidates:
        if path.isascii():
            return path
    raise RuntimeError(f"Chroma 저장 경로에 한글 등 영문이 아닌 문자가 있어 색인을 열 수 없습니다: {CHROMA_DIR}\n"
                       "프로젝트 폴더에서 실행하거나, .env에 CHROMA_DIR=영문 경로 를 지정하세요.")


@lru_cache(maxsize=1)
def _local_model(name: str):
    """sentence-transformers 모델은 로드에 수 초, 메모리 수 GB가 들어 프로세스에서 한 번만 만든다.
    처음 쓸 때 Hugging Face에서 모델을 내려받는다 (bge-m3 약 2.2GB)."""
    from sentence_transformers import SentenceTransformer   # 무거운 의존성이라 로컬 임베딩을 쓸 때만 올린다
    model = SentenceTransformer(name, device="cpu")
    model.max_seq_length = LOCAL_EMBEDDING_MAX_TOKENS
    return model


class LocalEmbeddings(Embeddings):
    """CPU에서 돌리는 로컬 임베딩 — API 호출·비용·한도가 없다. 코사인 거리에 맞춰 단위 벡터로 정규화한다."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = 16):
        self.model_name, self.batch_size = model_name, batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = _local_model(self.model_name).encode(texts, batch_size=self.batch_size,
                                                        normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> Embeddings:
    if EMBEDDING_PROVIDER == "local":
        return LocalEmbeddings(EMBEDDING_MODEL)
    # 색인 텍스트·청크는 토큰 한도(8,191)보다 훨씬 짧으므로 길이 검사(tiktoken 토큰화)를 끄고
    # 문자열 그대로 보낸다. 토큰 배열 입력을 받지 않는 OpenAI 호환 프록시(GMS)에서도 동작한다.
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, check_embedding_ctx_length=False, **openai_kwargs())


def get_store(collection: str, embeddings: Embeddings | None = None) -> Chroma:
    """컬렉션을 열고, 없으면 만든다."""
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=chroma_path(),
        collection_metadata={"hnsw:space": "cosine"},   # 생성 시에만 적용된다
    )


def count_chunks(collection: str) -> int:
    """임베딩·API 키 없이 저장된 청크 수만 센다. 컬렉션이 없으면 0 (색인 없음 안내용)."""
    if not CHROMA_DIR.exists():
        return 0
    client = chromadb.PersistentClient(path=chroma_path())
    if collection not in {c.name for c in client.list_collections()}:
        return 0
    return client.get_collection(collection).count()
