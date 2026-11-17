# src/vectorstore/index_internal.py — 사내 문서 색인 동기화: 카탈로그와 색인을 맞춘다
#
#   새 문서 → 색인 / 지문(내용·등급·열람 주체·버전·제목)이 바뀐 문서 → 이전 청크를 지우고 다시 색인
#   / 카탈로그에서 빠졌거나 검사에 걸린 문서 → 청크 삭제
# 권한만 바꿔도 다시 색인하는 이유: 색인에 옛 권한이 남으면 권한을 거둔 사람에게도 계속 검색된다.
# 내용이 바뀌었는데 버전이 그대로면 경고한다 (같은 버전 번호로 다른 내용이 인용되지 않게).
# 청크마다 AI에게 내리는 지시처럼 보이는 구절을 점검해(security/guard.scan) 표시하고 보고서에 올린다.
# 결과: data/index/internal_manifest.json(문서별 현재 버전·지문·청크 수), 감사 로그의 ingest 이벤트
#
# 실행 (프로젝트 루트에서)
#   python -m src.vectorstore.index_internal           동기화 + 보고서
#   python -m src.vectorstore.index_internal --check   카탈로그 검사만 (색인하지 않음)
import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma

from src.config import INTERNAL_COLLECTION, INTERNAL_MANIFEST
from src.data_prep.internal import Catalog, InternalDoc, load_catalog, read_text, table_chunks
from src.security import acl, audit, guard
from src.vectorstore.index_docs import get_splitter
from src.vectorstore.store import get_store


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    unregistered: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    flagged: dict[str, list[str]] = field(default_factory=dict)   # 문서 ID → 의심 구절 종류
    chunks: int = 0                                                # 동기화 뒤 색인에 있는 청크 수

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


def doc_chunks(doc: InternalDoc) -> list[str]:
    if doc.source_type == "internal_table":
        return table_chunks(doc)
    return get_splitter().split_text(read_text(doc))


def indexed(store: Chroma) -> dict[str, dict]:
    """색인에 지금 들어 있는 문서별 상태. 카탈로그나 manifest가 아니라 색인 자체를 기준으로 본다
    (manifest만 믿으면, 색인을 지운 PC에서 '이미 색인됨'으로 건너뛴다)."""
    out: dict[str, dict] = {}
    for m in store.get(include=["metadatas"])["metadatas"]:
        d = out.setdefault(m["doc_id"], {"fp": m.get("doc_fp"), "content_sha": m.get("content_sha"),
                                         "version": m.get("version"), "level": m.get("level"),
                                         "allow": m.get("allow"), "chunks": 0, "flagged": []})
        d["chunks"] += 1
        for kind in filter(None, str(m.get("injection", "")).split(", ")):
            if kind not in d["flagged"]:
                d["flagged"].append(kind)
    return out


def index_doc(doc: InternalDoc, store: Chroma) -> tuple[int, list[str]]:
    """문서의 이전 청크를 지우고 다시 넣는다. (청크 수, 의심 구절 종류)"""
    store.delete(where={"doc_id": doc.id})
    chunks, kinds, metas = doc_chunks(doc), [], []
    fp, sha = doc.fingerprint, doc.content_sha
    for i, chunk in enumerate(chunks):
        found = guard.scan(chunk)
        kinds += [k for k in found if k not in kinds]
        metas.append({
            "doc_id": doc.id, "doc_title": doc.title, "kind": doc.kind, "level": doc.level,
            "allow": " | ".join(doc.allow), "owner": doc.owner, "version": doc.version, "updated": doc.updated,
            "source_type": doc.source_type, "source_url": "", "chunk_index": i,
            "doc_fp": fp, "content_sha": sha, "flagged": bool(found), "injection": ", ".join(found),
            **acl.acl_metadata(doc.allow),
        })
    if chunks:
        # 청크 앞에 문서 제목을 붙인다 (index_docs와 같은 이유: 본문만으로 주제가 드러나지 않는 청크)
        store.add_texts(texts=[f"[{doc.title}]\n{c}" for c in chunks],
                        ids=[f"{doc.id}:{i}" for i in range(len(chunks))], metadatas=metas)
    return len(chunks), kinds


def sync(store: Chroma | None = None, catalog: Catalog | None = None, manifest: Path | None = INTERNAL_MANIFEST,
         record: bool = True) -> SyncReport:
    store = store or get_store(INTERNAL_COLLECTION)
    catalog = catalog or load_catalog()
    rep = SyncReport(rejected=list(catalog.rejected), unregistered=list(catalog.unregistered))
    current = indexed(store)
    wanted = {d.id: d for d in catalog.docs}

    for doc_id in sorted(current.keys() - wanted.keys()):   # 카탈로그에서 빠졌거나 검사에 걸린 문서
        store.delete(where={"doc_id": doc_id})
        rep.removed.append(doc_id)
    for doc in catalog.docs:
        old = current.get(doc.id)
        if old and old["fp"] == doc.fingerprint:
            rep.unchanged.append(doc.id)
            if old["flagged"]:
                rep.flagged[doc.id] = old["flagged"]
            continue
        _, kinds = index_doc(doc, store)
        if kinds:
            rep.flagged[doc.id] = kinds
        if old is None:
            rep.added.append(f"{doc.id} v{doc.version}")
            continue
        what = []
        if old["version"] != doc.version:
            what.append(f"v{old['version']} → v{doc.version}")
        elif old["content_sha"] != doc.content_sha:
            rep.warnings.append(f"{doc.id}: 내용이 바뀌었는데 버전(v{doc.version})이 그대로입니다")
            what.append("내용 변경")
        if old["level"] != doc.level or old["allow"] != " | ".join(doc.allow):
            what.append(f"등급·권한 {old['level']}/{old['allow']} → {doc.level}/{' | '.join(doc.allow)}")
        rep.updated.append(f"{doc.id} ({', '.join(what) or '제목 변경'})")

    state = indexed(store)
    rep.chunks = sum(d["chunks"] for d in state.values())
    if manifest is not None and (rep.changed or not manifest.exists()):
        write_manifest(manifest, catalog, state)
    if record and (rep.changed or rep.rejected):
        audit.record("ingest", None, added=rep.added, updated=rep.updated, removed=rep.removed,
                     rejected=[d for d, _ in rep.rejected], flagged=sorted(rep.flagged), chunks=rep.chunks)
    return rep


def write_manifest(path: Path, catalog: Catalog, state: dict) -> None:
    docs = []
    for d in catalog.docs:
        s = state.get(d.id, {})
        docs.append({"id": d.id, "title": d.title, "kind": d.kind, "level": d.level, "allow": list(d.allow),
                     "version": d.version, "updated": d.updated, "fingerprint": d.fingerprint[:12],
                     "chunks": s.get("chunks", 0), "flagged": s.get("flagged", [])})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "synced_at": datetime.now(audit.KST).isoformat(timespec="seconds"),
        "documents": docs,
        "rejected": [{"id": i, "reason": r} for i, r in catalog.rejected],
        "unregistered": catalog.unregistered,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def print_report(rep: SyncReport) -> None:
    for label, items in (("새로 색인", rep.added), ("다시 색인", rep.updated), ("삭제", rep.removed)):
        if items:
            print(f"{label} {len(items)}건: " + ", ".join(items))
    print(f"변경 없음 {len(rep.unchanged)}건 · 색인 청크 {rep.chunks}개")
    for doc_id, reason in rep.rejected:
        print(f"[색인 안 함] {doc_id}: {reason}")
    for f in rep.unregistered:
        print(f"[미등록 파일 — 색인 안 함] {f}")
    for w in rep.warnings:
        print(f"[경고] {w}")
    for doc_id, kinds in rep.flagged.items():
        print(f"[검토 필요 — AI에게 내리는 지시로 보이는 구절] {doc_id}: {', '.join(kinds)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="사내 문서 색인 동기화")
    ap.add_argument("--check", action="store_true", help="카탈로그 검사만 하고 색인하지 않는다")
    args = ap.parse_args()
    if args.check:
        cat = load_catalog()
        print(f"색인 대상 {len(cat.docs)}건")
        for d in cat.docs:
            print(f"  {d.id:14} {d.level} v{d.version:5} {', '.join(d.allow):24} {d.title}")
        print_report(SyncReport(rejected=cat.rejected, unregistered=cat.unregistered))
        return
    print_report(sync())


if __name__ == "__main__":
    main()
