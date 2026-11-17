# src/data_prep/internal.py — 사내 문서 카탈로그 읽기·검사, 원본 파일 → 색인할 텍스트
#
# 원본(Markdown·CSV)과 메타데이터(등급·열람 주체·버전)를 나눠, 메타데이터는 카탈로그 한 곳에서 관리한다.
# 검사에 걸리면 그 문서는 색인 대상에서 빠진다 (권한 설정이 틀린 문서가 넓게 열리는 것보다 안 보이는 쪽이 낫다):
#   필수 항목 누락 · 등급·주체 표기 오류(acl.validate) · 파일 없음 · 지원하지 않는 형식 · 같은 ID 중복
# 폴더에 있지만 카탈로그에 없는 파일은 '미등록'으로 보고만 하고 색인하지 않는다.
import csv
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.config import INTERNAL_CATALOG, INTERNAL_DIR, TABLE_ROWS_PER_CHUNK
from src.security import acl

FORMATS = {".md": "internal_doc", ".csv": "internal_table"}   # 확장자 → 출처 유형
REQUIRED = ("id", "file", "title", "kind", "level", "allow", "owner", "version", "updated")


@dataclass(frozen=True)
class InternalDoc:
    id: str
    title: str
    kind: str            # 발명 신고서 · 실험 보고서 · 실험 데이터 · 회의록 …
    level: str           # 공개 | 내부 | 기밀
    allow: tuple[str, ...]
    owner: str
    version: str
    updated: str
    path: Path

    @property
    def source_type(self) -> str:
        return FORMATS[self.path.suffix.lower()]

    @property
    def content_sha(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    @property
    def fingerprint(self) -> str:
        """내용과 등급·열람 주체·버전·제목을 함께 해시한다 — 권한만 바뀌어도 다시 색인되게."""
        key = "\n".join([self.content_sha, self.level, ",".join(sorted(self.allow)), self.version, self.title])
        return hashlib.sha256(key.encode()).hexdigest()


@dataclass
class Catalog:
    docs: list[InternalDoc] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)   # (문서 ID 또는 파일, 이유)
    unregistered: list[str] = field(default_factory=list)          # 카탈로그에 없는 파일
    teams: dict[str, str] = field(default_factory=dict)            # 팀 ID → 팀 이름


def load_catalog(path: Path = INTERNAL_CATALOG, root: Path = INTERNAL_DIR) -> Catalog:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cat = Catalog(teams={str(k): str(v) for k, v in (data.get("teams") or {}).items()})
    seen, registered = set(), set()
    for i, raw in enumerate(data.get("documents") or [], 1):
        doc_id = str(raw.get("id") or f"#{i}")
        missing = [k for k in REQUIRED if raw.get(k) in (None, "", [])]
        if missing:
            cat.rejected.append((doc_id, f"필수 항목 없음: {', '.join(missing)}"))
            continue
        file = root / str(raw["file"])
        registered.add(file.resolve())
        allow = [str(p) for p in raw["allow"]] if isinstance(raw["allow"], list) else [str(raw["allow"])]
        errors = acl.validate(str(raw["level"]), allow)
        if doc_id in seen:
            errors.append("ID 중복")
        if not file.exists():
            errors.append(f"파일 없음: {raw['file']}")
        elif file.suffix.lower() not in FORMATS:
            errors.append(f"지원하지 않는 형식: {file.suffix}")
        if errors:
            cat.rejected.append((doc_id, " / ".join(errors)))
            continue
        seen.add(doc_id)
        cat.docs.append(InternalDoc(doc_id, str(raw["title"]), str(raw["kind"]), str(raw["level"]), tuple(allow),
                                    str(raw["owner"]), str(raw["version"]), str(raw["updated"]), file))
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.resolve() != path.resolve() and f.resolve() not in registered:
            cat.unregistered.append(str(f.relative_to(root)).replace("\\", "/"))
    return cat


def read_text(doc: InternalDoc) -> str:
    return doc.path.read_text(encoding="utf-8-sig")


def table_chunks(doc: InternalDoc, rows_per_chunk: int = TABLE_ROWS_PER_CHUNK) -> list[str]:
    """CSV를 행 묶음마다 Markdown 표로 바꾼다. 청크마다 머리글과 '몇 행 중 몇 행'을 붙여, 검색된 청크 하나만 봐도
    열 이름과 위치가 보이게 한다 (머리글 없는 숫자 행만 오면 LLM이 어느 열의 값인지 모른다)."""
    rows = list(csv.reader(io.StringIO(read_text(doc))))
    if not rows:
        return []
    header, body = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
    out = []
    for start in range(0, len(body), rows_per_chunk):
        part = body[start:start + rows_per_chunk]
        lines = [f"표: {doc.title} — 전체 {len(body)}행 중 {start + 1}~{start + len(part)}행",
                 "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(r) + " |" for r in part]
        out.append("\n".join(lines))
    return out
