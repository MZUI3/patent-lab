# src/data_prep/items.py
# 항목 메타데이터 적재 (F2001): 특허 1건을 하나의 색인 텍스트와 메타데이터로 만든다.
# 실행: python -m src.data_prep.items (프로젝트 루트에서) — 건수와 색인 텍스트 1건을 출력한다.
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.config import PATENTS_JSON, PATENTS_ONDEMAND_JSON

SOURCE_TYPE = "item_meta"   # 근거 출처 유형 구분용 (F2007)


@dataclass
class Item:
    id: str          # 출원번호 — 색인 청크 식별자로 고정해 재색인 시 덮어쓰기만 일어나게 한다 (F2002)
    text: str        # 임베딩할 색인 텍스트
    metadata: dict   # Chroma에 함께 저장할 값 (str/int만)


def clean(value: str) -> str:
    """연속 공백·줄바꿈을 한 칸으로 줄인다."""
    return re.sub(r"\s+", " ", value or "").strip()


def split_multi(value: str) -> list[str]:
    """KIPRIS는 여러 값을 '|'로 이어 준다 (IPC, 출원인)."""
    return [clean(v) for v in (value or "").split("|") if clean(v)]


def format_date(value: str) -> str:
    """'20190305' → '2019-03-05'. 형식이 다르면 원문을 그대로 둔다."""
    digits = re.sub(r"\D", "", value or "")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}" if len(digits) == 8 else clean(value)


def build_item_text(p: dict) -> str:
    """사용자가 특허를 찾을 단서(명칭·출원인·연도·분류·상태·기술 내용)를 한 텍스트로 묶는다.

    요약을 넣었기 때문에 "스캔 체인 고장 진단 방법" 같은 기술 조건 검색이 걸린다.
    빈 속성은 줄째로 뺀다.
    """
    ipc = p.get("ipc_list") or split_multi(p.get("IPC", ""))
    status = clean(p.get("RegistrationStatus", ""))
    reg_no = clean(p.get("RegistrationNumber", ""))
    if status and reg_no:
        status = f"{status} (등록번호 {reg_no}, 등록일 {format_date(p.get('RegistrationDate', ''))})"

    lines = [
        ("발명의 명칭", clean(p.get("InventionName", ""))),
        ("출원인", ", ".join(split_multi(p.get("Applicant", "")))),
        ("출원번호", clean(p.get("ApplicationNumber", ""))),
        ("출원일", format_date(p.get("ApplicationDate", ""))),
        ("IPC 분류", ", ".join(ipc)),
        ("등록 상태", status),
        ("요약", clean(p.get("Abstract", ""))),
    ]
    return "\n".join(f"{label}: {value}" for label, value in lines if value)


def build_item_metadata(p: dict) -> dict:
    """필터·근거 표시에 쓸 값. Chroma 메타데이터는 None·리스트를 받지 않으므로 문자열로 펼친다."""
    ipc = p.get("ipc_list") or split_multi(p.get("IPC", ""))
    app_date = format_date(p.get("ApplicationDate", ""))
    return {
        "item_id": clean(p["ApplicationNumber"]),
        "title": clean(p.get("InventionName", "")),
        "applicant": ", ".join(split_multi(p.get("Applicant", ""))),
        "application_date": app_date,
        "application_year": int(app_date[:4]) if app_date[:4].isdigit() else 0,
        "ipc_main": ipc[0] if ipc else "",
        "ipc_all": " | ".join(ipc),
        "status": clean(p.get("RegistrationStatus", "")),
        "registration_number": clean(p.get("RegistrationNumber", "")),
        "source_type": SOURCE_TYPE,
        "field": p.get("field", ""),   # 수집 분야 (config.KIPRIS_FIELDS의 이름, 검색으로 받았으면 "검색 수집")
        "source": "KIPRIS Plus 특허·실용신안 검색",
        "source_url": f"https://doi.org/10.8080/{clean(p['ApplicationNumber'])}",   # KIPRIS 공보 페이지
    }


def load_patents(path: Path = PATENTS_JSON, ondemand: Path | None = PATENTS_ONDEMAND_JSON) -> list[dict]:
    """분야별 일괄 수집분 + 검색할 때 추가로 받은 특허."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path}가 없습니다. 먼저 python -m src.data_prep.collect_meta 로 메타데이터를 수집하세요."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if ondemand is not None and ondemand.exists():
        rows += json.loads(ondemand.read_text(encoding="utf-8"))
    return rows


def to_items(patents: list[dict]) -> list[Item]:
    """출원번호나 명칭이 없는 항목은 검색·표시가 불가능하므로 제외한다. 출원번호 기준 중복도 제거한다."""
    items: dict[str, Item] = {}
    for p in patents:
        app_no = clean(p.get("ApplicationNumber", ""))
        if not app_no or not clean(p.get("InventionName", "")):
            continue
        items.setdefault(app_no, Item(id=app_no, text=build_item_text(p), metadata=build_item_metadata(p)))
    return list(items.values())


def load_items(path: Path = PATENTS_JSON, ondemand: Path | None = PATENTS_ONDEMAND_JSON) -> list[Item]:
    return to_items(load_patents(path, ondemand))


def main() -> None:
    raw = load_patents()
    items = load_items()
    print(f"원본 {len(raw)}건 → 항목 {len(items)}건 (빈 값·중복 제외 {len(raw) - len(items)}건)")
    if not items:
        return
    print("분야 분포:", dict(Counter(it.metadata["field"] for it in items)))

    no_abstract = sum(1 for it in items if "\n요약: " not in it.text)
    lengths = sorted(len(it.text) for it in items)
    print(f"요약 없는 항목: {no_abstract}건")
    print(f"색인 텍스트 길이: 최소 {lengths[0]} / 중앙 {lengths[len(lengths) // 2]} / 최대 {lengths[-1]}자")
    print("출원연도 분포:", sorted(Counter(it.metadata["application_year"] for it in items).items()))

    sample = items[0]
    print(f"\n[색인 텍스트 예시] id={sample.id}\n{sample.text}")
    print(f"\n[메타데이터 예시]\n{json.dumps(sample.metadata, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
