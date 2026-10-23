# src/data_prep/kipris.py — KIPRIS Plus 호출과 특허 자체 문서(요약·청구항) 수집
import re
import xml.etree.ElementTree as ET

from src.config import DETAIL_CACHE_DIR, KIPRIS_CALL_INTERVAL
from src.data_prep import http
from src.data_prep.docs import SourceDoc
from src.env import kipris_key

BASE_URL = "http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice"

# 검색 응답: 표준 필드명 → 원본 응답에서 올 수 있는 태그 후보 (응답 형식이 두 가지라 둘 다 대응)
KEY_MAP = {
    "ApplicationNumber": ["applicationNumber", "ApplicationNumber"],
    "InventionName": ["inventionTitle", "InventionName"],
    "Applicant": ["applicantName", "Applicant"],
    "ApplicationDate": ["applicationDate", "ApplicationDate"],
    "IPC": ["ipcNumber", "InternationalpatentclassificationNumber"],
    "RegistrationStatus": ["registerStatus", "RegistrationStatus"],
    "RegistrationNumber": ["registerNumber", "RegistrationNumber"],
    "RegistrationDate": ["registerDate", "RegistrationDate"],
    "OpeningNumber": ["openNumber", "OpeningNumber"],
    "Abstract": ["astrtCont", "Abstract"],
}


def call(op: str, params: dict, detail: str) -> bytes:
    """KIPRIS 오퍼레이션을 호출하고 resultCode가 00인 응답 원문을 돌려준다."""
    key = kipris_key()
    res = http.get(f"{BASE_URL}/{op}", params={**params, "ServiceKey": key},
                   tag=f"kipris:{op}", detail=detail, interval=KIPRIS_CALL_INTERVAL, secret=key)
    root = ET.fromstring(res.content)
    code = root.findtext("header/resultCode")
    if code != "00":
        raise RuntimeError(f"KIPRIS 오류 {code}: {root.findtext('header/resultMsg')}")
    return res.content


def find_total(root: ET.Element) -> int:
    for tag in ("totalCount", "TotalSearchCount"):
        el = next(root.iter(tag), None)
        if el is not None and (el.text or "").strip().isdigit():
            return int(el.text.strip())
    return 0


def item_nodes(root: ET.Element) -> list[ET.Element]:
    return list(root.iter("item")) or list(root.iter("PatentUtilityInfo"))


def parse_search(xml_bytes: bytes) -> tuple[int, list[dict]]:
    """검색 응답 → (전체 건수, 표준 필드로 바꾼 행 목록)."""
    root = ET.fromstring(xml_bytes)
    rows = []
    for node in item_nodes(root):
        raw = {child.tag: (child.text or "").strip() for child in node}
        row = {std: next((raw[t] for t in tags if raw.get(t)), "") for std, tags in KEY_MAP.items()}
        row["ipc_list"] = [x.strip() for x in row["IPC"].split("|") if x.strip()]
        rows.append(row)
    return find_total(root), rows


def _clean(text: str | None) -> str:
    """일부 공보는 청구항에 HTML 태그(<P>, <SDOCLAIM> 등)가 섞여 온다 — 태그를 지우고 공백을 정리한다."""
    text = re.sub(r"<[^>]{1,40}>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def patent_document(app_no: str) -> SourceDoc:
    """선택한 특허 자체를 다룬 문서. 서지 상세를 한 번만 받아 캐시한다 (NF2002)."""
    cache = DETAIL_CACHE_DIR / f"{app_no}.xml"
    if cache.exists():
        xml = cache.read_bytes()
    else:
        xml = call("getBibliographyDetailInfoSearch", {"applicationNumber": app_no}, detail=f"applno={app_no}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(xml)

    root = ET.fromstring(xml)
    title = _clean(root.findtext(".//biblioSummaryInfo/inventionTitle"))
    abstract = _clean(root.findtext(".//abstractInfo/astrtCont"))
    claims = [_clean(c.findtext("claim")) for c in root.iter("claimInfo")]
    claims = [c for c in claims if c and not re.fullmatch(r"\d+\.\s*삭제", c)]   # "3. 삭제" 같은 빈 청구항 제외

    sections = [f"[발명의 명칭] {title}"]
    if abstract:
        sections.append(f"[요약]\n{abstract}")
    if claims:
        sections.append("[청구항]\n" + "\n\n".join(claims))
    return SourceDoc(
        doc_id=f"kipris-{app_no}",
        title=f"{title} (특허공보 {app_no})",
        url=f"https://doi.org/10.8080/{app_no}",   # KIPRIS 공보 페이지로 연결되는 DOI
        source_type="patent_claims",
        text="\n\n".join(sections),
    )
