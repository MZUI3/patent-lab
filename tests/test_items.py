from src.data_prep.items import build_item_metadata, build_item_text, format_date, to_items

RAW = {
    "ApplicationNumber": " 1020230136220 ",
    "InventionName": "지터 측정 회로 및 지터 측정 방법",
    "Applicant": "삼성전자주식회사|한양대학교 에리카산학협력단",
    "ApplicationDate": "20231012",
    "IPC": "G01R 31/3183|G01R 29/26",
    "RegistrationStatus": "공개",
    "RegistrationNumber": "",
    "Abstract": "지터를  측정하는\n회로이다.",
    "field": "반도체 테스트",
}


def test_format_date():
    assert format_date("20231012") == "2023-10-12"
    assert format_date("2023") == "2023"


def test_item_text_has_every_field_and_normalized_values():
    text = build_item_text(RAW)
    for label in ("발명의 명칭", "출원인", "출원번호", "출원일", "IPC 분류", "등록 상태", "요약"):
        assert f"{label}: " in text
    assert "출원인: 삼성전자주식회사, 한양대학교 에리카산학협력단" in text
    assert "요약: 지터를 측정하는 회로이다." in text


def test_empty_fields_are_dropped_and_registration_expanded():
    text = build_item_text({**RAW, "Abstract": "", "RegistrationStatus": "등록",
                            "RegistrationNumber": "1012041380000", "RegistrationDate": "20121116"})
    assert "요약:" not in text
    assert "등록 상태: 등록 (등록번호 1012041380000, 등록일 2012-11-16)" in text


def test_metadata_types_for_chroma():
    meta = build_item_metadata(RAW)
    assert meta["item_id"] == "1020230136220"
    assert meta["application_year"] == 2023 and isinstance(meta["application_year"], int)
    assert meta["ipc_main"] == "G01R 31/3183" and meta["ipc_all"] == "G01R 31/3183 | G01R 29/26"
    assert meta["source_url"].endswith("/1020230136220")
    assert all(isinstance(v, (str, int)) for v in meta.values())   # Chroma는 None·리스트를 받지 않는다


def test_to_items_drops_invalid_and_deduplicates():
    items = to_items([RAW, RAW, {**RAW, "ApplicationNumber": "2"}, {**RAW, "InventionName": ""},
                      {**RAW, "ApplicationNumber": ""}])
    assert [it.id for it in items] == ["1020230136220", "2"]
