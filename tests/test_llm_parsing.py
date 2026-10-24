from src.rag.llm import parse_json_array, parse_json_object


def test_array_inside_code_block_and_prose():
    text = '검색어는 다음과 같습니다.\n```json\n["지터", "위상 고정 루프"]\n```\n이상입니다.'
    assert parse_json_array(text) == ["지터", "위상 고정 루프"]


def test_array_of_objects_is_read_whole():
    text = '[{"n": 3, "reason": "가장 가깝다"}, {"n": 1, "reason": "비슷"}]'
    assert parse_json_array(text) == [{"n": 3, "reason": "가장 가깝다"}, {"n": 1, "reason": "비슷"}]


def test_invalid_array_returns_empty():
    assert parse_json_array("배열이 없습니다") == []
    assert parse_json_array("[1, 2") == []


def test_object_with_trailing_text():
    text = '{"elements": [], "summary": "요약"} 끝.'
    assert parse_json_object(text) == {"elements": [], "summary": "요약"}


def test_invalid_object_returns_empty_dict():
    assert parse_json_object("없음") == {}
    assert parse_json_object("[1, 2]") == {}
