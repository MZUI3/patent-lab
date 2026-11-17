# 문서별 열람 권한 — 주체 계산, 벡터 DB 필터, 설정 검사
import pytest

from src.security import acl
from src.security.acl import User

KIM = User("kim", "김", "researcher", ("dft",))
GUEST = User("guest", "방문자", "guest")
ADMIN = User("choi", "최", "admin")


def test_principals_by_role():
    assert KIM.principals == ["all", "employee", "team:dft", "role:researcher", "user:kim"]
    assert "employee" not in GUEST.principals          # 방문자는 임직원 문서를 못 본다
    assert ADMIN.principals == ["all", "employee", "role:admin", "user:choi"]   # 관리자도 팀 문서 권한은 따로


def test_where_filter_and_can_read_agree():
    meta = {"doc_id": "X", **acl.acl_metadata(["team:dft", "role:ip"])}
    assert acl.where_for(KIM) == {"$or": [{"acl_all": True}, {"acl_employee": True}, {"acl_team_dft": True},
                                          {"acl_role_researcher": True}, {"acl_user_kim": True}]}
    assert acl.can_read(KIM, meta)
    assert not acl.can_read(GUEST, meta)
    assert not acl.can_read(ADMIN, meta)
    assert acl.can_read(User("park", "박", "ip", ("ip",)), meta)


@pytest.mark.parametrize("level, allow, problem", [
    ("기밀", ["employee"], "임직원 전체"),
    ("내부", ["all"], "외부 방문자"),
    ("비밀", ["team:dft"], "알 수 없는 등급"),
    ("공개", [], "비어 있음"),
    ("내부", ["team:DFT!"], "표기 오류"),
])
def test_validate_rejects_risky_or_broken_settings(level, allow, problem):
    assert any(problem in e for e in acl.validate(level, allow))


def test_validate_accepts_normal_settings():
    assert acl.validate("기밀", ["team:dft", "role:ip", "user:kim"]) == []
    assert acl.validate("공개", ["all"]) == []


def test_max_level():
    assert acl.max_level(["공개", "기밀", "내부"]) == "기밀"
    assert acl.max_level([]) == "공개"
