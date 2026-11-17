# 로그인 — 해시 검증, 잠금, 계정 파일 검사, 운영 환경 데모 계정 거부
import pytest

from src.security import auth
from src.security.auth import AccountsError, LoginGuard, authenticate, hash_password, parse_accounts, verify_password

FAST = 1_000   # 테스트용 반복 수 (실제 60만 회)


def accounts(demo=False):
    return parse_accounts({"demo": demo, "users": {
        "kim": {"name": "김", "role": "researcher", "teams": ["dft"], "password": hash_password("pw-kim-123", iterations=FAST)},
    }}, "test")


def test_hash_roundtrip_and_salt():
    h1, h2 = hash_password("secret", iterations=FAST), hash_password("secret", iterations=FAST)
    assert h1 != h2                                      # 솔트가 달라 같은 비밀번호도 해시가 다르다
    assert verify_password("secret", h1) and not verify_password("Secret", h1)
    assert not verify_password("secret", "plain-text") and not verify_password("x", "md5$1$a$b")


def test_login_ok_fail_and_unknown_user():
    acc, guard = accounts(), LoginGuard()
    assert authenticate("KIM ", "pw-kim-123", acc, guard).status == "ok"    # 아이디 대소문자·공백 무시
    assert authenticate("kim", "wrong", acc, guard).status == "fail"
    assert authenticate("nobody", "pw-kim-123", acc, guard).status == "fail"


def test_lock_after_repeated_failures_then_unlock():
    now = [0.0]
    guard = LoginGuard(max_failures=3, lock_seconds=60, clock=lambda: now[0])
    acc = accounts()
    assert [authenticate("kim", "x", acc, guard).status for _ in range(3)] == ["fail", "fail", "locked"]
    assert authenticate("kim", "pw-kim-123", acc, guard).status == "locked"   # 잠긴 동안은 맞는 비밀번호도 거부
    now[0] = 61
    assert authenticate("kim", "pw-kim-123", acc, guard).status == "ok"


def test_accounts_file_rejects_plaintext_password_and_unknown_role():
    with pytest.raises(AccountsError, match="해시"):
        parse_accounts({"users": {"a": {"role": "researcher", "password": "1234"}}}, "t")
    with pytest.raises(AccountsError, match="role"):
        parse_accounts({"users": {"a": {"role": "root", "password": "pbkdf2_sha256$1$a$b"}}}, "t")


def test_production_refuses_demo_accounts(monkeypatch):
    monkeypatch.setattr(auth, "APP_ENV", "production")
    assert auth.startup_problem(accounts(demo=True)) is not None
    assert auth.startup_problem(accounts(demo=False)) is None
    monkeypatch.setattr(auth, "APP_ENV", "local")
    assert auth.startup_problem(accounts(demo=True)) is None


def test_repo_demo_accounts_file_is_valid():
    acc = auth.load_accounts()
    assert acc.demo and {"kim", "lee", "park", "choi", "guest"} <= set(acc.users)
