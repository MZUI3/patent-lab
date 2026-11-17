# src/security/auth.py — 로그인: 계정 파일의 비밀번호 해시 확인, 연속 실패 잠금
#
# 계정은 config/users.toml(로컬 데모용 가상 계정)에 있고, 배포할 때는 USERS_TOML(시크릿에 넣은 TOML 원문)이나
# USERS_FILE(경로)로 바꾼다. 비밀번호는 PBKDF2-SHA256(60만 회) 해시로만 저장하고 상수 시간으로 비교한다.
# 없는 아이디도 같은 계산을 거쳐 응답 시간으로 계정 존재 여부가 드러나지 않게 한다.
# 운영 환경(APP_ENV=production)에서는 데모 계정 파일(demo = true)로 시작하지 않는다.
#
# 해시 만들기: python -m src.security.auth hash        계정 확인: python -m src.security.auth list
import base64
import hashlib
import hmac
import os
import re
import secrets
import sys
import time
import tomllib
from dataclasses import dataclass
from getpass import getpass
from threading import Lock

from src import env  # noqa: F401  (.env를 환경변수로 올린다 — USERS_TOML을 .env에 둔 경우)
from src.config import APP_ENV, LOGIN_LOCK_MINUTES, LOGIN_MAX_FAILURES, USERS_FILE
from src.security.acl import ROLES, User

ITERATIONS = 600_000
_SLUG = re.compile(r"^[a-z0-9_]+$")
# 없는 아이디를 확인할 때 쓰는 해시 — 결과는 버리고 계산 시간만 같게 한다
_DUMMY = "pbkdf2_sha256$600000$AAAAAAAAAAAAAAAAAAAAAA==$" + "A" * 43 + "="


class AccountsError(RuntimeError):
    """계정 파일 형식 오류."""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def hash_password(password: str, *, iterations: int = ITERATIONS, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt, digest = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt), int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


@dataclass
class Accounts:
    users: dict[str, User]
    hashes: dict[str, str]
    demo: bool          # 데모 계정 파일인가 — 운영 환경에서는 거부
    source: str         # 어디서 읽었나 (화면·로그용, 내용은 담지 않는다)


def parse_accounts(data: dict, source: str) -> Accounts:
    users, hashes = {}, {}
    for username, u in (data.get("users") or {}).items():
        where = f"{source} [users.{username}]"
        if not _SLUG.match(username):
            raise AccountsError(f"{where}: 아이디는 영문 소문자·숫자·_만 쓴다")
        if u.get("role") not in ROLES:
            raise AccountsError(f"{where}: role은 {', '.join(ROLES)} 중 하나")
        teams = tuple(u.get("teams", []))
        if not all(isinstance(t, str) and _SLUG.match(t) for t in teams):
            raise AccountsError(f"{where}: teams는 영문 소문자 팀 아이디 목록")
        if not str(u.get("password", "")).startswith("pbkdf2_sha256$"):
            raise AccountsError(f"{where}: password에는 해시만 쓴다 (python -m src.security.auth hash)")
        users[username] = User(username, str(u.get("name", username)), u["role"], teams,
                               int(u.get("daily_limit", 30)))
        hashes[username] = u["password"]
    if not users:
        raise AccountsError(f"{source}: 계정이 없습니다")
    return Accounts(users, hashes, bool(data.get("demo", False)), source)


def load_accounts() -> Accounts:
    """USERS_TOML(시크릿 원문) → USERS_FILE(경로) 순서로 읽는다."""
    if raw := os.getenv("USERS_TOML"):
        return parse_accounts(tomllib.loads(raw), "USERS_TOML")
    if not USERS_FILE.exists():
        raise AccountsError(f"계정 파일이 없습니다: {USERS_FILE}")
    with USERS_FILE.open("rb") as f:
        return parse_accounts(tomllib.load(f), USERS_FILE.name)


def startup_problem(accounts: Accounts) -> str | None:
    if APP_ENV == "production" and accounts.demo:
        return ("운영 환경(APP_ENV=production)에서는 데모 계정 파일로 시작할 수 없습니다. "
                "USERS_TOML 시크릿에 실제 계정을 넣으세요.")
    return None


class LoginGuard:
    """아이디별 연속 실패를 세어, 잠금 시간 안에 max_failures번 틀리면 그 시간 동안 잠근다 (프로세스 메모리)."""

    def __init__(self, max_failures: int = LOGIN_MAX_FAILURES, lock_seconds: int = LOGIN_LOCK_MINUTES * 60,
                 clock=time.monotonic):
        self.max_failures, self.lock_seconds, self.clock = max_failures, lock_seconds, clock
        self._fails: dict[str, list[float]] = {}
        self._until: dict[str, float] = {}
        self._lock = Lock()

    def locked_for(self, username: str) -> int:
        """남은 잠금 시간(초). 잠기지 않았으면 0."""
        with self._lock:
            return max(0, int(self._until.get(username, 0) - self.clock() + 0.999))

    def failure(self, username: str) -> bool:
        """실패를 기록하고, 이번 실패로 잠겼으면 True."""
        now = self.clock()
        with self._lock:
            recent = [t for t in self._fails.get(username, []) if now - t < self.lock_seconds] + [now]
            self._fails[username] = recent
            if len(recent) >= self.max_failures:
                self._until[username] = now + self.lock_seconds
                self._fails[username] = []
                return True
            return False

    def success(self, username: str) -> None:
        with self._lock:
            self._fails.pop(username, None)


GUARD = LoginGuard()


@dataclass
class LoginResult:
    status: str               # ok | fail | locked
    user: User | None = None
    wait: int = 0             # 잠겼으면 남은 초


def authenticate(username: str, password: str, accounts: Accounts, guard: LoginGuard = GUARD) -> LoginResult:
    username = username.strip().lower()
    if wait := guard.locked_for(username):
        return LoginResult("locked", wait=wait)
    stored = accounts.hashes.get(username)
    ok = verify_password(password, stored or _DUMMY) and stored is not None
    if ok:
        guard.success(username)
        return LoginResult("ok", accounts.users[username])
    if guard.failure(username):
        return LoginResult("locked", wait=guard.locked_for(username))
    return LoginResult("fail")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "hash":
        pw = getpass("비밀번호: ")
        if pw != getpass("한 번 더: "):
            raise SystemExit("두 비밀번호가 다릅니다.")
        if len(pw) < 10:
            raise SystemExit("10자 이상으로 정하세요.")
        print(hash_password(pw))
    elif cmd == "list":
        acc = load_accounts()
        print(f"{acc.source} (데모 계정: {'예' if acc.demo else '아니오'})")
        for u in acc.users.values():
            print(f"  {u.username:10} {u.role_label:8} 팀 {','.join(u.teams) or '-':10} 일일 한도 {u.daily_limit}  {u.name}")
    else:
        raise SystemExit("사용법: python -m src.security.auth hash|list")


if __name__ == "__main__":
    main()
