# src/security/acl.py — 누가 어떤 문서를 볼 수 있나 (문서별 열람 권한)
#
# 문서마다 열람할 수 있는 주체(allow)를 카탈로그에 적고, 색인할 때 청크 메타데이터에 펼쳐 둔다.
# 검색은 사용자의 주체 목록으로 만든 필터를 벡터 DB 질의 자체에 넣는다. 권한 밖 청크는 검색 결과에 오지 않으므로
# LLM 프롬프트에도 들어갈 수 없다. 검색 뒤 can_read로 한 번 더 보는 것은 필터를 빠뜨리는 실수에 대비한 이중 확인이다.
#
# 주체 표기: all(외부 방문자 포함 모두) · employee(임직원 전체) · team:<팀> · role:<역할> · user:<아이디>
# 시스템 관리자(admin)도 문서는 자기 주체로 열람 권한이 있는 것만 본다 — 운영 권한과 열람 권한을 나눈다.
import re
from collections.abc import Iterable
from dataclasses import dataclass

LEVELS = ("공개", "내부", "기밀")     # 민감도 등급, 뒤로 갈수록 높다
ROLES = {"guest": "방문자", "researcher": "연구원", "ip": "특허팀", "admin": "시스템 관리자"}
FLAG_PREFIX = "acl_"
_PRINCIPAL = re.compile(r"^(all|employee|(team|role|user):[a-z0-9_]+)$")


def level_rank(level: str) -> int:
    if level not in LEVELS:
        raise ValueError(f"알 수 없는 민감도 등급: {level!r} (가능: {', '.join(LEVELS)})")
    return LEVELS.index(level)


def max_level(levels: Iterable[str]) -> str:
    """가장 높은 등급. 비어 있으면 공개."""
    return max(levels, key=level_rank, default=LEVELS[0])


@dataclass(frozen=True)
class User:
    username: str
    name: str
    role: str                     # ROLES의 키
    teams: tuple[str, ...] = ()
    daily_limit: int = 30         # 하루 LLM 요청 상한 (무료 등급 한도·비용 관리)

    @property
    def principals(self) -> list[str]:
        """이 사용자가 해당하는 주체 전부. 문서의 allow와 하나라도 겹치면 열람할 수 있다."""
        out = ["all"]
        if self.role != "guest":
            out.append("employee")
        out += [f"team:{t}" for t in self.teams]
        out += [f"role:{self.role}", f"user:{self.username}"]
        return out

    @property
    def role_label(self) -> str:
        return ROLES.get(self.role, self.role)


def flag(principal: str) -> str:
    """'team:dft' → 'acl_team_dft'. Chroma 메타데이터는 리스트를 받지 않아 주체마다 불리언 키로 펼친다."""
    return FLAG_PREFIX + principal.replace(":", "_")


def acl_metadata(allow: Iterable[str]) -> dict[str, bool]:
    return {flag(p): True for p in allow}


def where_for(user: User) -> dict:
    """벡터 DB 검색 필터: 사용자의 주체 중 하나라도 문서의 열람 주체에 있으면 통과."""
    clauses = [{flag(p): True} for p in user.principals]
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


def can_read(user: User, metadata: dict) -> bool:
    return any(metadata.get(flag(p)) is True for p in user.principals)


def validate(level: str, allow: list[str]) -> list[str]:
    """등급·열람 주체 설정의 오류 목록. 하나라도 있으면 그 문서는 색인하지 않는다 (설정이 틀리면 막는 쪽으로)."""
    errors = []
    if level not in LEVELS:
        errors.append(f"알 수 없는 등급 '{level}'")
    if not allow:
        errors.append("열람 주체(allow)가 비어 있음")
    bad = [p for p in allow if not _PRINCIPAL.match(str(p))]
    if bad:
        errors.append(f"주체 표기 오류 {bad}")
    if level in ("내부", "기밀") and "all" in allow:
        errors.append(f"{level} 문서를 all(외부 방문자 포함)에 열 수 없음")
    if level == "기밀" and "employee" in allow:
        errors.append("기밀 문서는 임직원 전체(employee)에 열 수 없음 — 팀·역할·사용자로 지정")
    return errors
