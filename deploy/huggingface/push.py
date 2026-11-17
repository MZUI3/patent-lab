# deploy/huggingface/push.py — 이 저장소를 Hugging Face Space(Docker)로 올린다
#
# 준비  1) huggingface.co 계정 → 설정 > Access Tokens에서 쓰기(write) 토큰 발급
#       2) (처음 한 번) 이 스크립트가 Space를 만든다. 하드웨어는 기본값 CPU basic(무료)
#       3) Space 설정 > Variables and secrets
#            시크릿: GEMINI_API_KEY, USERS_TOML(운영 계정 TOML 원문 — config/users.toml 형식, demo = false)
#            변수:   LLM_MAX_LEVEL=내부 (가상 데이터 데모일 때만. 기본값은 Gemini 무료 등급 → 공개)
# 실행  HF_TOKEN=hf_... python deploy/huggingface/push.py --space <아이디>/patent-lab [--public]
#
# GitHub README와 Space README(앞머리 YAML 필요)가 달라, Space에는 deploy/huggingface/README.md를 README.md로 올린다.
# 공개 Space로 열면 저장소 파일(data/의 KIPRIS 수집 데이터 포함)도 공개된다 — 먼저 KIPRIS Plus 재배포 조건을 확인한다.
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
# 작업 폴더를 그대로 올리므로 git이 무시하는 로컬 파일(.env·색인·로그·가상환경)도 여기서 빼야 한다
IGNORE = [".git/*", ".github/*", ".claude/*", "venv/*", ".venv/*", ".env", "chroma_db/*", "logs/*", "*.log", "docs/*",
          "tests/*", "deploy/*", "*.zip", "*.pdf", "submission/*", "체크리스트/*", "**/__pycache__/*",
          ".pytest_cache/*", ".ruff_cache/*", "README.md"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Hugging Face Space로 배포")
    ap.add_argument("--space", required=True, help="<아이디>/<Space 이름>")
    ap.add_argument("--public", action="store_true", help="처음 만들 때 공개 Space로 (기본은 비공개)")
    args = ap.parse_args()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN 환경변수(쓰기 권한 토큰)가 필요합니다.")
    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(ROOT), ignore_patterns=IGNORE,
                      commit_message="Patent Lab 배포")
    api.upload_file(path_or_fileobj=str(ROOT / "deploy" / "huggingface" / "README.md"), path_in_repo="README.md",
                    repo_id=args.space, repo_type="space", commit_message="Space README")
    print(f"올렸습니다: https://huggingface.co/spaces/{args.space} (빌드 10~20분, 진행 상황은 Space의 Logs 탭)")


if __name__ == "__main__":
    main()
