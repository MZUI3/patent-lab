# Dockerfile — Patent Lab (Streamlit). Hugging Face Spaces(Docker, 무료 CPU basic 2 vCPU·16GB)에서 그대로 돈다.
#
# 빌드  docker build -t patent-lab .                     임베딩 모델(bge-m3, 약 2.2GB)을 이미지에 넣는다
#       docker build --build-arg PRELOAD_MODEL=0 ...      빌드 확인용 (모델은 처음 검색할 때 받는다)
# 실행  docker run -p 8501:8501 -e GEMINI_API_KEY=... -e USERS_TOML="$(cat users.prod.toml)" patent-lab
#
# 시크릿(환경변수): GEMINI_API_KEY · USERS_TOML(운영 계정 — 데모 계정 파일로는 시작하지 않는다)
# 선택: LLM_MAX_LEVEL(기본 경로 상한) · LOCAL_LLM_BASE_URL(사내 모델) · KIPRIS_API_KEY · WIKI_CONTACT
FROM python:3.12-slim

ARG PRELOAD_MODEL=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    APP_ENV=production

# Hugging Face Spaces는 UID 1000 사용자로 컨테이너를 돌린다. 앱 폴더(색인·감사 로그를 쓰는 곳)는 그 사용자 소유로
RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown -R user:user /home/user
WORKDIR /home/user/app

# CPU용 torch를 먼저 깔아 CUDA 패키지(수 GB)를 피한다. 나머지는 requirements.txt의 고정 버전을 그대로 쓴다
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==$(grep '^torch==' requirements.txt | cut -d= -f3)" \
 && pip install -r requirements.txt

USER user
# 무료 Space는 재시작할 때 디스크가 비워지므로, 모델을 이미지에 넣어 뜰 때마다 2.2GB를 받지 않게 한다
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"; fi
# 모델을 넣었으면 실행 중에는 Hugging Face Hub에 묻지 않는다 (넣지 않았으면 처음 검색할 때 받는다)
ENV HF_HUB_OFFLINE=${PRELOAD_MODEL}

COPY --chown=user . .

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status != 200)"
# 운영에서는 코드가 바뀌지 않으므로 파일 감시를 끈다 (켜 두면 transformers 모듈을 훑으며 로그를 쏟아낸다)
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.fileWatcherType=none"]
