#!/bin/sh
# 설치 및 테스트 — 저장소 루트에서 ./install_and_build.sh (Linux · macOS · Windows Git Bash)
# 파이썬을 바꾸려면: PYTHON=python3.11 ./install_and_build.sh
set -e
cd "$(dirname "$0")/src"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

echo '1. 가상환경 (.venv)'
"$PY" -m venv .venv
if [ -f .venv/bin/activate ]; then . .venv/bin/activate; else . .venv/Scripts/activate; fi

echo '2. 의존성 설치'
python -m pip install --upgrade pip
pip install -r requirements.txt

echo '3. 설치 확인 (모듈을 전부 불러온다 — ROM 없이 된다)'
PYTHONPATH=. PYTHONIOENCODING=utf-8 python -c "import scripts.play; print('설치 확인 완료')"

echo
echo '다음 파일을 넣은 뒤 실행하세요:'
echo '  src/assets/roms/pokered.gb   (ROM)'
echo '  src/assets/roms/pokered.sym  (pret/pokered 빌드 산출물)'
echo '실행: cd src && python play.py   (명령: auto · rec on · rec off · stop)'
