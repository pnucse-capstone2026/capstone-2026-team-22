from pathlib import Path

from pyboy import PyBoy
import os
import sys

_ROOT = Path(__file__).resolve().parents[1]

# --- 설정 ---
ROM_FILE = str(_ROOT / "assets" / "roms" / "pokered.gb")
LOAD_FILE = str(_ROOT / "assets" / "states" / "init.state")
SAVE_FILE = str(_ROOT / "assets" / "states" / "init2.state")
EMULATION_SPEED = 3  # 4. 게임 속도 (1~5 추천)
# ------------

# 롬 파일 확인
if not os.path.exists(ROM_FILE):
    print(f"오류: '{ROM_FILE}' 파일을 찾을 수 없습니다.")
    sys.exit()

try:
    # PyBoy 초기화
    pyboy = PyBoy(ROM_FILE, window="SDL2")

    # [추가됨] 시작 파일 불러오기 로직
    if os.path.exists(LOAD_FILE):
        with open(LOAD_FILE, "rb") as f:
            pyboy.load_state(f)
        print(f"✅ '{LOAD_FILE}' 파일을 성공적으로 불러왔습니다!")
    else:
        print(f"⚠️ '{LOAD_FILE}' 파일이 없습니다. 게임을 처음부터 시작합니다.")

except Exception as e:
    print(f"초기화 중 오류 발생: {e}")
    sys.exit()

# 게임 속도 설정
pyboy.set_emulation_speed(EMULATION_SPEED)

print("--- PyBoy 에뮬레이터 실행 중 ---")
print(f"1. 게임을 진행하세요. (속도: {EMULATION_SPEED}배)")
print(f"2. 원하는 지점에서 **게임 창(X 버튼)을 닫으세요**.")
print(f"   (창을 닫으면 '{SAVE_FILE}'에 현재 상태가 저장됩니다.)")

# 게임 루프
try:
    while pyboy.tick():
        pass
except OSError:
    pass  # 창 닫을 때 발생하는 에러 무시

# --- 저장 로직 ---
try:
    # 창이 닫히면 저장 수행
    with open(SAVE_FILE, "wb") as f:
        pyboy.save_state(f)
    print(f"\n💾 게임 상태가 '{SAVE_FILE}'에 저장되었습니다.")

except Exception as e:
    print(f"저장 중 오류 발생: {e}")

finally:
    pyboy.stop()