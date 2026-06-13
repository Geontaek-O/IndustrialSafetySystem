"""
통합 런처 — main.py(웹/API)와 realtime.py(카메라/AI)를 한 터미널에서 동시 실행

사용법:
    python start.py

동작:
  - 두 스크립트를 자식 프로세스로 띄움
  - 각 출력 앞에 [MAIN]/[REAL] 태그 붙여서 한 터미널에 합쳐 출력
  - Ctrl+C 한 번이면 둘 다 정상 종료
  - 한 쪽이 죽으면 다른 쪽도 종료 (선택: AUTO_RESTART=True 면 재시작)

환경변수:
  AUTO_RESTART=1   한 프로세스가 죽으면 자동 재시작 (기본 0)
  STOP_ON_FAIL=1   한 쪽이 죽으면 다른 쪽도 즉시 종료 (기본 1)

[시스템 구조]
  start.py (런처)
  ├─ realtime.py → 포트 8000 (카메라 캡처 + YOLO 안전모 감지 + 얼굴 인식)
  │    └─ 3초 대기 후 ↓  (GPU/카메라 먼저 점유해야 충돌 없음)
  └─ main.py     → 포트 8080 (웹 대시보드 + REST API + DB 연동)
"""
import os
import sys
import time
import signal
import threading
import subprocess
from pathlib import Path

# ─────────── 설정 ───────────
# [실행 순서 설정] REAL → MAIN 순서로 띄움
# REAL(realtime.py)을 먼저 띄우는 이유:
#   GPU와 카메라를 realtime.py가 먼저 점유해야 main.py와 충돌하지 않음
# 색상 코드: 터미널 출력에서 REAL/MAIN 구분용
SCRIPTS = [
    {"name": "REAL", "file": "realtime.py", "color": "\033[35m"},  # 마젠타 — AI 서버
    {"name": "MAIN", "file": "main.py",     "color": "\033[36m"},  # 시안   — 웹 서버
]
RESET_COLOR = "\033[0m"

# 환경변수로 동작 제어
AUTO_RESTART = os.environ.get("AUTO_RESTART", "0") == "1"  # 자동 재시작 여부
STOP_ON_FAIL = os.environ.get("STOP_ON_FAIL", "1") == "1"  # 한 쪽 죽으면 다른 쪽도 종료
RESTART_DELAY = 3.0  # 재시작 대기 시간 (초)

# ─────────── 전역 상태 ───────────
processes = {}          # { "REAL": Popen객체, "MAIN": Popen객체 }
shutting_down = threading.Event()   # 종료 신호 — True가 되면 감시 루프 탈출


# [로그 출력] 자식 프로세스 stdout을 실시간으로 읽어 [REAL]/[MAIN] 태그 붙여 터미널 출력
def stream_output(name: str, color: str, proc: subprocess.Popen):
    """
    자식 프로세스의 stdout을 실시간으로 읽어 [TAG] 접두사 붙여 출력.
    각 자식마다 별도 데몬 스레드에서 실행됨.
    데몬 스레드이므로 런처 종료 시 자동 소멸.
    """
    prefix = f"{color}[{name}]{RESET_COLOR} "
    try:
        for line in iter(proc.stdout.readline, b""):
            if not line:
                break
            try:
                text = line.decode("utf-8", errors="replace").rstrip()
            except Exception:
                text = repr(line)
            print(f"{prefix}{text}", flush=True)
    except Exception as e:
        print(f"{prefix}[런처] 출력 스트림 오류: {e}", flush=True)


# [프로세스 시작] 스크립트 1개를 자식 프로세스로 실행, 로그 스트리밍 스레드 시작
def spawn(spec: dict) -> subprocess.Popen:
    """
    스크립트 1개를 자식 프로세스로 띄우고 stdout 스트리밍 스레드 시작.

    주요 처리:
      - sys.executable: 현재 가상환경의 Python 그대로 사용 (패키지 상속)
      - -u 플래그: 출력 버퍼링 OFF → 실시간 로그 확인 가능
      - CREATE_NEW_PROCESS_GROUP (Windows): Ctrl+C 신호 그룹 분리
      - PYTHONIOENCODING=utf-8: 한글/이모지 출력 시 cp949 오류 방지
    """
    name = spec["name"]
    file = spec["file"]
    color = spec["color"]

    if not Path(file).exists():
        print(f"{color}[{name}]{RESET_COLOR} ❌ 파일 없음: {file}", flush=True)
        return None

    # ★ 같은 Python 인터프리터 사용 (가상환경 그대로 상속됨)
    # ★ Windows에서 Ctrl+C 신호 그룹 분리: 런처가 신호 받으면 자식에게 전파
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    # ★ 자식 프로세스에 UTF-8 인코딩 강제 (cp949 이모지 인코딩 오류 방지)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [sys.executable, "-u", file],   # -u: 출력 버퍼링 끔 (실시간 로그)
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,       # stderr도 stdout으로 합쳐 캡처
        bufsize=1,
        creationflags=creationflags,
        env=env,
    )
    print(f"{color}[{name}]{RESET_COLOR} ✓ 시작 (PID {proc.pid})", flush=True)

    # stdout 스트리밍 스레드 시작 (daemon=True → 런처 종료 시 자동 소멸)
    t = threading.Thread(target=stream_output, args=(name, color, proc), daemon=True)
    t.start()

    return proc


# [프로세스 종료] 모든 자식 프로세스에 종료 신호 전송, timeout 초 안에 안 되면 강제 kill
def terminate_all(timeout: float = 5.0):
    """
    모든 자식 프로세스를 정상 종료 시도 → timeout 초 안에 안 되면 강제 kill.

    종료 단계:
      1) SIGTERM (Linux) / CTRL_BREAK_EVENT (Windows) 전송
      2) timeout 초까지 대기
      3) 미종료 프로세스 강제 kill
    """
    if not processes:
        return
    print("\n🛑 모든 자식 프로세스 종료 중...", flush=True)

    # 1단계: 정상 종료 신호
    for name, proc in processes.items():
        if proc.poll() is None:  # 아직 살아있음
            try:
                if os.name == "nt":
                    # Windows는 CTRL_BREAK_EVENT가 가장 안정적
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
                print(f"  → [{name}] 종료 신호 전송", flush=True)
            except Exception as e:
                print(f"  → [{name}] 종료 신호 실패: {e}", flush=True)

    # 2단계: 대기 후 미종료 시 강제 kill
    deadline = time.time() + timeout
    for name, proc in processes.items():
        remain = max(0.5, deadline - time.time())
        try:
            proc.wait(timeout=remain)
            print(f"  ✓ [{name}] 정상 종료", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  ⚠ [{name}] 타임아웃 → 강제 kill", flush=True)
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass


# [종료 신호 처리] Ctrl+C 1회 → graceful 종료 / Ctrl+C 2회 → 즉시 강제 kill
def signal_handler(signum, frame):
    """
    Ctrl+C(SIGINT) / SIGTERM 수신 시 호출.
      - 첫 번째 Ctrl+C: graceful 종료 (terminate_all → sys.exit(0))
      - 두 번째 Ctrl+C: 즉시 강제 kill → sys.exit(1)
    """
    if shutting_down.is_set():
        # 두 번째 Ctrl+C → 즉시 종료
        print("\n💀 강제 종료", flush=True)
        for proc in processes.values():
            try: proc.kill()
            except: pass
        sys.exit(1)
    shutting_down.set()
    terminate_all()
    sys.exit(0)


# [메인 실행] 신호 핸들러 등록 → 자식 프로세스 순차 시작 → 1초 주기 감시 루프 (죽으면 재시작 or 종료)
def main():
    """
    런처 메인 함수.

    흐름:
      1) SIGINT/SIGTERM 핸들러 등록
      2) SCRIPTS 순서대로 spawn() — 각 3초 간격
         (realtime.py가 GPU/카메라 먼저 초기화하도록 여유 부여)
      3) 1초 주기 감시 루프
         - 자식이 죽으면:
           a) AUTO_RESTART=True → RESTART_DELAY초 후 재시작
           b) STOP_ON_FAIL=True → 나머지 프로세스도 종료
         - 모든 자식 소멸 시 런처도 종료
    """
    print("=" * 60)
    print("🚀 통합 런처 시작")
    print(f"   AUTO_RESTART={AUTO_RESTART}  STOP_ON_FAIL={STOP_ON_FAIL}")
    print("   Ctrl+C 한 번이면 모든 프로세스 종료")
    print("=" * 60)

    signal.signal(signal.SIGINT, signal_handler)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, signal_handler)

    # 자식 프로세스 띄우기
    for spec in SCRIPTS:
        proc = spawn(spec)
        if proc:
            processes[spec["name"]] = proc
        time.sleep(3.0)  # 두 프로세스가 동시에 GPU/카메라 접근 충돌 방지

    if not processes:
        print("❌ 띄울 프로세스가 없음. 종료.")
        return

    # 메인 감시 루프 — 자식 상태 모니터링 (1초 주기)
    try:
        while not shutting_down.is_set():
            time.sleep(1)
            for spec in SCRIPTS:
                name = spec["name"]
                proc = processes.get(name)
                if proc is None:
                    continue
                rc = proc.poll()
                if rc is None:
                    continue  # 살아있음 → 계속 대기

                # 자식이 죽었을 때 처리
                color = spec["color"]
                print(f"{color}[{name}]{RESET_COLOR} ⚠ 프로세스 종료됨 (exit code {rc})", flush=True)
                processes.pop(name, None)

                if AUTO_RESTART and not shutting_down.is_set():
                    # a) 자동 재시작
                    print(f"{color}[{name}]{RESET_COLOR} 🔄 {RESTART_DELAY}초 후 재시작...", flush=True)
                    time.sleep(RESTART_DELAY)
                    new_proc = spawn(spec)
                    if new_proc:
                        processes[name] = new_proc
                elif STOP_ON_FAIL:
                    # b) 다른 프로세스도 함께 종료
                    print(f"⚠ STOP_ON_FAIL=1 → 다른 프로세스도 종료", flush=True)
                    shutting_down.set()
                    terminate_all()
                    return

            if not processes:
                print("ℹ️ 모든 프로세스 종료됨. 런처 종료.", flush=True)
                return

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()