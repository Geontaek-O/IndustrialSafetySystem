import os
import sys
import time
import signal
import threading
import subprocess
from pathlib import Path

# ===== 설정 =====
SCRIPTS = [
    {"name": "REAL", "file": "realtime.py", "color": "\033[35m"},
    {"name": "MAIN", "file": "main.py",     "color": "\033[36m"},
]
RESET_COLOR = "\033[0m"

# 환경변수로 동작 제어
AUTO_RESTART = os.environ.get("AUTO_RESTART", "0") == "1"   # 죽으면 자동 재시작
STOP_ON_FAIL = os.environ.get("STOP_ON_FAIL", "1") == "1"   # 하나 죽으면 다 종료
RESTART_DELAY = 3.0

# ===== 전역 상태 =====
processes = {}
shutting_down = threading.Event()


def stream_output(name: str, color: str, proc: subprocess.Popen):
    """자식 프로세스 stdout에 [이름] 접두사 붙여서 출력."""
    prefix = f"{color}[{name}]{RESET_COLOR} "
    try:
        for line in iter(proc.stdout.readline, b""):
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            print(f"{prefix}{text}", flush=True)
    except Exception as e:
        print(f"{prefix}[런처] 출력 스트림 오류: {e}", flush=True)


def spawn(spec: dict) -> subprocess.Popen:
    name, file, color = spec["name"], spec["file"], spec["color"]

    if not Path(file).exists():
        print(f"{color}[{name}]{RESET_COLOR} ❌ 파일 없음: {file}", flush=True)
        return None

    # 윈도우는 CREATE_NEW_PROCESS_GROUP 줘야 Ctrl+C 신호를 자식한테 따로 보낼 수 있음
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

    # 윈도우 cp949 환경에서 이모지 출력 시 UnicodeEncodeError 방지
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # -u: unbuffered (실시간 로그용)
    # sys.executable: 현재 파이썬 그대로 사용 → venv 상속됨
    proc = subprocess.Popen(
        [sys.executable, "-u", file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        creationflags=creationflags,
        env=env,
    )
    print(f"{color}[{name}]{RESET_COLOR} ✓ 시작 (PID {proc.pid})", flush=True)

    # daemon=True: 메인 종료 시 함께 종료
    threading.Thread(target=stream_output, args=(name, color, proc), daemon=True).start()
    return proc


def terminate_all(timeout: float = 5.0):
    """2단계 종료: 정상 종료 신호 → 타임아웃 시 강제 kill."""
    if not processes:
        return
    print("\n🛑 모든 자식 프로세스 종료 중...", flush=True)

    # 1단계: 정상 종료 신호
    for name, proc in processes.items():
        if proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
                print(f"  → [{name}] 종료 신호 전송", flush=True)
            except Exception as e:
                print(f"  → [{name}] 종료 신호 실패: {e}", flush=True)

    # 2단계: 대기 후 강제 종료
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


def signal_handler(signum, frame):
    # 첫 Ctrl+C: 정상 종료 시도 / 두 번째: 강제 kill
    if shutting_down.is_set():
        print("\n💀 강제 종료", flush=True)
        for proc in processes.values():
            try: proc.kill()
            except: pass
        sys.exit(1)
    shutting_down.set()
    terminate_all()
    sys.exit(0)


def main():
    print("=" * 60)
    print("🚀 통합 런처 시작")
    print(f"   AUTO_RESTART={AUTO_RESTART}  STOP_ON_FAIL={STOP_ON_FAIL}")
    print("   Ctrl+C 한 번이면 모든 프로세스 종료")
    print("=" * 60)

    signal.signal(signal.SIGINT, signal_handler)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, signal_handler)

    # 두 프로세스가 GPU/카메라 동시 접근 시 충돌 → 3초 텀 두고 순차 시작
    for spec in SCRIPTS:
        proc = spawn(spec)
        if proc:
            processes[spec["name"]] = proc
        time.sleep(3.0)

    if not processes:
        print("❌ 띄울 프로세스가 없음. 종료.")
        return

    # 1초 간격으로 자식 프로세스 상태 감시
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
                    continue  # 살아있음

                color = spec["color"]
                print(f"{color}[{name}]{RESET_COLOR} ⚠ 프로세스 종료됨 (exit code {rc})", flush=True)
                processes.pop(name, None)

                if AUTO_RESTART and not shutting_down.is_set():
                    print(f"{color}[{name}]{RESET_COLOR} 🔄 {RESTART_DELAY}초 후 재시작...", flush=True)
                    time.sleep(RESTART_DELAY)
                    new_proc = spawn(spec)
                    if new_proc:
                        processes[name] = new_proc
                elif STOP_ON_FAIL:
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