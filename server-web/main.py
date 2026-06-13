"""
╔══════════════════════════════════════════════════════════════════════╗
║  main.py — 웹 대시보드 + REST API 서버 (포트 8080)                    ║
║                                                                      ║
║  실행: python main.py  (또는 start.py 가 자동으로 띄움)               ║
║  포트: 8080  (realtime.py 는 8000 — 절대 같은 포트 금지)              ║
║                                                                      ║
║  [웹 페이지 URL]                                                      ║
║   /              → /login 리다이렉트                                  ║
║   /login         → 웹 로그인                                         ║
║   /register      → 회원가입 + AI 얼굴 등록                            ║
║   /index         → 메인 대시보드                                      ║
║   /map           → 실시간 위치 지도                                   ║
║   /tables        → 작업자 목록                                        ║
║   /charts        → 통계 차트                                          ║
║   /live          → 카메라 라이브 뷰 (게스트 허용)                     ║
║   /live2         → 상세 라이브 대시보드                               ║
║   /realtime      → realtime.py UI 프록시                             ║
║   /daily_report  → 일일 안전 보고서                                   ║
║   /mypage        → 마이페이지                                         ║
║   /logout        → 로그아웃                                           ║
║                                                                      ║
║  [앱 전용 JSON API]                                                   ║
║   POST /api/login           → 로그인 + 출근 체크인                    ║
║   POST /api/logout          → 로그아웃                                ║
║   POST /api/send_location   → GPS 위치 전송 + 구역 알림               ║
║   POST /api/send_heart_rate → 심박수 전송                             ║
║   POST /api/emergency_sos   → 긴급 SOS (낙하/음성)                   ║
║   POST /api/stt_alert       → 음성 STT SOS 감지                       ║
║   GET  /api/poll_alerts     → 안전모 알림 폴링                        ║
║                                                                      ║
║  [데이터/관리 API]                                                    ║
║   GET/POST /api/zones              → 안전구역 불러오기/저장           ║
║   POST /api/construction_zone      → 공사구역 상세 저장               ║
║   GET  /api/worker_status_list     → 작업자 현황 목록                 ║
║   GET  /api/worker_logs/{email}    → 작업자 당일 이벤트 로그          ║
║   GET  /api/get_recent_logs        → 최근 5분 GPS 로그 (지도용)       ║
║   GET  /api/get_violations         → 오늘 위반 목록                   ║
║   GET  /api/download_violations    → 위반 내역 CSV 다운로드           ║
║   POST /api/data                   → IoT 센서 데이터 수신             ║
║                                                                      ║
║  [realtime.py 연동]                                                   ║
║   POST /api/violation   → AI 감지 위반 수신 → DB 저장 + 앱 알림      ║
║   POST /api/void_violation → 미식별자 위반 정정                       ║
║   GET  /violations      → AI + DB 위반 머지 반환                      ║
║   GET  /video_feed      → MJPEG 스트림 프록시                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import random
import uvicorn
import os
import pickle
import asyncio
import json
import zlib
import gzip as _gzip
import io as _io
import tempfile
import traceback
import csv
# ★ [신규] STT (음성 인식) — SOS 키워드 감지용
try:
    import speech_recognition as _sr
    import io as _audio_io
    _STT_AVAILABLE = True
except ImportError:
    _STT_AVAILABLE = False
    print("⚠️ speech_recognition 미설치 — pip install SpeechRecognition 실행 필요")

# STT 음성 인식 결과에서 이 키워드가 있으면 SOS 상황으로 판정
# /api/stt_alert 엔드포인트에서 사용
SOS_KEYWORDS = ["살려", "도와", "구해", "위험", "사람", "다쳤", "쓰러",
                "help", "sos", "emergency"]

def _recognize_korean(wav_bytes: bytes) -> str:
    """WAV 바이트 → 한국어 텍스트 (실패 시 빈 문자열, SOS 키워드 검출용)"""
    if not _STT_AVAILABLE:
        return ""
    try:
        r = _sr.Recognizer()
        with _sr.AudioFile(_audio_io.BytesIO(wav_bytes)) as src_audio:
            audio = r.record(src_audio)
        return r.recognize_google(audio, language="ko-KR")
    except _sr.UnknownValueError:
        return ""
    except Exception as e:
        print(f"⚠️ STT 오류: {e}")
        return ""

import threading
import calendar
import time   # ★ API 호출 제한 방지용 (날씨 캐싱)
import httpx  # ★ realtime 서버 프록시 및 외부 API 비동기 호출용
import cv2
import numpy as np
from datetime import datetime, date
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from databases import Database

# AI 모듈 임포트 (가상환경 100% 보장)
import Register

# =============================================
#  설정 변수 (경로, 상수, API 키 등)
# =============================================
FACE_DB_DIR  = "face_db"   # 한 사람당 face_db/{name}.json 파일 1개
MAX_FACES = 30

# 🌟 OpenWeather API 키
OPENWEATHER_API_KEY = "c79ac27b872a8aa855cd5aa83f0c6e1a"

# ★ Realtime 서버 (start.py가 별도 프로세스로 실행됨)
# realtime.py 는 port 8000 에서 동작 (main.py 는 8080) — 절대 같은 포트 쓰면 안 됨
# main.py 가 /video_feed, /violations, /stats 등을 프록시로 중계함
REALTIME_SERVER_URL = "http://127.0.0.1:8000"

# ★ Realtime의 위반 증거 이미지 폴더 (realtime.py와 동일 경로)
EVIDENCE_DIR = "evidence"

# ★ AI 처리를 위한 스레드풀 및 세마포어
# _ai_executor  : 얼굴 임베딩 추출(CPU/GPU 연산)을 별도 스레드에서 실행
#                 → FastAPI 비동기 루프가 블로킹되지 않도록 분리
# _ai_semaphore : 동시 AI 처리 최대 2건 제한 (GPU OOM 방지)
# _register_lock: Register.MAX_FACES 전역 변수 임시 변경 시 동시성 보호
_ai_executor  = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai_worker")
_ai_semaphore = asyncio.Semaphore(2)
_register_lock = threading.Lock()
_background_tasks = set()

# ★ [신규] 사용자별 현재 구역 상태 추적 (진입/퇴장 시에만 1회 알림 발생)
# 형식: { email: {"zone_name": str, "zone_type": str} } 또는 {email: None}
# 이전 구역과 비교해 DANGER 진입/퇴장 시에만 zone_alert 생성 → 반복 진동 방지
_user_zone_states: dict = {}
_user_zone_lock = threading.Lock()

# ═════════════════════════════════════════════════════════════════
# ★★★ [신규] 안전모 미착용 알림 (앱으로 진동/알림 — 폴링 방식) ★★★
#   - realtime.py가 /api/violation 으로 위반(이름)을 보내면:
#       name → email 매핑 후 _pending_helmet_alerts 에 적재
#   - 앱은 /api/send_location 응답의 helmet_alert 를 보고 즉시 진동/알림
#   - 서버가 큐에서 1회만 내려주므로 중복 진동이 없음
#   - (앱이 켜져 있을 때만 동작. 앱 종료/백그라운드 알림은 폴링 방식 범위 밖)
# ═════════════════════════════════════════════════════════════════
_pending_helmet_alerts: dict = {}     # { email: [ {type, image_url, message, ts}, ... ] }
_helmet_alert_lock = threading.Lock()
HELMET_ALERT_TTL = 120.0              # 폴링으로 안 가져간 알림 보관 시간(초)
# ★ 알림 전달 방식: 앱이 /api/send_location 또는 /api/poll_alerts 호출 시
#   응답 본문의 helmet_alert 필드로 1회 전달 후 큐 비움 → 중복 진동 없음

# ★ [신규] safety_zones 메모리 캐시
#   - 구역 저장(/api/save_zones) 시 자동 갱신 → DB 재조회 없이 즉시 반영
#   - worker_status_list / index 초기 로드 양쪽에서 사용
#   - 우선순위: DANGER(1) > CONSTRUCTION(2) > SAFE(3) > ATTENDANCE(4)
_cached_zones: list = []          # [{"zone_name":str, "zone_type":str, "path":list, "priority":int}, ...]
_zone_cache_lock = threading.Lock()

def _refresh_zone_cache_sync(rows):
    """DB rows → 메모리 캐시 갱신 (동기, 잠금 포함)"""
    priority = {"DANGER": 1, "CONSTRUCTION": 2, "SAFE": 3, "ATTENDANCE": 4}
    parsed = []
    for r in rows:
        try:
            path = json.loads(r["path_data"])
            parsed.append({
                "zone_name": r["label"],
                "zone_type": r["zone_type"],
                "path":      path,
                "priority":  priority.get(r["zone_type"], 99)
            })
        except Exception:
            continue
    with _zone_cache_lock:
        global _cached_zones
        _cached_zones = parsed

async def _ensure_zone_cache():
    """캐시가 비어 있으면 DB에서 한 번 로드"""
    with _zone_cache_lock:
        if _cached_zones:
            return
    rows = await database.fetch_all("SELECT * FROM safety_zones")
    _refresh_zone_cache_sync(rows)

def _zone_for_pos(lat: float, lon: float):
    """
    캐시된 구역 목록에서 현재 좌표가 속한 가장 우선순위 높은 구역 반환.
    반환: {"zone_name": str, "zone_type": str} 또는 None
    """
    with _zone_cache_lock:
        zones = list(_cached_zones)
    matched = [z for z in zones if point_in_polygon(lat, lon, z["path"])]
    if not matched:
        return None
    matched.sort(key=lambda z: z["priority"])
    top = matched[0]
    return {"zone_name": top["zone_name"], "zone_type": top["zone_type"]}

# ★ [신규] 사용자별 최신 심박 상태 추적 (index '현재 상태' 표시용)
#   - _user_heart_rates  : { email: {"bpm":int,"avg":int,"status":str,"label":str,"level":str,"ts":str} }
#   - _user_hr_last_logged: { email: status }  ← 상태가 바뀔 때만 이력(violations)에 기록 (스팸 방지)
_user_heart_rates: dict   = {}
_user_hr_last_logged: dict = {}
_user_hr_lock = threading.Lock()

# 앱(MainActivity.kt)의 HeartRateAlert enum → 한글 라벨/위험도 매핑
#   bpm < 50 → TOO_LOW(서맥) / bpm > 120 → TOO_HIGH(빈맥) / bpm > 100 → HIGH(주의) / 그 외 NORMAL(정상)
#   level: normal/warn/danger → index.html 뱃지 색상 결정
HR_STATUS_MAP = {
    "NORMAL":   {"label": "심박 정상",     "level": "normal", "kor": "정상"},
    "HIGH":     {"label": "심박 주의",     "level": "warn",   "kor": "주의"},
    "TOO_HIGH": {"label": "심박 위험·빈맥", "level": "danger", "kor": "빈맥"},
    "TOO_LOW":  {"label": "심박 위험·서맥", "level": "danger", "kor": "서맥"},
}

# 1. 앱 초기 설정
app = FastAPI()

# =============================================
#  ⚡ 성능 최적화: 선택적 GZip 압축 미들웨어
#  - JSON/HTML/CSS/JS/SVG 등 텍스트 응답만 gzip 압축 (모바일·저속망에서 큰 효과)
#  - /video_feed 같은 라이브 스트림(multipart)·CSV 스트림은 절대 건드리지 않고 그대로 통과
#    → 스트림 버퍼링/끊김 없음
#  - 클라이언트가 gzip 을 지원할 때만 동작, 작은 응답(<500B)은 압축 생략
# =============================================
class _SelectiveGZipResponder:
    def __init__(self, app, minimum_size, compressible):
        self.app = app
        self.minimum_size = minimum_size
        self.compressible = compressible
        self.send = None
        self.start_message = None
        self.compress = False
        self.buffer = bytearray()

    async def run(self, scope, receive, send):
        self.send = send
        await self.app(scope, receive, self._send)

    async def _send(self, message):
        t = message["type"]
        if t == "http.response.start":
            ctype, has_enc = "", False
            for k, v in message.get("headers", []):
                lk = k.lower()
                if lk == b"content-type":
                    ctype = v.decode("latin-1").lower()
                elif lk == b"content-encoding":
                    has_enc = True
            self.compress = (not has_enc) and any(ctype.startswith(c) for c in self.compressible)
            if self.compress:
                self.start_message = message      # 본문 모은 뒤 헤더 수정해 전송
            else:
                await self.send(message)          # 비압축: 즉시 통과 (스트리밍 안전)
            return
        if t == "http.response.body":
            if not self.compress:
                await self.send(message)          # 진정한 패스스루 — 버퍼링 없음
                return
            self.buffer += message.get("body", b"")
            if message.get("more_body", False):
                return
            body = bytes(self.buffer)
            if len(body) < self.minimum_size:
                await self.send(self.start_message)
                await self.send({"type": "http.response.body", "body": body, "more_body": False})
                return
            buf = _io.BytesIO()
            with _gzip.GzipFile(mode="wb", fileobj=buf, compresslevel=6) as gz:
                gz.write(body)
            gzipped = buf.getvalue()
            headers = [(k, v) for k, v in self.start_message.get("headers", [])
                       if k.lower() not in (b"content-length", b"content-encoding")]
            headers.append((b"content-encoding", b"gzip"))
            headers.append((b"content-length", str(len(gzipped)).encode("latin-1")))
            headers.append((b"vary", b"Accept-Encoding"))
            new_start = dict(self.start_message)
            new_start["headers"] = headers
            await self.send(new_start)
            await self.send({"type": "http.response.body", "body": gzipped, "more_body": False})
            return
        await self.send(message)


class SelectiveGZipMiddleware:
    COMPRESSIBLE = (
        "application/json", "text/html", "text/css", "text/plain",
        "application/javascript", "text/javascript", "image/svg+xml",
    )
    def __init__(self, app, minimum_size: int = 500):
        self.app = app
        self.minimum_size = minimum_size

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        accept = ""
        for k, v in scope.get("headers", []):
            if k == b"accept-encoding":
                accept = v.decode("latin-1").lower()
                break
        if "gzip" not in accept:
            await self.app(scope, receive, send)
            return
        await _SelectiveGZipResponder(self.app, self.minimum_size, self.COMPRESSIBLE).run(scope, receive, send)


# =============================================
#  ⚡ 성능 최적화: 정적 파일 장기 캐시
#  - /static, /evidence 의 파일에 Cache-Control 부여 → 재접속 시 재다운로드 안 함
# =============================================
class CachedStaticFiles(StaticFiles):
    def __init__(self, *args, cachecontrol: str = "public, max-age=86400", **kwargs):
        self.cachecontrol = cachecontrol
        super().__init__(*args, **kwargs)

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        if resp.status_code == 200:
            resp.headers.setdefault("Cache-Control", self.cachecontrol)
        return resp


# ⚡ gzip 미들웨어 등록 (세션 미들웨어보다 바깥쪽에서 최종 응답을 압축)
app.add_middleware(SelectiveGZipMiddleware, minimum_size=500)

# 세션 미들웨어
app.add_middleware(
    SessionMiddleware, 
    secret_key="safety_key_1234_hsu",
    same_site="lax", 
    https_only=False,
    max_age=3600
)

# 정적 파일 및 템플릿 경로 (⚡ 1일 캐시)
app.mount("/static", CachedStaticFiles(directory="static", cachecontrol="public, max-age=86400"), name="static")

# 위반 증거 이미지 마운트 (⚡ 증거 이미지는 한번 생성되면 안 바뀜 → 7일 캐시)
os.makedirs(EVIDENCE_DIR, exist_ok=True)
app.mount("/evidence", CachedStaticFiles(directory=EVIDENCE_DIR, cachecontrol="public, max-age=604800"), name="evidence")
templates = Jinja2Templates(directory="templates")

# 업로드 영상 임시 저장 폴더
UPLOAD_DIR = "temp_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 2. 데이터베이스 설정 (MySQL)
DATABASE_URL = "mysql+aiomysql://root:0426@localhost/uhs_db"
database = Database(DATABASE_URL)

# [데이터] 안전 법령 리스트
safety_laws = [
    {"title": "산업안전보건법 제5조", "content": "사업주는 근로자의 생명 보전과 안전 및 보건을 유지·증진하고 국가의 예방 시책에 따라야 합니다.", "source": "국가법령정보센터", "url": "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=55226"},
    {"title": "산업안전보건법 제29조", "content": "사업주는 소속 근로자에게 정기적으로 안전보건교육을 실시해야 합니다.", "source": "산업안전보건법", "url": "https://www.law.go.kr/lsInfoP.do?lsiSeq=250667"},
    {"title": "산업안전보건법 제36조", "content": "사업주는 유해·위험요인을 찾아내어 위험성평가를 실시해야 합니다.", "source": "국가법령정보센터", "url": "https://www.law.go.kr/lsInfoP.do?lsiSeq=250667#J36:0"},
    {"title": "산업안전보건법 제38조", "content": "사업주는 기계·기구, 폭발성 물질, 전기 작업 시 안전조치를 취해야 합니다.", "source": "산업안전보건법", "url": "https://www.law.go.kr/lsInfoP.do?lsiSeq=250667#J38:0"},
    {"title": "산업안전보건법 제51조", "content": "사업주는 산업재해 발생 위험 시 작업을 중지시키고 대피시켜야 합니다.", "source": "국가법령정보센터", "url": "https://www.law.go.kr/lsInfoP.do?lsiSeq=250667#J51:0"},
    {"title": "중대재해처벌법 제4조", "content": "경영책임자는 안전·보건 관리체계 구축 및 이행 조치를 취해야 합니다.", "source": "중대재해처벌법", "url": "https://www.law.go.kr/lsInfoP.do?lsiSeq=228514#J4:0"},
    {"title": "중대재해처벌법 제6조", "content": "안전 조치 위반으로 중대재해 발생 시 사업주 등은 처벌받을 수 있습니다.", "source": "국가법령정보센터", "url": "https://www.law.go.kr/lsInfoP.do?lsiSeq=228514#J6:0"},
    {"title": "안전보건규칙 제22조", "content": "사업주는 작업장 내에 근로자가 사용할 안전한 통로를 설치해야 합니다.", "source": "산업안전보건기준 규칙", "url": "https://www.law.go.kr/lsInfoP.do?lsiSeq=244353#J22:0"},
    {"title": "안전보건규칙 제32조", "content": "사업주는 유해 작업 시 보호구를 지급하고 근로자는 이를 착용해야 합니다.", "source": "국가법령정보센터", "url": "https://www.law.go.kr/lsInfoP.do?lsiSeq=244353#J32:0"},
    {"title": "안전보건규칙 제42조", "content": "2미터 이상 추락 위험 장소 작업 시 안전한 작업 발판을 설치해야 합니다.", "source": "산업안전보건기준 규칙", "url": "https://www.law.go.kr/lsInfoP.do?lsiSeq=244353#J42:0"}
]

# =============================================
#  헬퍼 함수: 오늘 출근한 사용자 이메일 집합 가져오기
# =============================================
async def get_today_attendee_emails() -> set:
    """오늘 attendance 테이블에 기록이 있는 사용자 이메일 집합을 반환"""
    today = date.today()
    rows = await database.fetch_all(
        "SELECT email FROM attendance WHERE access_date = :d",
        {"d": today}
    )
    return {r['email'] for r in rows}


# [서버 시작] DB 연결 + 7일 지난 GPS 로그 삭제 + realtime.py 헬스체크 + 구역 캐시 초기 로드
@app.on_event("startup")
async def startup():
    # 1) DB 연결
    await database.connect()

    # 2) 7일 지난 GPS 로그 자동 삭제 (raw_logs 테이블 무한 증가 방지)
    try:
        result = await database.execute(
            "DELETE FROM raw_logs WHERE reg_date < DATE_SUB(NOW(), INTERVAL 7 DAY)"
        )
        print(f"🧹 7일 이상 된 위치 로그 정리 완료 ({result}건 삭제)")
    except Exception as e:
        print(f"⚠️ 로그 정리 실패: {e}")

    # 🌐 Realtime 서버 헬스체크
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{REALTIME_SERVER_URL}/stats")
                if r.status_code == 200:
                    print(f"✅ Realtime 서버 연결 OK: {REALTIME_SERVER_URL}")
                    return
        except Exception:
            if attempt < 4:
                await asyncio.sleep(2.0)
    print(f"⚠️ Realtime 서버 연결 실패 ({REALTIME_SERVER_URL}) - realtime.py 확인 요망")

    # ★ 구역 캐시 초기 로드
    try:
        rows = await database.fetch_all("SELECT * FROM safety_zones")
        _refresh_zone_cache_sync(rows)
        print(f"🗺️  구역 캐시 로드 완료 ({len(_cached_zones)}개)")
    except Exception as e:
        print(f"⚠️ 구역 캐시 초기 로드 실패: {e}")

# [서버 종료] DB 연결 해제 + AI 스레드풀 정리
@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    _ai_executor.shutdown(wait=False)

# =============================================
#  [최적화] 날씨 API 가져오기 함수 (캐싱 적용)
# =============================================
_cached_weather = {"temp": "-", "description": "로딩 중...", "icon": ""}
_last_weather_time = 0

async def get_weather(city="Suwon"):
    global _cached_weather, _last_weather_time
    current_time = time.time()
    
    # ★ 중요: OpenWeather API는 무료 버전에서 호출 횟수 제한이 있습니다.
    # live2.html 그래프가 5초마다 호출하므로, 서버가 매번 외부 API를 찌르지 않도록
    # 최근 60초 이내면 기존에 저장된(캐시된) 온도를 즉시 반환하여 API 차단을 막습니다.
    if current_time - _last_weather_time < 60 and _cached_weather["temp"] != "-":
        return _cached_weather

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric&lang=kr"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                _cached_weather = {
                    "temp": round(data["main"]["temp"], 1),
                    "description": data["weather"][0]["description"],
                    "icon": data["weather"][0]["icon"]
                }
                _last_weather_time = current_time
                return _cached_weather
    except Exception as e:
        print(f"⚠️ 날씨 API 연결 실패: {e}")
    
    return _cached_weather

# [날씨] 수원 현재 날씨 조회 (live2.html 5초 폴링용, 60초 캐시 적용)
@app.get("/api/weather")
async def api_weather():
    """
    live2.html에서 5초마다 수원 날씨 온도를 가져가기 위한 전용 API.
    내부적으로 60초 캐시를 사용하므로 OpenWeather API 호출 제한에 걸리지 않음.
    """
    return await get_weather("Suwon")

# =============================================
#  AI 얼굴 등록 (가상환경 100% 가동 보장)
# =============================================

def _extract_ai_sync(video_path: str, max_faces: int):
    with _register_lock:
        original = Register.MAX_FACES
        Register.MAX_FACES = max_faces
        try:
            features = Register.extract_faces_from_video(video_path)
        finally:
            Register.MAX_FACES = original

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    reid_emb = Register.get_reid_embedding(frame) if ret else None
    self_sim  = Register.compute_self_sim(features) if len(features) >= 2 else 0.45
    return features, reid_emb, self_sim

# ✅ 수정: 성공 여부를 bool 값으로 반환하도록 변경
async def process_ai_features(email: str, video_path: str) -> bool:
    async with _ai_semaphore:
        try:
            if not os.path.exists(video_path):
                print(f"영상 파일 없음: {video_path}")
                return False
            size = os.path.getsize(video_path)
            if size < 1024:
                print(f"영상 파일 너무 작음: {size}B")
                return False
            print(f"🎬 AI 처리 시작: {email} ({size:,}B)")

            user_row = await database.fetch_one(
                "SELECT name FROM users WHERE email = :e", {"e": email}
            )
            user_name = user_row["name"] if user_row else email

            loop = asyncio.get_event_loop()
            features, reid_emb, self_sim = await loop.run_in_executor(
                _ai_executor, _extract_ai_sync, video_path, MAX_FACES
            )
            print(f"🔍 얼굴 추출: {len(features)}개 | self_sim={self_sim:.3f}")

            if len(features) == 0:
                await database.execute("UPDATE users SET facestat='failed' WHERE email=:e", {"e": email})
                print(f"❌ {email} 얼굴 0개 → facestat='failed'")
                return False

            features_f16  = np.array(features, dtype=np.float32).astype(np.float16).tolist()
            face_data_bin = zlib.compress(pickle.dumps(features_f16))
            reid_data_bin = None
            if reid_emb is not None:
                reid_f16      = np.array(reid_emb, dtype=np.float32).astype(np.float16).tolist()
                reid_data_bin = zlib.compress(pickle.dumps(reid_f16))

            await database.execute("""
                UPDATE users
                SET face_data=:fd, reid_data=:rd, self_sim_lower=:ss, facestat='completed'
                WHERE email=:e
            """, {"fd": face_data_bin, "rd": reid_data_bin, "ss": self_sim, "e": email})

            sync_face_db_json(user_name, features, reid_emb, self_sim)

            raw_size = len(pickle.dumps(features))
            ratio = (1 - len(face_data_bin) / raw_size) * 100 if raw_size > 0 else 0
            print(f"✅ {email}({user_name}) AI 등록 완료 | 압축률 {ratio:.0f}%")
            return True

        except Exception as e:
            traceback.print_exc()
            try:
                await database.execute("UPDATE users SET facestat='failed' WHERE email=:e", {"e": email})
            except Exception:
                pass
            print(f"❌ {email} AI 등록 실패: {e}")
            return False

        finally:
            # --- [추가] 영상 삭제 전 첫 프레임 추출 및 프로필 저장 ---
            if os.path.exists(video_path):
                try:
                    profile_dir = os.path.join("static", "profiles")
                    os.makedirs(profile_dir, exist_ok=True)
                    profile_path = os.path.join(profile_dir, f"{email}.jpg")

                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        cv2.imwrite(profile_path, frame)
                except Exception as img_e:
                    print(f"⚠️ 프로필 이미지 추출 실패: {img_e}")
            # ---------------------------------------------------------

            if os.path.exists(video_path):
                try: os.remove(video_path)
                except: pass

def sync_face_db_json(name: str, features: list, reid_emb, self_sim: float):
    """한 사람 = face_db/{name}.json 파일 1개 (realtime.py 폴더 방식과 동일)"""
    try:
        os.makedirs(FACE_DB_DIR, exist_ok=True)

        use_insightface = getattr(Register, "USE_INSIGHTFACE", True)

        features_list = [f.tolist() if isinstance(f, np.ndarray) else list(f) for f in features]
        reid_list = (reid_emb.tolist() if isinstance(reid_emb, np.ndarray)
                     else list(reid_emb) if reid_emb is not None else None)

        entry = {
            "name": name,
            "features": features_list,
            "outfit_upper": None,
            "reid_embedding": reid_list,
            "processed_files": [],
            "self_sim_lower": self_sim,
            "engine": "insightface" if use_insightface else "deepface",
            "updated_at": datetime.now().isoformat(),
        }

        # 파일명: 공백→_ 치환, 한글 그대로 유지
        safe_name = name.replace(" ", "_")
        file_path = os.path.join(FACE_DB_DIR, f"{safe_name}.json")

        # 원자적 쓰기: 임시 파일 → rename
        fd, tmp = tempfile.mkstemp(suffix=".json", dir=FACE_DB_DIR)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
            os.replace(tmp, file_path)
        except Exception:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
            raise

        print(f"📋 face_db 동기화 완료: {file_path} (얼굴 {len(features_list)}개)")
    except Exception as e:
        print(f"⚠️ face_db 동기화 실패: {e}")

# =============================================
#  구역 설정 저장/불러오기 API (MySQL 연동)
# =============================================

# [구역 관리 - 조회] 저장된 안전구역 목록 불러오기 (지도 초기 로드 시 호출, 공사구역 상세 포함)
@app.get("/api/zones")
async def get_zones(request: Request):
    """DB에서 저장된 안전구역 설정 불러오기 (공사구역은 상세정보 함께 반환)"""
    if 'user_name' not in request.session:
        return JSONResponse(status_code=401, content={"error": "로그인 필요"})
    try:
        rows = await database.fetch_all("SELECT * FROM safety_zones")

        # 공사구역 상세 매핑 (있으면 매칭, 없으면 None)
        cz_map = {}
        try:
            cz_rows = await database.fetch_all("SELECT * FROM construction_zone_details")
            for cz in cz_rows:
                d = dict(cz)
                for k in ("start_date", "end_date"):
                    if d.get(k):
                        d[k] = d[k].strftime("%Y-%m-%d") if hasattr(d[k], "strftime") else str(d[k])
                cz_map[d["zone_name"]] = d
        except Exception as e:
            print(f"⚠️ construction_zone_details 조회 실패(테이블 미생성?): {e}")

        zones = []
        for r in rows:
            path_data = json.loads(r["path_data"])
            coord_count = len(path_data) if isinstance(path_data, list) else 0
            print(f"  📍 로드: {r['label']} ({r['zone_type']}) — 좌표 {coord_count}개")
            z = {
                "type": r["zone_type"],
                "label": r["label"],
                "color": r["color"],
                "path": path_data
            }
            if r["zone_type"] == "CONSTRUCTION":
                z["details"] = cz_map.get(r["label"])  # ✅ 공사구역 상세정보
            zones.append(z)
        print(f"🗺️  구역 로드 완료: {len(zones)}개")
        return {"zones": zones}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# [구역 관리 - 저장] 안전구역 설정 저장 (관리자 전용, 기존 전체 삭제 후 덮어쓰기)
@app.post("/api/zones")
async def save_zones(request: Request):
    """안전구역 설정 DB에 저장 (관리자 전용)"""
    if 'user_name' not in request.session:
        return JSONResponse(status_code=401, content={"error": "로그인 필요"})

    if not request.session.get('is_admin'):
        return JSONResponse(status_code=403, content={"error": "관리자만 구역을 저장할 수 있습니다."})

    try:
        body = await request.json()
        zones = body.get("zones", [])
        user_email = request.session.get("user_email", "unknown")

        for z in zones:
            if not all(k in z for k in ("type", "label", "path")):
                return JSONResponse(status_code=400, content={"error": "구역 데이터 형식 오류"})
            # ✅ 수정됨: CONSTRUCTION 타입 추가 허용
            if z["type"] not in ("DANGER", "ATTENDANCE", "SAFE", "CONSTRUCTION"):
                return JSONResponse(status_code=400, content={"error": f"알 수 없는 구역 타입: {z['type']}"})

        # 기존 구역 데이터를 전부 삭제하고 새 데이터로 덮어쓰기
        await database.execute("DELETE FROM safety_zones")

        # 새 구역 데이터 삽입
        for z in zones:
            path_json = json.dumps(z["path"])
            
            await database.execute(
                """INSERT INTO safety_zones (zone_type, label, color, path_data, saved_by)
                   VALUES (:type, :label, :color, :path, :saved_by)""",
                {
                    "type": z["type"],
                    "label": z["label"],
                    "color": z["color"],
                    "path": path_json,
                    "saved_by": user_email
                }
            )

        print(f"🗺️  구역 설정 DB 저장 완료: {len(zones)}개 by {user_email}")

        # ★ 구역 이름이 바뀌어도 즉시 반영되도록 캐시 갱신
        fresh_rows = await database.fetch_all("SELECT * FROM safety_zones")
        _refresh_zone_cache_sync(fresh_rows)
        print(f"🗺️  구역 캐시 갱신 완료 ({len(_cached_zones)}개)")

        # ★ 공사구역에서 사라진 이름의 상세정보는 자동 정리
        try:
            current_cz_names = [z["label"] for z in zones if z["type"] == "CONSTRUCTION"]
            if current_cz_names:
                placeholders = ",".join([f":n{i}" for i in range(len(current_cz_names))])
                params = {f"n{i}": n for i, n in enumerate(current_cz_names)}
                await database.execute(
                    f"DELETE FROM construction_zone_details WHERE zone_name NOT IN ({placeholders})",
                    params
                )
            else:
                await database.execute("DELETE FROM construction_zone_details")
        except Exception as e:
            print(f"⚠️ 공사구역 상세 정리 실패(테이블 미생성?): {e}")

        return {"success": True, "count": len(zones)}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# =============================================
#  🚧 공사구역 상세정보 (이름 PK, 기준고도, 층간높이, 공사기간)
# =============================================

# [공사구역 - 상세 저장] 공사구역 기준고도·층간높이·공사기간 저장/수정 (관리자 전용)
@app.post("/api/construction_zone")
async def save_construction_zone(request: Request):
    """
    공사구역 상세정보 저장/갱신.
    body: {
      zone_name (PK, 필수),
      base_altitude (기준 고도, 필수 — 1층 바닥의 GPS 고도(m)),
      floor_height (층간 높이 m, NULL 가능),
      start_date / end_date (YYYY-MM-DD, NULL 가능)
    }
    """
    if 'user_name' not in request.session:
        return JSONResponse(status_code=401, content={"error": "로그인 필요"})
    if not request.session.get('is_admin'):
        return JSONResponse(status_code=403, content={"error": "관리자만 설정 가능합니다."})

    try:
        d = await request.json()
        zone_name = (d.get("zone_name") or "").strip()
        if not zone_name:
            return JSONResponse(status_code=400, content={"error": "zone_name 필수"})

        try:
            base_alt = float(d.get("base_altitude"))
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={"error": "base_altitude 숫자 필수"})

        floor_h = d.get("floor_height")
        try:
            floor_h = float(floor_h) if floor_h not in (None, "", "null") else None
        except (TypeError, ValueError):
            floor_h = None

        start_date = d.get("start_date") or None
        end_date   = d.get("end_date")   or None

        # ★ UPSERT (PK: zone_name)
        await database.execute("""
            INSERT INTO construction_zone_details
                (zone_name, base_altitude, floor_height, start_date, end_date)
            VALUES (:n, :b, :f, :s, :e)
            ON DUPLICATE KEY UPDATE
                base_altitude = VALUES(base_altitude),
                floor_height  = VALUES(floor_height),
                start_date    = VALUES(start_date),
                end_date      = VALUES(end_date)
        """, {"n": zone_name, "b": base_alt, "f": floor_h, "s": start_date, "e": end_date})

        print(f"🚧 공사구역 상세 저장: {zone_name} (고도 {base_alt}m, 층간 {floor_h}m)")
        return {"success": True}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# [공사구역 - 단건 조회] 공사구역 상세 1건 조회 (지도에서 구역 클릭 시 팝업용)
@app.get("/api/construction_zone/{zone_name}")
async def get_construction_zone(zone_name: str):
    """공사구역 상세 1건 조회 (구역 클릭 팝업용)"""
    try:
        row = await database.fetch_one(
            "SELECT * FROM construction_zone_details WHERE zone_name = :n",
            {"n": zone_name}
        )
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        d = dict(row)
        for k in ("start_date", "end_date"):
            if d.get(k):
                d[k] = d[k].strftime("%Y-%m-%d") if hasattr(d[k], "strftime") else str(d[k])
        return d
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# [공사구역 - 목록 조회] 전체 공사구역 상세 목록 (지도 로드 시 일괄 캐싱용)
@app.get("/api/construction_zones")
async def list_construction_zones():
    """전체 공사구역 상세 목록 (지도 로드 시 한 일괄 받아서 캐싱)"""
    try:
        rows = await database.fetch_all("SELECT * FROM construction_zone_details")
        result = []
        for r in rows:
            d = dict(r)
            for k in ("start_date", "end_date"):
                if d.get(k):
                    d[k] = d[k].strftime("%Y-%m-%d") if hasattr(d[k], "strftime") else str(d[k])
            result.append(d)
        return {"zones": result}
    except Exception as e:
        print(f"⚠️ construction_zones 조회 실패(테이블 미생성?): {e}")
        return {"zones": []}

# [작업자 상태 - 수동 변경] 관리자 또는 시스템이 작업자 상태(출근중/위험구역 등) 직접 변경
@app.post("/api/update_worker_status")
async def update_worker_status(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        status_text = data.get("status")

        # ✅ 빈 값이면 status 덮어쓰지 않음 (NULL 박히는 것 방지)
        if not email or not status_text:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "email/status 필수"}
            )

        await database.execute(
            "UPDATE users SET status = :s WHERE email = :e",
            {"s": status_text, "e": email}
        )
        return {"success": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ❤️ 메모리에 저장된 현재 심박 상태를 worker_status_list 응답용 필드로 변환
def _hr_fields_for(email: str, status_type: str) -> dict:
    # 미출근/로그아웃 상태면 심박 정보 숨김
    if status_type in ("never_logged_in", "logged_out"):
        return {"hr_bpm": None, "hr_status": None, "hr_label": None, "hr_level": None}
    with _user_hr_lock:
        hr = _user_heart_rates.get(email)
    if not hr:
        return {"hr_bpm": None, "hr_status": None, "hr_label": None, "hr_level": None}
    return {
        "hr_bpm":    hr["bpm"],
        "hr_status": hr["status"],
        "hr_label":  hr["label"],
        "hr_level":  hr["level"],   # normal / warn / danger
    }


# [작업자 현황 목록] 전체 작업자 출근여부·현재구역·심박상태 종합 반환 (index.html 사이드바 갱신용)
@app.get("/api/worker_status_list")
async def get_worker_status():
    try:
        users = await database.fetch_all("SELECT name, phone, email, status FROM users")
        today_attendees = await get_today_attendee_emails()

        # ★ 각 작업자의 최신 위치 좌표를 한 번에 조회 (5분 이내)
        loc_rows = await database.fetch_all("""
            SELECT r.user_id, r.lat, r.lon
            FROM raw_logs r
            INNER JOIN (
                SELECT user_id, MAX(reg_date) AS latest
                FROM raw_logs
                WHERE reg_date > DATE_SUB(NOW(), INTERVAL 5 MINUTE)
                  AND lat IS NOT NULL AND lon IS NOT NULL
                GROUP BY user_id
            ) m ON r.user_id = m.user_id AND r.reg_date = m.latest
            WHERE r.lat IS NOT NULL AND r.lon IS NOT NULL
        """)
        # email → (lat, lon) 매핑
        latest_pos = {r["user_id"]: (float(r["lat"]), float(r["lon"])) for r in loc_rows}

        # ★ 구역 캐시가 비어 있으면 DB에서 로드
        await _ensure_zone_cache()

        result = []
        for r in users:
            u = dict(r)
            email     = u["email"]
            db_status = u.get("status") or ""

            # ──────────────────────────────────────────────
            #  출근 여부 판단
            # ──────────────────────────────────────────────
            if email not in today_attendees:
                result.append({
                    "email":        email,
                    "name":         u["name"],
                    "phone":        u["phone"],
                    "status_type":  "never_logged_in",
                    "status":       "미출근",
                    "current_zone": "",
                    **_hr_fields_for(email, "never_logged_in")
                })
                continue

            if db_status == "미출근":
                result.append({
                    "email":        email,
                    "name":         u["name"],
                    "phone":        u["phone"],
                    "status_type":  "logged_out",
                    "status":       "로그아웃",
                    "current_zone": "",
                    **_hr_fields_for(email, "logged_out")
                })
                continue

            # 긴급 상태는 좌표 계산과 무관하게 우선 반영
            if db_status == "긴급":
                result.append({
                    "email":        email,
                    "name":         u["name"],
                    "phone":        u["phone"],
                    "status_type":  "emergency",
                    "status":       "긴급",
                    "current_zone": "",
                    **_hr_fields_for(email, "emergency")
                })
                continue

            # ──────────────────────────────────────────────
            #  ★ 좌표 기반 구역 판정 (캐시 사용, DB 재조회 없음)
            # ──────────────────────────────────────────────
            pos = latest_pos.get(email)
            current_zone = ""
            status_type  = "online"
            status_text  = "출근중"

            if pos:
                zone = _zone_for_pos(pos[0], pos[1])
                if zone:
                    current_zone = zone["zone_name"]   # ← safety_zones.label 에서 직접 읽음
                    zone_type_map = {
                        "DANGER":       ("danger",       "위험구역"),
                        "CONSTRUCTION": ("construction", "공사구역"),
                        "ATTENDANCE":   ("work_zone",    "근무지"),
                        "SAFE":         ("safe",         "안전구역"),
                    }
                    status_type, status_text = zone_type_map.get(
                        zone["zone_type"], ("online", "출근중")
                    )

            result.append({
                "email":        email,
                "name":         u["name"],
                "phone":        u["phone"],
                "status_type":  status_type,
                "status":       status_text,
                "current_zone": current_zone,
                **_hr_fields_for(email, status_type)
            })

        return {"users": result}
    except Exception as e:
        traceback.print_exc()
        return {"users": [], "error": str(e)}


# =============================================
#  📍 위험구역 진입 감지 (Ray Casting 알고리즘)
#  반직선이 다각형 변과 교차하는 횟수가 홀수 → 내부, 짝수 → 외부
# =============================================
def point_in_polygon(lat: float, lon: float, polygon: list) -> bool:
    """
    Ray Casting 알고리즘으로 점이 다각형 내부에 있는지 판단.
    polygon: [{"lat": ..., "lng": ...}, ...] 형태의 좌표 목록
    /api/send_location, /api/worker_logs, /api/worker_status_list 에서 모두 사용.
    """
    if not polygon or len(polygon) < 3:
        return False

    n = len(polygon)
    inside = False
    j = n - 1

    for i in range(n):
        try:
            xi, yi = float(polygon[i]["lat"]), float(polygon[i]["lng"])
            xj, yj = float(polygon[j]["lat"]), float(polygon[j]["lng"])
        except (KeyError, TypeError, ValueError):
            j = i
            continue

        intersect = ((yi > lon) != (yj > lon)) and \
                    (lat < (xj - xi) * (lon - yi) / ((yj - yi) or 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i

    return inside


# =============================================
#  🔔 [신규] 특정 작업자 실시간 로그/이벤트 조회 API
# =============================================
# [작업자 이벤트 로그] 특정 작업자의 당일 위반기록 + 구역 변화 로그 최신 10건 (index.html 상세 모달용)
@app.get("/api/worker_logs/{email}")
async def get_worker_logs(email: str):
    try:
        # 1. 당일 위반 기록 조회
        query_viol = """
            SELECT 'violation' as type, violation_type as content, detected_at as time 
            FROM violations 
            WHERE name = (SELECT name FROM users WHERE email = :e)
              AND DATE(detected_at) = CURDATE()
        """
        rows_viol = await database.fetch_all(query_viol, {"e": email})
        
        # 2. 당일 위치 기록 조회 (구역 판별을 위해 최근 100개 로그 가져옴)
        query_loc = """
            SELECT 'location' as type, lat, lon, reg_date as time 
            FROM raw_logs 
            WHERE user_id = :e AND lat IS NOT NULL
              AND DATE(reg_date) = CURDATE()
            ORDER BY time DESC LIMIT 100
        """
        rows_loc = await database.fetch_all(query_loc, {"e": email})
        
        # 3. 구역 설정 불러오기
        zones = await database.fetch_all("SELECT * FROM safety_zones")
        parsed_zones = []
        for z in zones:
            try:
                parsed_zones.append({
                    "type": z["zone_type"],
                    "path": json.loads(z["path_data"])
                })
            except Exception:
                pass
                
        priority_map = {"DANGER": 1, "CONSTRUCTION": 2, "ATTENDANCE": 3, "SAFE": 4}
        type_korean = {"DANGER": "위험구역", "CONSTRUCTION": "공사구역", "ATTENDANCE": "출근구역", "SAFE": "안전구역"}
        
        loc_logs = []
        last_zone_type = None
        
        # [수정된 로직] 위치 변화 감지 시 구역명(위험/공사/출근 등)으로 로그 생성
        for r in rows_loc:
            lat, lon = float(r["lat"]), float(r["lon"])
            matched = []
            for z in parsed_zones:
                if point_in_polygon(lat, lon, z["path"]):
                    matched.append({
                        "zone_type": z["type"],
                        "priority": priority_map.get(z["type"], 99)
                    })
            
            current_zone_type = "SAFE"
            if matched:
                matched.sort(key=lambda x: x["priority"])
                current_zone_type = matched[0]["zone_type"]
                
            if current_zone_type != last_zone_type:
                k_type = type_korean.get(current_zone_type, "안전구역")
                loc_logs.append({
                    "type": r["type"],
                    "content": f"위치 갱신 ({k_type})",
                    "time": r["time"]
                })
                last_zone_type = current_zone_type
                
            if len(loc_logs) >= 10:
                break
                
        logs = []
        for r in rows_viol:
            logs.append({"type": r["type"], "content": r["content"], "time": r["time"]})
        logs.extend(loc_logs)
        
        # 최신 시간순 정렬
        logs.sort(key=lambda x: x["time"], reverse=True)
        logs = logs[:10]
        
        result = []
        for log in logs:
            time_str = log["time"].strftime('%H:%M:%S') if log["time"] else ""
            result.append({
                "type": log["type"],
                "content": log["content"],
                "time": time_str
            })
            
        return {"logs": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# =============================================
#  앱 전용 JSON API
# =============================================

# [앱 - 로그인] 이메일+비밀번호 인증 또는 출근구역 진입 체크인, 출근 처리 + 프로필 URL 반환
@app.post("/api/login")
async def api_login(request: Request):
    """
    앱 전용 로그인 + 출근 체크인 처리.

    두 가지 모드:
      1) is_checkin=True : 출근구역 진입 시 비밀번호 없이 출근 상태만 변경
      2) 일반 로그인     : email + password 검증 → 출근 처리 + 프로필 URL 반환
    """
    try:
        data      = await request.json()
        email     = data.get("email")
        is_checkin = data.get("is_checkin", False)
        today = date.today()

        # ── 지도에서 출근구역 진입 시 (비밀번호 없이 상태만 업데이트) ──
        if is_checkin and email:
            u = await database.fetch_one(
                "SELECT name FROM users WHERE email = :e", {"e": email}
            )
            if u:
                await database.execute(
                    "UPDATE users SET status = '출근 중' WHERE email = :e AND status != '출근 중'",
                    {"e": email}
                )
                await database.execute(
                    "INSERT IGNORE INTO attendance (email, access_date) VALUES (:e, :d)",
                    {"e": email, "d": today}
                )
                return {"success": True, "message": "출근 처리 완료", "name": u["name"]}
            else:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "message": "등록되지 않은 사용자입니다."}
                )

        # ── 일반 로그인 (이메일 + 비밀번호) ──
        password = data.get("password")
        u = await database.fetch_one(
            "SELECT * FROM users WHERE email = :e AND password = :p",
            {"e": email, "p": password}
        )
        if u:
            # ✅ 이미 출근 중이면 불필요하게 UPDATE 쿼리를 날리지 않음
            await database.execute(
                "UPDATE users SET status = '출근 중' "
                "WHERE email = :e AND (status IS NULL OR status != '출근 중')",
                {"e": email}
            )
            await database.execute(
                "INSERT IGNORE INTO attendance (email, access_date) VALUES (:e, :d)",
                {"e": email, "d": today}
            )

            # --- [수정] 프로필 이미지 URL 확인 ---
            profile_path = os.path.join("static", "profiles", f"{email}.jpg")
            profile_url = f"/static/profiles/{email}.jpg" if os.path.exists(profile_path) else ""
            # ------------------------------------

            return {"success": True, "name": u["name"], "gender": u["gender"], "profile_url": profile_url}
        else:
            return JSONResponse(
                status_code=401,
                content={"success": False, "message": "이메일 또는 비밀번호가 틀렸습니다."}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"서버 에러: {str(e)}"}
        )


# =============================================
#  📱 앱 로그아웃 API (지도 잔상 제거)
# =============================================
# [앱 - 로그아웃] 상태 미출근 변경 + 메모리 구역/심박 상태 정리 (지도에서 마커 즉시 사라짐)
@app.post("/api/logout")
async def api_logout(request: Request):
    """
    앱에서 로그아웃 시 호출되는 엔드포인트.
    - 사용자 상태를 '미출근'으로 변경 → /api/get_recent_logs 의 status 필터가 자동으로 지도에서 제외함
    - ⚠️ raw_logs DELETE 는 제거됨: 재로그인 직후 모바일 다음 GPS 핑까지 마커가 안 뜨는 문제가 있었음
      (status 필터만으로 지도 잔상은 충분히 제거되고, 재로그인 시 기존 5분 이내 위치가 살아있어 아이콘이 즉시 뜸)
    """
    try:
        data = await request.json()
        email = data.get("email")
        if not email:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "이메일이 필요합니다."}
            )

        # 1. 상태를 '미출근'으로 변경 → 지도 폴링에서 status 필터로 자동 숨김 처리됨
        await database.execute(
            "UPDATE users SET status = '미출근' WHERE email = :e",
            {"e": email}
        )

        # 2. ✅ 메모리상의 구역 상태 정리 (다음 로그인 시 다시 진입 알림이 정상 발생)
        with _user_zone_lock:
            _user_zone_states.pop(email, None)

        # 3. ✅ 메모리상의 심박 상태 정리 (다음 로그인 시 잔상 방지)
        with _user_hr_lock:
            _user_heart_rates.pop(email, None)
            _user_hr_last_logged.pop(email, None)

        print(f"📱 앱 로그아웃: {email}")
        return {"success": True, "message": "로그아웃 처리 완료"}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"서버 에러: {str(e)}"}
        )


# =============================================
#  지도용: 현재 출근 중인 작업자들의 최신 위치
# =============================================
# [지도 - 작업자 위치] 현재 출근 중인 작업자들의 최신 GPS 위치 반환 (지도 마커 표시용, 최근 5분)
@app.get("/api/active_workers")
async def api_active_workers(request: Request):
    if 'user_name' not in request.session:
        return JSONResponse(status_code=401, content={"error": "로그인 필요"})
    try:
        rows = await database.fetch_all("""
            SELECT u.email, u.name, u.status, r.lat, r.lon, r.reg_date
            FROM users u
            INNER JOIN (
                SELECT user_id, lat, lon, reg_date
                FROM raw_logs r1
                WHERE reg_date > DATE_SUB(NOW(), INTERVAL 5 MINUTE)
                  AND lat IS NOT NULL AND lon IS NOT NULL
                  AND reg_date = (
                      SELECT MAX(reg_date)
                      FROM raw_logs r2
                      WHERE r2.user_id = r1.user_id
                        AND r2.lat IS NOT NULL AND r2.lon IS NOT NULL
                        AND r2.reg_date > DATE_SUB(NOW(), INTERVAL 5 MINUTE)
                  )
              ) r ON r.user_id = u.email
            WHERE u.status IS NULL OR u.status != '미출근'
        """)
        workers = []
        for r in rows:
            d = dict(r)
            if d.get('reg_date'):
                d['reg_date'] = d['reg_date'].strftime('%Y-%m-%d %H:%M:%S')
            workers.append(d)
        return {"workers": workers}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


async def _find_current_zone(lat: float, lon: float) -> Optional[dict]:
    """
    현재 위치가 속한 가장 우선순위 높은 구역을 반환.
    ★ 메모리 캐시(_cached_zones)를 사용 → DB 재조회 없음.
       캐시가 비어있으면 _ensure_zone_cache() 로 한 번만 로드.
    반환: {"zone_name": ..., "zone_type": ...} 또는 None
    """
    await _ensure_zone_cache()
    return _zone_for_pos(lat, lon)


def _make_alert(zone: Optional[dict], action: str) -> Optional[dict]:
    """
    위험구역(DANGER) 진입/퇴장 시에만 모바일 알림/진동 메시지 생성.
    그 외 구역(CONSTRUCTION, ATTENDANCE, SAFE)은 None 반환 → 모바일에서 알림/진동 없음.
    """
    if not zone or zone.get("zone_type") != "DANGER":
        return None

    name = zone["zone_name"]
    if action == "enter":
        message = f"⚠️ 위험구역 '{name}'에 진입했습니다. 즉시 안전 조치를 취하세요!"
    else:  # exit
        message = f"✅ 위험구역 '{name}'에서 벗어났습니다."

    return {
        "zone_name": name,
        "zone_type": "danger",
        "action":    action,            # ✅ "enter" 또는 "exit"
        "message":   message
    }


# =============================================
#  🔔 [신규] 안전모 알림 전용 폴링 (딜레이 단축)
#  - 앱이 짧은 주기(3초)로 호출하여 대기 중인 helmet_alert만 회수
#  - send_location 폴링(5~30초 적응형)과 무관하게 즉시 진동 가능
# =============================================
# [앱 - 안전모 알림 폴링] 대기 중인 안전모 미착용 알림 1건 꺼내 반환, 1회 전달 후 큐 비움 (3초 주기 폴링용)
@app.get("/api/poll_alerts")
async def api_poll_alerts(email: str):
    helmet_alert = None
    try:
        now_ts = time.time()
        with _helmet_alert_lock:
            q = _pending_helmet_alerts.get(email)
            if q:
                fresh = [a for a in q if (now_ts - a["ts"]) < HELMET_ALERT_TTL]
                if fresh:
                    helmet_alert = {
                        "type":      fresh[-1]["type"],
                        "message":   fresh[-1]["message"],
                        "image_url": fresh[-1]["image_url"],
                        "count":     len(fresh),
                    }
                _pending_helmet_alerts.pop(email, None)
    except Exception:
        helmet_alert = None
    return {"success": True, "message": "ok", "zone_alert": None, "helmet_alert": helmet_alert}


# [앱 - GPS 위치 전송] 앱이 주기적으로 호출, GPS 저장 + 위험구역 진입/퇴장 감지 + 안전모 알림 함께 전달
@app.post("/api/send_location")
async def api_send_location(request: Request):
    """
    앱에서 주기적으로 GPS 위치를 전송하는 핵심 API.

    처리 순서:
      1) raw_logs 테이블에 GPS 좌표 저장
      2) 현재 구역 판정 (메모리 캐시 사용, DB 재조회 없음)
      3) 이전 구역과 비교 → DANGER 진입/퇴장 시에만 zone_alert 반환
         (동일 구역 머무름 / 비위험 구역 이동은 알림 없음)
      4) users.status 갱신 (위험구역/공사구역/안전 등)
      5) 대기 중인 안전모 알림(helmet_alert) 함께 반환
    위치 저장 완료 → zone_alert: 진입/퇴장 순간에만 dict, 그 외엔 None
    """
    try:
        data  = await request.json()
        email = data.get("email", "unknown")
        lat   = float(data["lat"])
        lon   = float(data["lon"])
        alt   = data.get("alt", None)   # ✅ 고도 (선택, mobile GPS에서 전달)
        try:
            alt_val = float(alt) if alt is not None else None
        except (TypeError, ValueError):
            alt_val = None

        # 위치 저장 (alt 컬럼이 없을 경우 fallback)
        try:
            await database.execute(
                "INSERT INTO raw_logs (user_id, sensor_type, lat, lon, alt) VALUES (:u, :t, :lat, :lon, :a)",
                {"u": email, "t": data.get("sensor_type", "GPS"),
                 "lat": lat, "lon": lon, "a": alt_val}
            )
        except Exception:
            # alt 컬럼이 아직 추가되지 않은 환경 호환
            await database.execute(
                "INSERT INTO raw_logs (user_id, sensor_type, lat, lon) VALUES (:u, :t, :lat, :lon)",
                {"u": email, "t": data.get("sensor_type", "GPS"), "lat": lat, "lon": lon}
            )

        # ✅ 현재 구역 판정
        current = await _find_current_zone(lat, lon)

        # ✅ 이전 구역과 비교 — 진입/퇴장 상태 변화만 알림으로 반환
        with _user_zone_lock:
            prev = _user_zone_states.get(email)
            _user_zone_states[email] = current  # 최신화

        zone_alert = None

        prev_name = prev["zone_name"] if prev else None
        curr_name = current["zone_name"] if current else None
        prev_is_danger = bool(prev and prev.get("zone_type") == "DANGER")
        curr_is_danger = bool(current and current.get("zone_type") == "DANGER")

        # ✅ 위험구역(DANGER)에 대해서만 알림 발생 — 그 외 구역은 알림/진동 없음
        #   1) 위험구역 진입 (새 위험구역에 들어옴, 다른 위험구역에서 옮겨온 경우 포함) → enter 알림 1회
        #   2) 위험구역 퇴장 (다른 zone 으로 옮기든, no-zone 으로 나가든) → exit 알림 1회
        #   3) 동일 위험구역에 머무름 / 그 외 구역 간 이동 → 알림 없음
        if curr_is_danger and prev_name != curr_name:
            # 위험구역에 새로 진입 (다른 위험구역으로 옮겼을 때도 새 진입으로 처리)
            zone_alert = _make_alert(current, "enter")
            print(f"🔔 {email}: '{curr_name}' (DANGER) 진입")
        elif prev_is_danger and not curr_is_danger:
            # 위험구역에서 빠져나옴 (어디로 가든 — 다른 zone 이든 no-zone 이든 1회만)
            zone_alert = _make_alert(prev, "exit")
            print(f"🔕 {email}: '{prev_name}' (DANGER) 퇴장")

        # 사용자 상태(DB)는 매번 최신화
        if current:
            status_map = {
                "DANGER":       "위험구역",
                "CONSTRUCTION": "공사구역",
                "ATTENDANCE":   "출근구역",
                "SAFE":         "안전",
            }
            new_status = status_map.get(current["zone_type"], "안전")
        else:
            new_status = "안전"

        try:
            await database.execute(
                "UPDATE users SET status = :s WHERE email = :e",
                {"s": new_status, "e": email}
            )
        except Exception:
            pass

        # ★★★ [신규] 이 사용자에게 대기 중인 안전모 미착용 알림이 있으면 함께 내려준다 ★★★
        #   앱은 helmet_alert 가 있으면 즉시 진동/알림 → 한 번 내려주면 큐에서 제거(중복 진동 방지)
        helmet_alert = None
        try:
            now_ts = time.time()
            with _helmet_alert_lock:
                q = _pending_helmet_alerts.get(email)
                if q:
                    # TTL 안 지난 것 중 가장 최근 1건을 전달하고 큐 비움
                    fresh = [a for a in q if (now_ts - a["ts"]) < HELMET_ALERT_TTL]
                    if fresh:
                        helmet_alert = {
                            "type":      fresh[-1]["type"],
                            "message":   fresh[-1]["message"],
                            "image_url": fresh[-1]["image_url"],
                            "count":     len(fresh),
                        }
                    _pending_helmet_alerts.pop(email, None)
        except Exception:
            helmet_alert = None

        return {
            "success":      True,
            "message":      "위치 저장 완료",
            "zone_alert":   zone_alert,    # ✅ 진입/퇴장 순간에만 dict, 그 외엔 None
            "helmet_alert": helmet_alert,  # ★ 안전모 미착용 알림 (없으면 None)
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success":    False,
            "message":    f"서버 에러: {str(e)}",
            "zone_alert": None
        })

# =============================================
#  ❤️ [신규] 심박수 수신 API (앱 ApiService.sendHeartRate 대응)
#  - body: {email, bpm, avg_bpm, status}  status ∈ NORMAL/HIGH/TOO_HIGH/TOO_LOW
#  - 현재 심박 상태를 메모리에 저장 → index '현재 상태' 뱃지에 노출
#  - 상태가 바뀔 때만 violations 테이블에 기록 → index '이력' 모달 / 안전 경고 현황 /
#    charts.html 로그에 자동 노출 (image_url='heart_rate' 로 카메라/구역 로그와 구분)
# =============================================
# [앱 - 심박수 수신] 스마트워치 심박 데이터 수신, 상태 변화(정상→이상/회복)시에만 violations 기록
@app.post("/api/send_heart_rate")
async def api_send_heart_rate(request: Request):
    try:
        data   = await request.json()
        email  = data.get("email")
        status = (data.get("status") or "NORMAL").upper()
        try:
            bpm = int(data.get("bpm", 0) or 0)
        except (TypeError, ValueError):
            bpm = 0
        try:
            avg = int(data.get("avg_bpm", bpm) or bpm)
        except (TypeError, ValueError):
            avg = bpm

        if not email:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "email 필수", "zone_alert": None}
            )

        info  = HR_STATUS_MAP.get(status, HR_STATUS_MAP["NORMAL"])
        label = info["label"]
        level = info["level"]   # normal / warn / danger

        # 사용자 이름 조회 (이력은 violations.name 기준으로 매핑됨)
        user = await database.fetch_one(
            "SELECT name FROM users WHERE email = :e", {"e": email}
        )
        name = user["name"] if user else "미식별자"

        now_str = datetime.now().strftime("%H:%M:%S")

        # 1) 현재 심박 상태 메모리 최신화 (index '현재 상태' 표시용)
        with _user_hr_lock:
            _user_heart_rates[email] = {
                "bpm": bpm, "avg": avg, "status": status,
                "label": label, "level": level, "ts": now_str,
            }
            prev_logged = _user_hr_last_logged.get(email)
            changed = (prev_logged != status)
            if changed:
                _user_hr_last_logged[email] = status

        # 2) 상태 변화 시에만 violations 기록
        #    - 이상(주의/위험)으로 바뀌면 항상 기록
        #    - 정상으로 바뀐 경우는 직전이 이상상태였을 때(=회복)만 기록 → 첫 정상값 스팸 방지
        should_log = changed and (
            level != "normal" or (prev_logged not in (None, "NORMAL"))
        )
        if should_log and name != "미식별자":
            vtype = f"{label} ({bpm} bpm)" if level != "normal" else f"심박 정상 회복 ({bpm} bpm)"
            await database.execute(
                """INSERT INTO violations (name, violation_type, image_url, detected_at)
                   VALUES (:n, :t, 'heart_rate', NOW())""",
                {"n": name, "t": vtype}
            )
            print(f"❤️ 심박 기록: {name} {status} {bpm}bpm")

        # 앱은 응답 바디를 사용하지 않지만, SendLocationResponse 모델과 형태를 맞춰 반환
        return {"success": True, "message": "심박수 수신 완료", "zone_alert": None}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False, "message": f"서버 에러: {str(e)}", "zone_alert": None
        })

# =============================================
#  🚨 [신규] 긴급 상황 통합 API
#  앱 ApiService.sendEmergencySos 대응
#  emergency_type: "IMPACT" (낙하·충격감지) | "VOICE_SOS" (음성 구조요청)
#  → violations 테이블에 기록 → index/charts/map 에 실시간 반영
# =============================================
# [앱 - 긴급 SOS] 낙하·충격 감지(IMPACT) 또는 음성 구조 요청(VOICE_SOS) 수신, 상태 '긴급' 변경
@app.post("/api/emergency_sos")
async def api_emergency_sos(request: Request):
    """
    긴급 상황 통합 수신 API.

    emergency_type:
      "IMPACT"    → 낙하·충격 감지 (가속도 센서)
                    ★ 최근 1분 내 안전모 미착용 기록 있으면 무시 (오탐 방지)
      "VOICE_SOS" → 음성 구조 요청 (STT 경유 또는 앱 직접 호출)

    처리:
      violations 테이블 기록 + users.status = '긴급' → 지도 마커 빨간색
    """
    try:
        data           = await request.json()
        email          = data.get("email")
        emergency_type = (data.get("emergency_type") or "IMPACT").upper()
        message        = data.get("message") or ""

        if not email:
            return JSONResponse(status_code=400,
                content={"success": False, "message": "email 필수", "zone_alert": None})

        # 사용자 이름 조회
        user = await database.fetch_one(
            "SELECT name FROM users WHERE email = :e", {"e": email}
        )
        name = user["name"] if user else "미식별자"

        # -----------------------------------------------------------------
        # 🛠️ [수정] IMPACT(충격) 감지 시, 안전모 미착용 상태인지 교차 검증
        # -----------------------------------------------------------------
        if emergency_type == "IMPACT":
            # 최근 1분 이내에 이 작업자가 안전모 미착용으로 걸린 기록이 있는지 확인
            recent_no_helmet = await database.fetch_one("""
                SELECT id FROM violations 
                WHERE name = :n 
                  AND violation_type LIKE '%안전모 미착용%' 
                  AND detected_at > DATE_SUB(NOW(), INTERVAL 1 MINUTE)
            """, {"n": name})

            if recent_no_helmet:
                print(f"🛑 충격 감지 무시됨 (안전모 미착용 상태에서 발생): {name}")
                return {"success": True, "message": "안전모 미착용 상태이므로 충격 무시됨", "zone_alert": None}
            
            vtype     = f"낙하·충격 감지 ({message})" if message else "낙하·충격 감지"
            image_url = "emergency_impact"
        elif emergency_type == "VOICE_SOS":
            vtype      = f"음성 구조 요청 ({message})" if message else "음성 구조 요청"
            image_url  = "emergency_sos"
        else:
            vtype      = f"긴급 상황 ({emergency_type}): {message}"
            image_url  = "emergency_sos"

        # violations 테이블에 기록 → 기존 /api/get_violations 폴링으로 자동 노출
        await database.execute(
            """INSERT INTO violations (name, violation_type, image_url, detected_at)
               VALUES (:n, :t, :url, NOW())""",
            {"n": name, "t": vtype, "url": image_url}
        )

        # 작업자 상태를 "긴급" 로 변경 → map.html 마커 아이콘 색상 변경 트리거
        await database.execute(
            "UPDATE users SET status = '긴급' WHERE email = :e",
            {"e": email}
        )

        print(f"🚨 긴급 수신: {name} ({email}) [{emergency_type}] {message}")
        return {"success": True, "message": "긴급 상황 수신 완료", "zone_alert": None}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500,
            content={"success": False, "message": f"서버 에러: {str(e)}", "zone_alert": None})


# =============================================
#  기존 파일 내용 이어서 작성 (누락 없음)
# =============================================
# [앱 - 음성 SOS 감지] 마이크 WAV 파일 수신 → Google STT 변환 → SOS 키워드 포함 시에만 violations 기록
@app.post("/api/stt_alert")
async def api_stt_alert(
    email: str = Form(...),
    lat:   str = Form(...),
    lon:   str = Form(...),
    audio: UploadFile = File(...)
):
    """
    ★ [수정] 실제 STT 수행 → SOS 키워드 검출 시에만 is_sos=True 반환.
       이전 버전은 무조건 is_sos=True 였어서 마이크에 잡힌 모든 소리가
       가짜 SOS로 기록되는 버그가 있었음.
    """
    try:
        wav_bytes = await audio.read()

        # 1) 실제 음성 인식 (Google Web Speech, 무료, 키 불필요)
        text = _recognize_korean(wav_bytes)
        is_sos = any(k in text for k in SOS_KEYWORDS) if text else False
        print(f"🎙️ STT 결과: '{text}' → is_sos={is_sos}")

        # 2) SOS로 판정된 경우에만 위반 기록 + 위치 저장
        if is_sos:
            user = await database.fetch_one(
                "SELECT name FROM users WHERE email = :e", {"e": email}
            )
            name = user["name"] if user else "미식별자"
            await database.execute(
                """INSERT INTO violations (name, violation_type, image_url, detected_at)
                   VALUES (:n, :t, 'emergency_sos', NOW())""",
                {"n": name, "t": f"음성 구조 요청: \"{text}\""}
            )
            try:
                lat_f, lon_f = float(lat), float(lon)
                await database.execute(
                    "INSERT INTO raw_logs (user_id, sensor_type, lat, lon) VALUES (:u, 'GPS', :lat, :lon)",
                    {"u": email, "lat": lat_f, "lon": lon_f}
                )
            except Exception:
                pass

        # 3) 앱에는 실제 인식 결과만 반환 (앱은 is_sos가 true일 때만 SOS 동작)
        return {
            "success": True,
            "is_sos":  is_sos,
            "message": text or "음성 인식 실패"
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500,
            content={"success": False, "is_sos": False, "message": f"서버 에러: {str(e)}"})


# [긴급 이벤트 목록] 오늘 발생한 낙하충격·음성SOS 이벤트 최근 50건 조회 (charts.html 긴급 현황용)
@app.get("/api/emergency_events")
async def api_emergency_events():
    try:
        rows = await database.fetch_all(
            """SELECT name, violation_type, detected_at
               FROM violations
               WHERE image_url IN ('emergency_impact','emergency_sos')
                 AND DATE(detected_at) = CURDATE()
               ORDER BY detected_at DESC LIMIT 50"""
        )
        events = []
        for r in rows:
            d = dict(r)
            d["detected_at"] = r["detected_at"].strftime("%Y-%m-%d %H:%M:%S") if r["detected_at"] else ""
            events.append(d)
        return {"events": events}
    except Exception as e:
        return {"events": [], "error": str(e)}


# [AI 위반 수신] realtime.py AI가 감지한 안전모 미착용 등 위반을 DB에 기록 + 해당 작업자 앱에 알림 적재
@app.post("/api/violation")
async def api_receive_violation(request: Request):
    """
    realtime.py AI가 감지한 위반(안전모 미착용 등)을 수신.

    처리:
      1) violations 테이블 기록
      2) 이름이 이메일 형식이면 users 테이블에서 실제 이름으로 자동 치환
      3) 식별된 작업자면 _pending_helmet_alerts 큐에 알림 적재
         → 앱 다음 위치 전송(/api/send_location) 시 helmet_alert 로 전달
      4) 앱 알림 전송 로그도 violations 에 기록 (image_url='app_notified')
    """
    try:
        data = await request.json()
        name = data.get("name", "미식별자")
        vtype = data.get("type", "안전모 미착용")
        image_url = data.get("image_url", "")

        # ★ 이름이 비어있거나 'unknown' 으로 들어오면 '미식별자'로 일관 기록
        if not name or str(name).strip() == "" or name == "unknown":
            name = "미식별자"

        # ★ [신규] 이메일이 잘못 들어온 경우 → users 테이블에서 실제 이름으로 자동 치환
        if name and "@" in str(name):
            try:
                u_by_email = await database.fetch_one(
                    "SELECT name FROM users WHERE email = :e", {"e": name}
                )
                if u_by_email and u_by_email["name"]:
                    print(f"  🔁 email '{name}' → name '{u_by_email['name']}' 자동 치환")
                    name = u_by_email["name"]
            except Exception:
                pass

        await database.execute(
            """INSERT INTO violations (name, violation_type, image_url, detected_at)
               VALUES (:name, :type, :url, NOW())""",
            {
                "name": name,
                "type": vtype,
                "url": image_url
            }
        )
        print(f"⚠️ 위반 수신: {name} - {vtype}")

        # ═══════════════════════════════════════════════════════════
        # ★★★ [신규] 식별된 작업자면 본인 앱 알림 큐에 적재 (폴링) ★★★
        #   - 미식별자/이름없음 → 앱 매핑 불가 → 알림 생략(기록만)
        #   - 이름 → email 조회 → _pending_helmet_alerts 적재
        #   - 앱이 다음 위치 전송 시 응답으로 받아 즉시 진동/알림
        # ═══════════════════════════════════════════════════════════
        if name and name not in ("미식별자", "unknown") and not str(name).startswith("미식별자"):
            try:
                # ★ [수정] TRIM 후 정확 매칭 → 실패 시 LIKE 부분 매칭 (공백/한자 변형 대응)
                user = await database.fetch_one(
                    "SELECT email, name FROM users WHERE TRIM(name) = TRIM(:n) LIMIT 1",
                    {"n": name}
                )
                if not user:
                    user = await database.fetch_one(
                        "SELECT email, name FROM users WHERE name LIKE :n LIMIT 1",
                        {"n": f"%{str(name).strip()}%"}
                    )
                if not user:
                    print(f"  ⚠️ name='{name}' → users 매핑 실패. users.name 컬럼 확인 필요.")
                if user and user["email"]:
                    email = user["email"]
                    now_ts = time.time()
                    alert_obj = {
                        "type": vtype,
                        "image_url": image_url,
                        "message": f"⚠️ '{vtype}' 감지됨. 즉시 안전 조치를 취하세요!",
                        "ts": now_ts,
                    }
                    with _helmet_alert_lock:
                        q = _pending_helmet_alerts.setdefault(email, [])
                        q.append(alert_obj)
                        # TTL 지난 알림 정리 + 최근 5건만 유지
                        q[:] = [a for a in q if (now_ts - a["ts"]) < HELMET_ALERT_TTL][-5:]
                    print(f"  🔔 앱 알림 적재: {name} → {email} ({vtype})")

                    # ★★★ [신규] 앱 알림 전송 로그를 violations 테이블에 기록 ★★★
                    #   - image_url = "app_notified" 로 구분
                    #   - HTML에서 📱 배지로 표시
                    try:
                        await database.execute(
                            """INSERT INTO violations (name, violation_type, image_url, detected_at)
                               VALUES (:name, :type, 'app_notified', NOW())""",
                            {
                                "name": name,
                                "type": f"📱 앱 알림 전송 — {vtype}",
                            }
                        )
                        print(f"  📝 앱 알림 로그 기록: {name} ({vtype})")
                    except Exception as log_e:
                        print(f"  ⚠️ 앱 알림 로그 기록 실패(무시): {log_e}")
                else:
                    print(f"  ℹ️ 앱 알림 생략: '{name}' 의 email 매핑 없음")
            except Exception as e:
                print(f"  ⚠️ 앱 알림 처리 중 오류(기록은 정상): {e}")

        return {"success": True, "message": "위반 기록 저장 완료"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"서버 에러: {str(e)}"})


# ═══════════════════════════════════════════════════════════════════
# ★★★ [신규] /api/void_violation ★★★
#   - realtime.py 가 미식별자로 잘못 로그한 위반을 차후 진짜 식별 후 호출
#   - 같은 트랙이 처음에는 "미식별자#3" 으로 등록됐다가 잠시 후 "최승혁" 으로 확정되는 경우
#     → 미식별자#3 위반은 voiding 처리 (DB 에서 삭제)
#   - logged_at_unix 시각 전후 NOTIFY_COOLDOWN(30s) 내에 등록된 미식별자 위반만 대상
#   - 안전을 위해 'name LIKE 미식별자%' 이고 'violation_type = 안전모 미착용' 인 것만 삭제
#   - 앱 알림 로그(image_url = 'app_notified')는 대상에서 제외 (이미 알림 보냈으므로)
# ═══════════════════════════════════════════════════════════════════
# [위반 정정] 미식별자로 잘못 기록된 위반을 실제 이름 확정 후 DB에서 삭제 (realtime.py → main.py 정정 요청)
@app.post("/api/void_violation")
async def api_void_violation(request: Request):
    """
    미식별자 위반 정정 API.

    realtime.py가 처음에 '미식별자'로 기록했다가
    나중에 실제 이름을 확정하면 이 API로 잘못된 기록 삭제 요청.
    ±60초 시간 윈도우 내의 '안전모 미착용' 미식별자 기록만 삭제 대상.
    app_notified 기록은 삭제 제외 (이미 앱에 알림 보냈으므로).
    """
    try:
        data = await request.json()
        logged_at_unix = float(data.get("logged_at_unix") or 0.0)
        replaced_by = (data.get("replaced_by") or "").strip()

        if logged_at_unix <= 0:
            return {"success": False, "message": "logged_at_unix 누락"}

        # 시간 윈도우 ±60초 (realtime.py 의 NOTIFY_COOLDOWN=30s 보다 약간 넓게)
        from datetime import datetime, timedelta
        center = datetime.fromtimestamp(logged_at_unix)
        t_start = center - timedelta(seconds=60)
        t_end   = center + timedelta(seconds=60)

        # 1) 안전모 미착용 + 미식별자% + 이미지 첨부된 것 (원본 위반) 만 삭제
        rows = await database.fetch_all(
            """SELECT id, name, violation_type, image_url, detected_at
                 FROM violations
                WHERE name LIKE '미식별자%'
                  AND violation_type = '안전모 미착용'
                  AND (image_url IS NULL OR image_url <> 'app_notified')
                  AND detected_at BETWEEN :ts AND :te
                ORDER BY detected_at DESC""",
            {"ts": t_start, "te": t_end}
        )
        if not rows:
            return {"success": True, "voided": 0, "message": "대상 없음"}

        ids_to_delete = [r["id"] for r in rows]
        # IN 절로 일괄 삭제 (databases 라이브러리 호환을 위해 직접 SQL 구성)
        placeholders = ", ".join([f":id{i}" for i in range(len(ids_to_delete))])
        params = {f"id{i}": v for i, v in enumerate(ids_to_delete)}
        await database.execute(
            f"DELETE FROM violations WHERE id IN ({placeholders})",
            params
        )
        print(f"  🧹 void_violation: {len(ids_to_delete)}건 삭제 "
              f"(replaced_by={replaced_by}, window={t_start}~{t_end})", flush=True)
        return {"success": True, "voided": len(ids_to_delete), "ids": ids_to_delete}
    except Exception as e:
        print(f"  ⚠️ /api/void_violation 실패: {e}", flush=True)
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


# [위반 목록 조회] 오늘 발생한 위반 전체 최대 300건 조회 (charts.html / live.html 폴링용)
@app.get("/api/get_violations")
async def api_get_violations():
    try:
        query = "SELECT * FROM violations WHERE DATE(detected_at) = CURDATE() ORDER BY detected_at DESC LIMIT 300"
        rows = await database.fetch_all(query)
        
        return {"violations": [
            {**dict(r), "detected_at": r['detected_at'].strftime('%Y-%m-%d %H:%M:%S') if r['detected_at'] else ""}
            for r in rows
        ]}
    except Exception as e:
        return {"violations": [], "error": str(e)}


# [위반 내역 CSV 다운로드] 전체 위반 이력을 CSV 파일로 다운로드 (Excel 한글 깨짐 방지 BOM 포함)
@app.get("/api/download_violations")
async def download_violations(request: Request):
    """
    violations 테이블 전체를 CSV 파일로 다운로드.
    BOM(\ufeff) 추가 → Excel에서 한글 깨짐 방지.
    StreamingResponse로 반환 → 대용량도 메모리 과부하 없음.
    """
    if 'user_name' not in request.session:
        return RedirectResponse(url="/login")

    try:
        rows = await database.fetch_all("SELECT name, violation_type, detected_at FROM violations ORDER BY detected_at DESC")
        stream = _io.StringIO()
        writer = csv.writer(stream)
        stream.write('\ufeff') 
        writer.writerow(["근로자 이름/식별자", "위반 유형", "발생 일시"])
        for r in rows:
            detected_time = r['detected_at'].strftime('%Y-%m-%d %H:%M:%S') if r['detected_at'] else ""
            writer.writerow([r['name'], r['violation_type'], detected_time])
        
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=violation_records.csv"
        return response
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# [IoT 센서 데이터 수신] 아두이노·라즈베리파이 등 외부 장치에서 GPS 또는 센서값 직접 전송 수신
@app.post("/api/data")
async def receive_api(request: Request):
    """
    IoT 디바이스(아두이노, 라즈베리파이 등) 범용 데이터 수신 엔드포인트.
    device_id 로 사용자 식별 (앱 email 대신 장치 ID 사용).
      - lat/lon 있으면 → raw_logs GPS 기록
      - value 있으면   → raw_logs 센서값 기록 (온도, 진동 등)
    """
    try:
        data = await request.json()
        u_id = data.get('device_id') or 'unknown'
        if 'lat' in data and 'lon' in data:
            await database.execute("INSERT INTO raw_logs (user_id, sensor_type, lat, lon) VALUES (:u, 'GPS', :lat, :lon)", 
                                {"u": u_id, "lat": data['lat'], "lon": data['lon']})
        elif 'value' in data:
            await database.execute("INSERT INTO raw_logs (user_id, sensor_type, data_value) VALUES (:u, :type, :v)", 
                                {"u": u_id, "type": u_id, "v": data['value']})
        return {"result": "success"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": "error", "message": str(e)})

# [지도 - GPS 로그] 최근 5분 이내 GPS 로그 조회 (map.html 작업자 마커 갱신용, 미출근 제외)
@app.get("/api/get_recent_logs")
async def get_recent_logs():
    """
    지도(map.html)에서 주기적으로 폴링하는 작업자 위치 로그.
    최근 5분 이내 GPS 기록만 반환 (미출근 상태 제외).
    alt 컬럼 미존재 환경 호환을 위해 3단계 fallback 쿼리 구조.
    """
    try:
        rows = None
        try:
            rows = await database.fetch_all("""
                SELECT 
                    r.user_id, r.lat, r.lon, r.alt, r.reg_date, 
                    COALESCE(u.name, '미등록 사용자') as name, 
                    COALESCE(u.phone, '') as phone,
                    u.birthdate,
                    COALESCE(u.gender, '') as gender,
                    u.status
                FROM raw_logs r
                LEFT JOIN users u ON u.email = r.user_id
                WHERE r.reg_date > DATE_SUB(NOW(), INTERVAL 5 MINUTE)
                  AND r.lat IS NOT NULL 
                  AND r.lon IS NOT NULL
                  AND (u.status IS NULL OR u.status != '미출근')
                ORDER BY r.reg_date DESC
                LIMIT 50
            """)
        except Exception:
            pass

        if rows is None:
            try:
                rows = await database.fetch_all("""
                    SELECT 
                        r.user_id, r.lat, r.lon, NULL as alt, r.reg_date, 
                        COALESCE(u.name, '미등록 사용자') as name, 
                        COALESCE(u.phone, '') as phone,
                        u.birthdate,
                        COALESCE(u.gender, '') as gender,
                        u.status
                    FROM raw_logs r
                    LEFT JOIN users u ON u.email = r.user_id
                    WHERE r.reg_date > DATE_SUB(NOW(), INTERVAL 5 MINUTE)
                      AND r.lat IS NOT NULL 
                      AND r.lon IS NOT NULL
                      AND (u.status IS NULL OR u.status != '미출근')
                ORDER BY r.reg_date DESC
                LIMIT 50
            """)
            except Exception:
                pass

        if rows is None:
            rows = await database.fetch_all("""
                SELECT 
                    r.user_id, r.lat, r.lon, NULL as alt, r.reg_date, 
                    COALESCE(u.name, '미등록 사용자') as name, 
                    '' as phone,
                    NULL as birthdate,
                    '' as gender,
                    u.status
                FROM raw_logs r
                LEFT JOIN users u ON u.email = r.user_id
                WHERE r.reg_date > DATE_SUB(NOW(), INTERVAL 5 MINUTE)
                  AND r.lat IS NOT NULL 
                  AND r.lon IS NOT NULL
                  AND (u.status IS NULL OR u.status != '미출근')
                ORDER BY r.reg_date DESC
                LIMIT 50
            """)

        def _serialize(r):
            d = dict(r)
            d["reg_date"] = r['reg_date'].strftime('%Y-%m-%d %H:%M:%S') if r['reg_date'] else ""
            bd = d.get("birthdate")
            if bd:
                d["birthdate"] = bd.strftime("%Y-%m-%d") if hasattr(bd, "strftime") else str(bd)
            else:
                d["birthdate"] = ""
            return d

        return {"logs": [_serialize(r) for r in rows]}
    except Exception as e:
        return {"logs": [], "error": str(e)}

# [작업자 상세] 가입일 + 당월 출근 날짜 목록 반환 (index.html 작업자 카드 출근 캘린더용)
@app.get("/api/worker_detail/{email}")
async def api_worker_detail(email: str):
    try:
        user = await database.fetch_one(
            "SELECT created_at FROM users WHERE email = :e", {"e": email}
        )

        logs = await database.fetch_all(
            """SELECT access_date FROM attendance
               WHERE email = :e
                 AND YEAR(access_date)  = YEAR(CURDATE())
                 AND MONTH(access_date) = MONTH(CURDATE())""",
            {"e": email}
        )

        today = date.today()
        days_in_month = calendar.monthrange(today.year, today.month)[1]

        return {
            "created_at": user['created_at'].strftime('%Y-%m-%d')
                          if user and user['created_at'] else "정보 없음",
            "attendance_days": [log['access_date'].strftime('%d') for log in logs],
            "year":  today.year,
            "month": today.month,
            "days_in_month": days_in_month,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# =============================================
#  웹 페이지 라우트
# =============================================

# [루트] 접속 시 /login 으로 리다이렉트
@app.get("/")
async def root():
    return RedirectResponse(url="/login")

# [웹 - 로그인] GET: 로그인 폼 / POST: 이메일+비밀번호 검증 → 세션 저장 → /index 이동
@app.api_route("/login", methods=["GET", "POST"], name="login")
async def login(request: Request):
    """
    웹 로그인.
    GET  → login.html 폼 표시
    POST → email+password 검증 → 세션 저장 → /index 이동
           성공 시 users.status='출근 중' + attendance 기록
    """
    if request.method == "POST":
        f = await request.form()
        email = f.get("email")
        password = f.get("password")

        u = await database.fetch_one(
            "SELECT * FROM users WHERE email = :e AND password = :p",
            {"e": email, "p": password}
        )

        if u:
            request.session['user_name'] = u['name']
            request.session['user_email'] = u['email']
            # isAdmin / is_admin 컬럼명 양쪽 호환 처리
            try:
                request.session['is_admin'] = bool(u['isAdmin'])
            except Exception:
                try:
                    request.session['is_admin'] = bool(u['is_admin'])
                except Exception:
                    request.session['is_admin'] = False
            
            today = date.today()
            await database.execute("UPDATE users SET status = '출근 중' WHERE email = :e", {"e": u['email']})
            await database.execute(
                "INSERT IGNORE INTO attendance (email, access_date) VALUES (:e, :d)",
                {"e": u['email'], "d": today}
            )
            
            # ★ POST 후 리다이렉트는 반드시 303(See Other) 사용
            # 302를 쓰면 브라우저가 POST 그대로 /index 를 재요청 → 로그인 후 이동 안 됨
            return RedirectResponse(url="/index", status_code=303)

        return templates.TemplateResponse(
            request=request, 
            name="login.html", 
            context={"error": "이메일 또는 비밀번호가 틀렸습니다."}
        )

    return templates.TemplateResponse(request=request, name="login.html", context={"error": None})

# [메인 대시보드] 출근/미출근 인원 현황·작업자 목록·날씨·오늘의 안전 법령 표시
@app.get("/index", name="index")
async def index(request: Request):
    if 'user_name' not in request.session: 
        return RedirectResponse(url="/login")
    
    today_attendees = await get_today_attendee_emails()

    total = await database.fetch_val("SELECT COUNT(*) FROM users") or 0
    on    = len(today_attendees)
    off   = total - on
    
    rows = await database.fetch_all("SELECT name, phone, email, status FROM users LIMIT 10")
    users_list = []

    loc_rows_idx = await database.fetch_all("""
        SELECT r.user_id, r.lat, r.lon
        FROM raw_logs r
        INNER JOIN (
            SELECT user_id, MAX(reg_date) AS latest
            FROM raw_logs
            WHERE reg_date > DATE_SUB(NOW(), INTERVAL 5 MINUTE)
              AND lat IS NOT NULL AND lon IS NOT NULL
            GROUP BY user_id
        ) m ON r.user_id = m.user_id AND r.reg_date = m.latest
        WHERE r.lat IS NOT NULL AND r.lon IS NOT NULL
    """)
    latest_pos_idx = {r["user_id"]: (float(r["lat"]), float(r["lon"])) for r in loc_rows_idx}

    await _ensure_zone_cache()

    for r in rows:
        u = dict(r)
        email     = u['email']
        db_status = u.get("status") or ""

        if email not in today_attendees:
            u['status_type'] = "never_logged_in"
            u['status']      = "미출근"
            u['current_zone'] = ""
        elif db_status == "미출근":
            u['status_type'] = "logged_out"
            u['status']      = "로그아웃"
            u['current_zone'] = ""
        elif db_status == "긴급":
            u['status_type'] = "emergency"
            u['status']      = "긴급"
            u['current_zone'] = ""
        else:
            pos = latest_pos_idx.get(email)
            current_zone = ""
            status_type  = "online"
            u['status']  = "출근중"

            if pos:
                zone = _zone_for_pos(pos[0], pos[1])
                if zone:
                    current_zone = zone["zone_name"]
                    zone_type_map = {
                        "DANGER":       ("danger",       "위험구역"),
                        "CONSTRUCTION": ("construction", "공사구역"),
                        "ATTENDANCE":   ("work_zone",    "근무지"),
                        "SAFE":         ("safe",         "안전구역"),
                    }
                    st, sv = zone_type_map.get(zone["zone_type"], ("online", "출근중"))
                    status_type  = st
                    u['status']  = sv

            u['status_type']  = status_type
            u['current_zone'] = current_zone

        hr = _hr_fields_for(email, u['status_type'])
        u['hr_bpm']    = hr.get('hr_bpm')
        u['hr_status'] = hr.get('hr_status')
        u['hr_label']  = hr.get('hr_label')
        u['hr_level']  = hr.get('hr_level')

        users_list.append(u)
    
    weather_info = await get_weather("Suwon")
    
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={
            "user_name": request.session['user_name'], 
            "on_duty": on, 
            "off_duty": off, 
            "today_law": random.choice(safety_laws),
            "weather": weather_info,
            "users": users_list
        }
    )


# [회원가입] GET: 가입 폼 / POST: DB 등록 + 영상 업로드 시 AI 얼굴 임베딩 추출 → face_db 저장
@app.api_route("/register", methods=["GET", "POST"], name="register")
async def register(
    request: Request,
    name: str = Form(None),
    email: str = Form(None),
    password: str = Form(None),
    phone: str = Form(None),
    birthdate: str = Form(None),
    gender: str = Form(None),
    video: UploadFile = File(None)
):
    """
    웹/앱 공통 회원가입 + AI 얼굴 등록.

    처리:
      1) users 테이블 DB 등록
      2) 영상 첨부 시: 임시 저장 → process_ai_features() 비동기 호출
         → 얼굴 임베딩 추출 → users.face_data 업데이트 + face_db/{이름}.json 생성
         → 첫 프레임 → static/profiles/{email}.jpg 프로필 이미지 저장
      3) AI 실패 시 DB 롤백(DELETE)
    웹: 완료 후 /login 리다이렉트
    앱: JSON {"success": true} 반환
    """
    if request.method == "POST":
        try:
            is_app = video is not None or "application/json" in request.headers.get("Accept", "")

            if not name or not email or not password:
                error = "이름, 이메일, 비밀번호는 필수 입력 항목입니다."
                if is_app: return JSONResponse(status_code=400, content={"success": False, "message": error})
                return templates.TemplateResponse(request=request, name="register.html", context={"error": error})

            query = """
                INSERT INTO users (name, email, password, phone, birthdate, gender, status, facestat) 
                VALUES (:n, :e, :p, :ph, :b, :g, '미출근', :fs)
            """
            await database.execute(query, {
                "n": name, "e": email, "p": password, 
                "ph": phone, "b": birthdate, "g": gender,
                "fs": "pending" if video else "none"
            })
            
            print(f"✅ 회원가입 1차 DB 등록 완료 (ID: {email})")

            if video and video.filename:
                valid_exts = ('.mp4', '.avi', '.mov')
                if not video.filename.lower().endswith(valid_exts):
                    await database.execute("DELETE FROM users WHERE email = :e", {"e": email})
                    error = "올바른 영상 파일(.mp4, .avi 등) 형식이 아닙니다."
                    if is_app: return JSONResponse(status_code=400, content={"success": False, "message": error})
                    return templates.TemplateResponse(request=request, name="register.html", context={"error": error})

                video_path = os.path.join(UPLOAD_DIR, f"{email}_{int(datetime.now().timestamp())}.mp4")
                content = await video.read()
                with open(video_path, "wb") as f:
                    f.write(content)
                print(f"📹 영상 임시 저장: {video_path} ({len(content):,} bytes)")

                is_success = await process_ai_features(email, video_path)
                
                if not is_success:
                    await database.execute("DELETE FROM users WHERE email = :e", {"e": email})
                    error = "얼굴 인식 및 데이터 처리에 실패했습니다. 선명한 영상을 다시 올려주세요."
                    if is_app: return JSONResponse(status_code=400, content={"success": False, "message": error})
                    return templates.TemplateResponse(request=request, name="register.html", context={"error": error})
                    
            if is_app:
                return {"success": True, "message": "가입을 환영합니다! 얼굴 등록이 완료되었습니다."}
            
            return RedirectResponse(url="/login", status_code=303)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ 서버 내부 에러: {error_msg}")
            if is_app: return JSONResponse(status_code=500, content={"success": False, "message": f"서버 에러: {error_msg}"})
            error = "이미 등록된 이메일이거나 서버 오류입니다."
            return templates.TemplateResponse(request=request, name="register.html", context={"error": error})
    
    return templates.TemplateResponse(request=request, name="register.html", context={"error": None})

# [비밀번호 찾기] 이메일 존재 확인 후 비밀번호 재설정 페이지로 이동
@app.api_route("/forgot_password", methods=["GET", "POST"])
async def forgot_password(request: Request):
    error = None
    if request.method == "POST":
        e = (await request.form()).get("email")
        if await database.fetch_one("SELECT * FROM users WHERE email = :e", {"e": e}): 
            return RedirectResponse(url=f"/reset_password/{e}", status_code=302)
        error = "등록되지 않은 이메일입니다."
    return templates.TemplateResponse(request=request, name="forgot_password.html", context={"error": error})

# [비밀번호 재설정] GET: 새 비밀번호 입력 폼 / POST: 비밀번호 변경 후 /login 이동
@app.api_route("/reset_password/{email}", methods=["GET", "POST"])
async def reset_password(request: Request, email: str):
    if request.method == "POST":
        pw = (await request.form()).get("new_password")
        await database.execute("UPDATE users SET password = :p WHERE email = :e", {"p": pw, "e": email})
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request=request, name="reset_password.html", context={"email": email})

# [지도 페이지] 실시간 작업자 위치 + 안전구역 표시 지도 (관리자 여부에 따라 구역 편집 기능 활성화)
@app.get("/map", name="map")
async def map_page(request: Request):
    if 'user_name' not in request.session: return RedirectResponse(url="/login")
    return templates.TemplateResponse(
        request=request, 
        name="map.html",
        context={
            "user_name": request.session.get('user_name', ''),
            "is_admin": bool(request.session.get('is_admin', False)), 
        }
    )

# [작업자 목록 테이블] 전체 작업자 정보(나이·상태·출근여부) 테이블 형태로 표시
@app.get("/tables", name="tables")
async def tables(request: Request):
    if 'user_name' not in request.session: return RedirectResponse(url="/login")

    today_attendees = await get_today_attendee_emails()
    rows = await database.fetch_all("SELECT * FROM users")
    users = [dict(r) for r in rows]
    curr_year = 2026
    
    for u in users:
        b = str(u.get('birthdate', ''))
        if b and len(b) >= 4:
            try: u['age'] = curr_year - int(b[:4]) + 1
            except: u['age'] = "미정"
        else: u['age'] = "미정"

        email = u['email']
        db_status = u.get("status") or ""

        if email not in today_attendees:
            u['status_type'] = "never_logged_in"
            u['status'] = "미출근"
        else:
            if db_status == "미출근":
                u['status_type'] = "logged_out"
                u['status'] = "로그아웃"
            elif "위험" in db_status:
                u['status_type'] = "danger"
                u['status'] = "위험구역"
            elif "공사" in db_status:                                   
                u['status_type'] = "construction"
                u['status'] = "공사구역"
            elif "안전" in db_status:
                u['status_type'] = "safe"
                u['status'] = "안전구역"
            elif "출근구역" in db_status or "근무지" in db_status:        
                u['status_type'] = "work_zone"
                u['status'] = "근무지"
            else:
                u['status_type'] = "online"
                u['status'] = "출근중"

    return templates.TemplateResponse(request=request, name="tables.html", context={"users": users, "user_name": request.session['user_name']})

# [통계 차트] 위반 유형별·시간대별 통계, 심박 이상·긴급 이벤트 현황 차트 페이지
@app.get("/charts", name="charts")
async def charts(request: Request):
    if 'user_name' not in request.session: return RedirectResponse(url="/login")
    return templates.TemplateResponse(
        request=request, 
        name="charts.html", 
        context={
            "user_name": request.session['user_name']
        }
    )

# [웹 - 로그아웃] 세션 삭제 + 상태 미출근 변경 + 메모리 심박/구역 상태 정리 → /login 이동
@app.get("/logout", name="logout")
async def logout(request: Request):
    """
    웹 로그아웃.
    처리: users.status='미출근' + 세션 삭제 + 메모리 심박/구역 상태 정리
    → /login 리다이렉트
    """
    e = request.session.get('user_email')
    if e:
        await database.execute("UPDATE users SET status = '미출근' WHERE email = :e", {"e": e})
        print(f"🌐 웹 로그아웃: {e}")
        with _user_hr_lock:
            _user_heart_rates.pop(e, None)
            _user_hr_last_logged.pop(e, None)
        with _user_zone_lock:
            _user_zone_states.pop(e, None)
    request.session.clear()
    return RedirectResponse(url="/login")


# [일일 보고서] 오늘 출근자·미출근자 목록 + 당일 전체 위반 이력 종합 (관리자용 일일 안전 보고서)
@app.get("/daily_report", name="daily_report")
async def daily_report(request: Request):
    if 'user_name' not in request.session: 
        return RedirectResponse(url="/login")
    
    try:
        today = date.today()
        today_str = f"{today.year}년 {today.month:02d}월 {today.day:02d}일"
        today_attendee_emails = await get_today_attendee_emails()
        
        all_users = await database.fetch_all("SELECT name, phone, email, status FROM users")
        attendees = []
        absentees = []
        
        for u in all_users:
            user_dict = dict(u)
            if user_dict['email'] in today_attendee_emails:
                attendees.append(user_dict)
            else:
                user_dict['status'] = '미출근'
                absentees.append(user_dict)
                
        query = """
            SELECT name, violation_type, image_url, detected_at 
            FROM violations 
            WHERE DATE(detected_at) = CURDATE() 
            ORDER BY detected_at DESC
        """
        rows = await database.fetch_all(query)
        
        violations = []
        for r in rows:
            v = dict(r)
            if v['detected_at']:
                if isinstance(v['detected_at'], str):
                    v['time'] = v['detected_at'].split()[1] if ' ' in v['detected_at'] else v['detected_at']
                else:
                    v['time'] = v['detected_at'].strftime('%H:%M:%S')
            else:
                v['time'] = ""
            violations.append(v)
            
        return templates.TemplateResponse(
            request=request, 
            name="daily_report.html", 
            context={
                "user_name": request.session['user_name'],
                "today_str": today_str,
                "attendees": attendees,
                "absentees": absentees,
                "violations": violations,
                "total_violations": len(violations)
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(
            content=f"<div style='padding:50px;font-family:sans-serif;'><h2>🚨 서버 내부 에러가 발생했습니다</h2><p style='color:red;'><b>에러 내용:</b> {str(e)}</p></div>", 
            status_code=500
        )


# [실시간 AI 뷰 프록시] realtime.py(포트 8000)의 HTML 페이지를 main.py가 대신 제공 (게스트 허용)
@app.get("/realtime", name="realtime")
async def realtime_view(request: Request):
    if 'user_name' not in request.session:
        request.session['user_name'] = '게스트'
        request.session['user_email'] = 'guest@safety.com'
        request.session['is_admin'] = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{REALTIME_SERVER_URL}/")
            return HTMLResponse(content=r.text, status_code=r.status_code)
    except Exception as e:
        return HTMLResponse(
            content=f"""
            <html><body style='font-family:sans-serif;padding:40px;background:#080c14;color:#e2e8f0'>
            <h2>⚠️ Realtime 서버 연결 실패</h2>
            <p>{e}</p>
            </body></html>
            """,
            status_code=503
        )

# [카메라 스트림 프록시] realtime.py MJPEG 스트림을 버퍼링 없이 클라이언트에 중계 (지연 최소화)
@app.get("/video_feed")
async def video_feed(request: Request):
    """★ [v2] 라이브 영상 프록시 — 딜레이 누적 방지
       - 기존: aiter_bytes(chunk_size=65536) → 64KB 차야 송출 → 작은 프레임 적체
       - 변경: aiter_raw()로 들어오는 즉시 클라이언트에 전달 (버퍼링 제거)
       - realtime.py(포트 8000)의 스트림을 그대로 클라이언트에 중계
       - + httpx 자체의 receive 버퍼도 최소화하기 위해 limits/timeout 명시
    """
    async def stream_proxy():
        try:
            limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)
            timeout = httpx.Timeout(connect=5.0, read=None, write=None, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                async with client.stream("GET", f"{REALTIME_SERVER_URL}/video_feed") as resp:
                    # ★ aiter_raw()는 들어오는 청크를 모으지 않고 그대로 yield
                    async for chunk in resp.aiter_raw():
                        if not chunk:
                            continue
                        yield chunk
        except Exception:
            pass
    return StreamingResponse(
        stream_proxy(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",  # nginx 등 리버스 프록시 버퍼링 비활성화
        }
    )

# [위반 통합 조회] AI 감지(realtime.py deque) + DB 기록(심박·SOS·앱알림 등) 머지 후 시간순 50건 반환
@app.get("/violations")
async def get_violations():
    """
    ★ [수정] realtime.py의 in-memory deque + main.py DB의 최근 위반을 머지하여 반환.
       이전 버전은 realtime.py만 프록시했어서 live.html에 app_notified/heart_rate/
       emergency_sos/emergency_impact 같은 main.py 발 이벤트가 표시되지 않았음.

    소스 1) realtime.py deque : AI 감지 위반 (안전모 미착용, 쓰러짐 등)
    소스 2) main.py DB        : 심박 이상, SOS, 낙하, 앱 알림 등
    → 시간 역순 정렬 후 최대 50건 반환
    """
    merged = []

    # 1) realtime.py deque (안전모 미착용, 쓰러짐 등 AI 검출)
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{REALTIME_SERVER_URL}/violations")
            if r.status_code == 200:
                for v in r.json():
                    nm = v.get("name") or "미식별자"
                    # 이메일이 잘못 들어온 경우 @ 앞부분으로 표시 (DB 치환은 별도 SQL로)
                    if "@" in str(nm):
                        nm = str(nm).split("@")[0]
                    merged.append({
                        "id":        f"rt-{v.get('id')}",
                        "time":      v.get("time", ""),
                        "date":      v.get("date", ""),
                        "name":      nm,
                        "type":      v.get("type", ""),
                        "image_url": v.get("image_url", "")
                    })
    except Exception:
        pass

    # 2) main.py DB의 오늘자 최근 위반 (앱 알림/심박/SOS/낙하/IoT 등 포함)
    try:
        rows = await database.fetch_all(
            """SELECT id, name, violation_type AS type, image_url, detected_at
               FROM violations
               WHERE DATE(detected_at) = CURDATE()
               ORDER BY detected_at DESC LIMIT 50"""
        )
        for r in rows:
            dt = r["detected_at"]
            nm = r["name"] or "미식별자"
            if "@" in str(nm):
                # DB 단에서 이메일이 섞여있으면 즉시 이름으로 치환 시도
                try:
                    u = await database.fetch_one(
                        "SELECT name FROM users WHERE email = :e", {"e": nm}
                    )
                    if u and u["name"]:
                        nm = u["name"]
                    else:
                        nm = str(nm).split("@")[0]
                except Exception:
                    nm = str(nm).split("@")[0]
            merged.append({
                "id":        f"db-{r['id']}",
                "time":      dt.strftime("%H:%M:%S") if dt else "",
                "date":      dt.strftime("%Y-%m-%d") if dt else "",
                "name":      nm,
                "type":      r["type"] or "",
                "image_url": r["image_url"] or ""
            })
    except Exception as e:
        print(f"⚠️ /violations DB 조회 실패: {e}")

    # 3) 시간 역순 정렬 (date+time)
    merged.sort(key=lambda x: f"{x['date']} {x['time']}", reverse=True)
    return JSONResponse(merged[:50])


# [이메일→이름 매핑] 전체 사용자의 이메일:이름 딕셔너리 반환 (index.html / live.html 이름 표시 변환용)
@app.get("/api/email_name_map")
async def api_email_name_map():
    """★ [신규] index.html / live.html이 이메일→이름 변환에 사용하는 매핑."""
    try:
        rows = await database.fetch_all("SELECT email, name FROM users")
        return {e["email"]: e["name"] for e in rows if e["email"] and e["name"]}
    except Exception:
        return {}

# [실시간 통계 프록시] realtime.py 현재 FPS + 총 위반 수 반환 (live2.html 상단 수치 표시용)
@app.get("/stats")
async def get_stats():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{REALTIME_SERVER_URL}/stats")
            return JSONResponse(content=r.json(), status_code=r.status_code)
    except Exception:
        return JSONResponse(
            content={"fps": 0, "total": 0, "error": "realtime 서버 미연결"},
            status_code=503
        )

# [라이브 뷰] 카메라 스트림 + 위반 목록 실시간 표시 페이지 (게스트도 접근 가능)
@app.get("/live")
async def live(request: Request):
    if 'user_name' not in request.session:
        request.session['user_name'] = '게스트'
        request.session['user_email'] = 'guest@safety.com'
        request.session['is_admin'] = False
    return templates.TemplateResponse(request=request, name="live.html", context={"user_name": request.session['user_name']})

# [상세 라이브 대시보드] 카메라 스트림 + FPS·날씨·작업자 현황 종합 표시 (로그인 필요)
@app.get("/live2")
async def live2(request: Request):
    if 'user_name' not in request.session: return RedirectResponse(url="/login")
    return templates.TemplateResponse(request=request, name="live2.html", context={"user_name": request.session['user_name']})


# =============================================
#  ⚙️ [수정 및 추가] 마이페이지 정보 로드 및 수정 처리 라우트
# =============================================
# [마이페이지 - 조회] 로그인된 사용자의 프로필 정보(이름·전화·생년월일·성별) 조회
@app.get("/mypage", name="mypage")
async def mypage(request: Request, error: str = None, success: str = None):
    """DB에서 로그인된 사용자의 상세 정보를 안전하게 조회하여 화면에 전달"""
    if 'user_name' not in request.session: 
        return RedirectResponse(url="/login")
    
    email = request.session.get('user_email')
    
    try:
        user = await database.fetch_one("SELECT * FROM users WHERE email = :e", {"e": email})
        user_info = dict(user) if user else {}
        
        # HTML 날짜 입력창(<input type="date">) 표준 규격(YYYY-MM-DD) 호환 예외 처리
        if user_info.get('birthdate'):
            user_info['birthdate'] = str(user_info['birthdate']).split(' ')[0]
            
        return templates.TemplateResponse(
            request=request, 
            name="mypage.html", 
            context={
                "user_name": request.session['user_name'],
                "user_info": user_info,
                "error": error,
                "success": success
            }
        )
    except Exception as e:
        print(f"⚠️ 마이페이지 정보 로드 오류: {e}")
        return RedirectResponse(url="/index")

# [마이페이지 - 수정] 이름·전화번호·생년월일·성별·비밀번호 변경 처리, 세션 이름도 즉시 동기화
@app.post("/mypage/update")
async def update_mypage(request: Request):
    """사용자 요청 정보 업데이트 수행 및 정교한 입력값 유효성/예외 검증"""
    if 'user_name' not in request.session: 
        return RedirectResponse(url="/login")
        
    email = request.session.get('user_email')
    form = await request.form()
    
    name = form.get("name")
    phone = form.get("phone")
    password = form.get("password")
    password_confirm = form.get("password_confirm")
    birthdate = form.get("birthdate")
    gender = form.get("gender")
    
    # 비밀번호 검증: 두 값이 입력되었으나 불일치할 경우 에러 처리
    if password and (password != password_confirm):
        return RedirectResponse(url="/mypage?error=비밀번호가 서로 일치하지 않습니다.", status_code=303)
        
    try:
        # 동적 쿼리 설계: 사용자가 새 비밀번호를 입력하지 않았다면 기존 비밀번호 유지
        update_query = """
            UPDATE users 
            SET name = :n, phone = :ph, birthdate = :b, gender = :g 
        """
        params = {
            "n": name, 
            "ph": phone, 
            "b": birthdate, 
            "g": gender, 
            "e": email
        }
        
        if password:
            update_query += ", password = :pw "
            params["pw"] = password
            
        update_query += " WHERE email = :e"
        
        # 안전한 트랜잭션 비동기 실행
        await database.execute(update_query, params)
        
        # 이름을 수정했을 경우를 대비하여 세션 동기화 처리
        if name:
            request.session['user_name'] = name
            
        print(f"✅ 회원정보 수정 완료 (ID: {email})")
        return RedirectResponse(url="/mypage?success=성공적으로 정보가 수정되었습니다.", status_code=303)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return RedirectResponse(url="/mypage?error=서버 처리 중 에러가 발생했습니다.", status_code=303)


# [게스트 로그인] DB 조회 없이 게스트 세션만 설정 → /index 이동 (시연·테스트용)
@app.get("/guest_login")
async def guest_login(request: Request):
    request.session['user_name'] = '게스트'
    request.session['user_email'] = 'guest@safety.com'
    request.session['is_admin'] = False
    return RedirectResponse(url="/index", status_code=302)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)