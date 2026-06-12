import os
os.environ.setdefault("ORT_CUDNN_CONV_USE_MAX_WORKSPACE", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import json, cv2, numpy as np, time, datetime, threading, math, queue
from collections import deque, Counter
from ultralytics import YOLO

# ───────── Hungarian 매칭 (겹침 시 라벨 swap 방지) ─────────
try:
    from scipy.optimize import linear_sum_assignment
    _HUNGARIAN_OK = True
    print("✅ [Tracker] scipy.linear_sum_assignment 사용 (Hungarian 매칭)")
except Exception as _e:
    _HUNGARIAN_OK = False
    print(f"⚠️ [Tracker] scipy 미설치 → greedy 매칭으로 폴백: {_e}")

from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import urllib.request

# ═════════════════════════════════════════════════════════════════
# 성능 최적화 스위치
# ═════════════════════════════════════════════════════════════════
USE_ESPCN_SR        = False
ESPCN_MIN_FACE_SIZE = 35
ENHANCE_LIGHTWEIGHT = True
USE_ASYNC_YOLO      = True

EMB_MULTI_EXTRACT   = 1
EMB_MULTISCALE      = False
EMB_MULTISCALE_RECONFIRM = True

AUTO_CALIBRATE_THRESHOLDS = True
USE_RECENT_ID_CACHE       = True
RECENT_ID_CACHE_TTL       = 180.0
RECENT_ID_CACHE_SIM       = 0.58
USE_FAR_TTA               = False
FAR_FACE_THRESHOLD_PX     = 55

USE_HIRES_VERIFY          = True
HIRES_ASYNC               = True

# ───────── 영상 송출 파라미터 (버퍼링 최적화) ─────────
STREAM_WIDTH    = 640       # 원격용 저해상도
STREAM_HEIGHT   = 360
STREAM_JPEG_Q   = 55        # JPEG 품질 ↓ = 대역폭 ↓
STREAM_SEND_FPS = 15        # 송출 FPS — 처리 FPS(30)와 분리해 버퍼 누적 ↓

# ───────── 동작(쓰러짐/넘어짐) 감지 ─────────
EARLY_ABANDON_UNKNOWN_WITH_FACE = 4
EARLY_ABANDON_MIN_ELAPSED       = 2.0
EARLY_ABANDON_MIN_RESPONSES     = 4
WORKER_MAX_CANDIDATES_PER_CALL = 4
USE_RECENT_UNKNOWN_CACHE  = True
RECENT_UNKNOWN_CACHE_TTL  = 60.0
RECENT_UNKNOWN_CACHE_SIM  = 0.70

CONFIRMED_TRACK_IOU_THR      = 0.20
CONFIRMED_TRACK_DIST_RATIO   = 1.8
ANCHOR_EMB_SWAP_THRESHOLD    = 0.40
ANCHOR_EMB_SWAP_FAST_KILL    = 3
ANCHOR_EMB_SWAP_WINDOW       = 8.0

FALL_VELOCITY_THRESH   = 0.45
BODY_TILT_THRESH_DEG   = 55
HEAD_FOOT_FLAT_THRESH  = 0.35
POSE_CONFIRM_FRAMES    = 5
POSE_MIN_DURATION      = 0.7
POSE_LOG_COOLDOWN      = 15.0
POSE_CHECK_INTERVAL    = 0.15
POSE_KPT_CONF_THRESH   = 0.40
POSE_MIN_KPT_VISIBLE   = 7
POSE_HISTORY_LEN       = 10
POSE_DISPLAY_DURATION  = 5.0
DEBUG_POSE             = True

# ───────── ESPCN 초해상도 (선택) ─────────
sr = None
if USE_ESPCN_SR:
    try:
        from cv2 import dnn_superres
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel("ESPCN_x4.pb")
        sr.setModel("espcn", 4)
        try:
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("✅ [ESPCN] CUDA 백엔드")
        except Exception:
            print("✅ [ESPCN] CPU 백엔드")
    except Exception as e:
        print(f"⚠️ [ESPCN] 로드 실패: {e}")
        sr = None
else:
    print("ℹ️ [ESPCN] OFF")

# ───────── 멀티 GPU 분배 ─────────
_NUM_GPUS = 0
try:
    import torch as _torch_probe
    _NUM_GPUS = _torch_probe.cuda.device_count()
    del _torch_probe
except Exception:
    pass

GPU_YOLO_MAIN = 0
if _NUM_GPUS >= 3:
    FACE_POOL_GPUS = [1, 2]
elif _NUM_GPUS == 2:
    FACE_POOL_GPUS = [1]
else:
    FACE_POOL_GPUS = [0]

GPU_HIRES = 3 if _NUM_GPUS >= 4 else (_NUM_GPUS - 1 if _NUM_GPUS >= 1 else 0)
print(f"🖥️  GPU {_NUM_GPUS}장 → YOLO:{GPU_YOLO_MAIN}, Face:{FACE_POOL_GPUS}, HiRes:{GPU_HIRES}")

# ───────── InsightFace 풀 ─────────
USE_INSIGHTFACE = True
face_pool = None
try:
    from insightface.app import FaceAnalysis
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    print(f"[ONNX] providers: {available_providers}")

    class FacePool:
        """여러 GPU에 InsightFace 인스턴스를 분산 배치하고 라운드로빈으로 분배."""

        def __init__(self, gpu_ids, det_size=(640, 640), det_thresh=0.30):
            """각 GPU별 FaceAnalysis 앱을 준비하고 카운터/락 초기화."""
            self.apps = []
            for gid in gpu_ids:
                try:
                    app = FaceAnalysis(name='buffalo_l', providers=[
                        ('CUDAExecutionProvider', {'device_id': gid}),
                        'CPUExecutionProvider',
                    ])
                    app.prepare(ctx_id=gid, det_size=det_size, det_thresh=det_thresh)
                    self.apps.append(app)
                    print(f"✅ [InsightFace] GPU {gid} det={det_size}")
                except Exception as e:
                    print(f"⚠️ GPU {gid} 로드 실패: {e}")
            if not self.apps:
                raise RuntimeError("FacePool 초기화 실패")
            self._counter = 0
            self._lock = threading.Lock()

        def acquire(self):
            """라운드로빈으로 다음 앱 핸들 반환."""
            with self._lock:
                app = self.apps[self._counter % len(self.apps)]
                self._counter += 1
            return app

    face_pool = FacePool(FACE_POOL_GPUS, det_size=(640, 640), det_thresh=0.25)
except Exception as e:
    print(f"⚠️ InsightFace 폴백: {e}")
    USE_INSIGHTFACE = False
    from deepface import DeepFace

# ───────── PyTorch CUDA ─────────
USE_CUDA = False
try:
    import torch
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        print(f"✅ [PyTorch] CUDA")
except Exception:
    pass

# ───────── 경로 ─────────
YOLO_MODEL       = r"runs\detect\helmet_yolo11\weights\best.pt"
YOLO_POSE_MODEL  = "yolo11n-pose.pt"
DB_PATH          = "face_db.json"
FACE_DB_DIR      = "face_db"
EVIDENCE_DIR     = "evidence"
HTML_FILE        = "realtime.html"
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(FACE_DB_DIR, exist_ok=True)
os.makedirs(os.path.join(EVIDENCE_DIR, "MotionAlerts"), exist_ok=True)
os.makedirs(os.path.join(EVIDENCE_DIR, "unknown"), exist_ok=True)

# ═════════════════════════════════════════════════════════════════
# 메인 서버(main.py)로 위반 알림 전송
# ═════════════════════════════════════════════════════════════════
MAIN_SERVER_URL       = "http://127.0.0.1:8080"
NOTIFY_MAIN_SERVER    = True
NOTIFY_TIMEOUT        = 3.0

NOTIFY_COOLDOWN       = 30.0
_notify_last_sent     = {}
_notify_lock          = threading.Lock()
_notify_queue         = queue.Queue(maxsize=200)


def _notify_worker_loop():
    """알림 큐를 비우면서 main.py에 위반/void 요청을 순차 전송하는 워커 스레드."""
    while True:
        try:
            payload = _notify_queue.get()
            if payload is None:
                continue
            if payload.get("__void__"):
                try:
                    data = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(
                        f"{MAIN_SERVER_URL}/api/void_violation",
                        data=data,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=NOTIFY_TIMEOUT) as resp:
                        _ = resp.read()
                    print(f"  🧹 main.py void 요청 전송: track#{payload.get('track_id')} "
                          f"→ {payload.get('replaced_by')}", flush=True)
                except Exception as e:
                    print(f"  ⚠️ void 요청 전송 실패(무시): {e}", flush=True)
                continue

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{MAIN_SERVER_URL}/api/violation",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=NOTIFY_TIMEOUT) as resp:
                _ = resp.read()
            print(f"  📡 메인서버 알림 전송: {payload.get('name')} ({payload.get('type')})",
                  flush=True)
        except Exception as e:
            print(f"  ⚠️ 메인서버 알림 전송 실패: {e}", flush=True)


def notify_main_server(name, vtype, image_url, track_id=None):
    """위반 발생 시 메인 서버 알림 큐에 enqueue. 이름/종류별 쿨다운 적용."""
    if not NOTIFY_MAIN_SERVER:
        return

    is_unknown = (not name) or (name in ("unknown", "미식별자")) or str(name).startswith("미식별자")
    send_name = name if (name and not is_unknown) else "미식별자"

    now = time.time()
    cooldown_key = send_name if not is_unknown else f"__unknown__:{vtype}"
    with _notify_lock:
        last = _notify_last_sent.get(cooldown_key, 0.0)
        if (now - last) < NOTIFY_COOLDOWN:
            return
        _notify_last_sent[cooldown_key] = now

    payload = {
        "name": send_name,
        "type": vtype,
        "image_url": image_url or "",
        "source": "realtime_ai",
        "track_id": int(track_id) if track_id is not None else None,
        "is_unknown": bool(is_unknown),
        "sent_at": now,
    }
    try:
        _notify_queue.put_nowait(payload)
    except queue.Full:
        print("  ⚠️ 알림 큐 가득참 — 이번 위반 전송 생략", flush=True)


if NOTIFY_MAIN_SERVER:
    threading.Thread(target=_notify_worker_loop, daemon=True).start()
    print(f"📡 [메인서버 알림] ON → {MAIN_SERVER_URL}/api/violation")
else:
    print("ℹ️ [메인서버 알림] OFF")

# ═════════════════════════════════════════════════════════════════
# 얼굴 인식 임계값/품질 게이트
# ═════════════════════════════════════════════════════════════════
FACE_THR_INSIGHT = 0.62
FACE_THR_DEEPF   = 0.84
FACE_MARGIN      = 0.11
FACE_STRONG_ABS  = 0.74
FACE_STRONG_MARG = 0.16
FACE_CONFUSION_MARGIN = 0.06
FACE_TOP3_VAR_MAX = 0.05

INSIGHT_DET_SCORE_MIN_NEAR = 0.55
INSIGHT_DET_SCORE_MIN_FAR  = 0.40
EMB_MIN_FACE_SIZE_NEAR     = 28
EMB_MIN_FACE_SIZE_FAR      = 18
EMB_MIN_SHARPNESS_NEAR     = 22
EMB_MIN_SHARPNESS_FAR      = 12

FACE_MAX_YAW   = 38.0
FACE_MAX_PITCH = 32.0
FACE_MAX_ROLL  = 32.0

FACE_VOTE_MIN       = 2
FACE_VOTE_RATIO     = 0.65
FACE_VOTE_STRONG    = 0.85
FACE_VOTE_WINDOW    = 8.0
FACE_VOTE_MARGIN_MIN = 0.13

RECONFIRM_INTERVAL        = 40.0
RECONFIRM_DISAGREE_MAX    = 3
RECONFIRM_SCORE_MIN       = 0.55
RECONFIRM_OTHER_MIN       = 0.66
KILL_COOLDOWN_AFTER_SWAP  = 2.0

# ───────── YOLO 파라미터 ─────────
YOLO_IMGSZ = 1280
YOLO_CONF_NEAR = 0.45
YOLO_CONF_FAR = 0.20
YOLO_CONF_MIN = 0.15
YOLO_INTERVAL = 3
YOLO_MIN_SIZE = 12
YOLO_ASPECT_MIN = 0.50
YOLO_ASPECT_MAX = 1.6
DET_NMS_IOU = 0.4
FAR_BOX_THRESHOLD = 40
TARGET_FPS = 30
DEBUG_MATCH = True

# ───────── 줌 / 트래커 / 안전모 ─────────
ZOOM_DURATION = 3.0
ZOOM_MIN_HEAD = 150
ZOOM_SCALE = 4
ZOOM_COOLDOWN = 2.0
ZOOM_SUSPECT_FRAMES = 2
TRACK_IOU_THR = 0.20
TRACK_DIST_RATIO = 2.0
TRACK_MAX_MISS = 60
MIN_TRACK_HITS = 2
MIN_TRACK_AGE = 2

HELMET_HISTORY_LEN = 12
HELMET_VIOLATION_FAST = 3
HELMET_VIOLATION_NEED = 5
HELMET_VIOLATION_FAR  = 4

BEST_SHOT_MAX               = 6
RE_REQUEST_INTERVAL         = 0.30
RE_REQUEST_INITIAL          = 0.15
IDENTIFICATION_GRACE        = 1.5
MIN_FACE_CHECKS             = 2
MAX_IDENTIFICATION_TIME     = 5.0
MAX_WORKER_RESPONSES        = 8
CANDIDATE_KEEP_TIME         = 6.0
MIN_SUCCESSFUL_RESPONSES    = 2

# ───────── 쿨다운 (식별자/미식별자 분리) ─────────
PERSON_LOG_COOLDOWN        = 180.0
UNKNOWN_LOG_COOLDOWN       = 3.0
UNKNOWN_SIM_THRESHOLD      = 0.72
UNKNOWN_SPATIAL_RADIUS     = 80

WORKER_COUNT = max(4, len(FACE_POOL_GPUS) * 4)
SAVE_ON_TRACK_LOST = True
LOST_TRACK_MIN_VIOLATIONS = 3
VIOLATION_ID_COUNTER = [0]

if not os.path.exists(YOLO_MODEL):
    print(f"❌ YOLO 모델 없음: {YOLO_MODEL}")
    exit()


# ═════════════════════════════════════════════════════════════════
# 카메라 수신 — 내부 버퍼 1프레임으로 제한해 지연 누적 방지
# ═════════════════════════════════════════════════════════════════
class CameraGrabber:
    """별도 스레드로 카메라에서 프레임을 즉시 비우면서 최신 프레임만 보관."""

    def __init__(self, src=0):
        """카메라 오픈, 내부 버퍼 1프레임 제한, MJPG 포맷, 30fps 설정."""
        self.cap = cv2.VideoCapture(src)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._update, daemon=True)
        self.t.start()

    def _update(self):
        """grab()으로 빠르게 큐를 비우고 retrieve()로 디코딩만 수행."""
        while self.running:
            if not self.cap.grab():
                time.sleep(0.005)
                continue
            ret, frame = self.cap.retrieve()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        """최신 프레임 사본 반환, 없으면 None."""
        with self.lock:
            if not self.ret or self.frame is None:
                return None
            return self.frame.copy()

    def is_alive(self):
        """스레드와 카메라가 모두 살아있는지 확인."""
        return self.t.is_alive() and self.cap.isOpened()

    def stop(self):
        """수신 종료 및 카메라 해제."""
        self.running = False
        if self.cap.isOpened():
            self.cap.release()


# ───────── 디지털 줌 / HiRes 재검증 ─────────
def digital_zoom_crop(frame, bbox, scale=2):
    """bbox 주변을 잘라 원본 크기로 확대 — 디지털 줌 효과."""
    hx, hy, hw, hh = bbox
    fh, fw = frame.shape[:2]

    cx = hx + hw // 2
    cy = hy + hh // 2
    crop_w = int(fw / scale)
    crop_h = int(fh / scale)

    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(fw, x1 + crop_w)
    y2 = min(fh, y1 + crop_h)

    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    zoomed = cv2.resize(crop, (fw, fh), interpolation=cv2.INTER_LINEAR)
    nx = int((hx - x1) * scale)
    ny = int((hy - y1) * scale)
    nw = int(hw * scale)
    nh = int(hh * scale)

    return zoomed, (nx, ny, nw, nh)


def verify_violation_hires(frame, bbox):
    """확대 영상으로 안전모 미착용을 재검증 — 멀리 있는 작은 머리의 오탐 줄임."""
    if not USE_HIRES_VERIFY or yolo_hires is None:
        return True
    try:
        zoomed, nbbox = digital_zoom_crop(frame, bbox, scale=2)
        if zoomed is None:
            return True
        results = list(yolo_hires(zoomed, imgsz=1280, half=True,
                                   device=GPU_HIRES, conf=0.30,
                                   stream=True, verbose=False))
        best_iou = 0.0
        best_cls = -1
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                iou_val = iou((x1, y1, x2-x1, y2-y1), nbbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_cls = int(box.cls[0])
        if best_iou < 0.20:
            return True
        return (best_cls == 0)
    except Exception:
        return True


# ───────── 유틸 ─────────
def base_face_thr():
    """기본 얼굴 인식 임계값 (백엔드별)."""
    return FACE_THR_INSIGHT if USE_INSIGHTFACE else FACE_THR_DEEPF


def clamp_thr(t):
    """임계값이 비정상적으로 튀지 않도록 안전 범위로 클램프."""
    return max(0.42, min(t, 0.68)) if USE_INSIGHTFACE else max(0.70, min(t, 0.86))


def cosine_sim(v1, v2):
    """두 벡터의 코사인 유사도."""
    a = np.asarray(v1, dtype=np.float32).ravel()
    b = np.asarray(v2, dtype=np.float32).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 1e-8 else 0.0


def iou(a, b):
    """두 박스의 IoU (xywh 형식)."""
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def safe_imwrite(path, img):
    """한글 경로 안전한 이미지 저장 (imencode + tofile)."""
    try:
        ext = os.path.splitext(path)[1] or ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok: return False
        buf.tofile(path)
        return os.path.exists(path)
    except Exception:
        return False


def compute_self_similarity_stats(features):
    """등록된 임베딩들 사이의 자기 유사도 통계 (평균/분위수/최소)."""
    n = len(features)
    if n < 2: return None
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_sim(features[i], features[j]))
    if not sims: return None
    sims_sorted = sorted(sims)
    return {
        "mean": float(np.mean(sims)),
        "p10": sims_sorted[max(0, int(len(sims_sorted) * 0.10))],
        "p25": sims_sorted[max(0, int(len(sims_sorted) * 0.25))],
        "min": sims_sorted[0],
        "n_pairs": len(sims),
    }


def auto_calibrate_threshold(features):
    """등록 사진의 자기 유사도 p10을 기반으로 개인별 임계값 자동 산출."""
    stats = compute_self_similarity_stats(features)
    if stats is None: return None, None
    thr = max(0.42, min(0.65, stats["p10"] - 0.07))
    return thr, stats


# ═════════════════════════════════════════════════════════════════
# 재진입 캐시 (식별자/미식별자)
# ═════════════════════════════════════════════════════════════════
class RecentIdentityCache:
    """최근 확정된 이름과 임베딩을 짧게 보관 — 같은 사람이 다시 나타나면 즉시 매칭."""

    def __init__(self, ttl=180.0, sim_threshold=0.58):
        """TTL과 유사도 임계값 설정."""
        self._cache = {}
        self._lock = threading.Lock()
        self.ttl = ttl
        self.sim_threshold = sim_threshold

    def add(self, name, embedding, now):
        """확정된 (이름, 임베딩) 추가. 사람당 최근 6개 유지."""
        if not name or name == "unknown" or embedding is None:
            return
        with self._lock:
            self._cache.setdefault(name, []).append((embedding, now))
            self._cache[name] = self._cache[name][-6:]
            self._gc(now)

    def lookup(self, embedding, now):
        """캐시에서 가장 유사한 이름과 점수를 반환. 임계값 미달이면 None."""
        if embedding is None: return None, 0.0
        with self._lock:
            self._gc(now)
            best_name, best_sim = None, 0.0
            for name, embs in self._cache.items():
                for emb, _ts in embs:
                    sim = cosine_sim(embedding, emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name
            if best_sim >= self.sim_threshold:
                return best_name, best_sim
            return None, best_sim

    def _gc(self, now):
        """TTL 지난 항목 제거."""
        cutoff = now - self.ttl
        for name in list(self._cache.keys()):
            self._cache[name] = [(e, t) for e, t in self._cache[name] if t >= cutoff]
            if not self._cache[name]:
                del self._cache[name]

    def status(self, now):
        """디버그용 — 캐시 내 이름별 개수 스냅샷."""
        with self._lock:
            self._gc(now)
            return {n: len(es) for n, es in self._cache.items()}


recent_id_cache = RecentIdentityCache(ttl=RECENT_ID_CACHE_TTL, sim_threshold=RECENT_ID_CACHE_SIM)


class RecentUnknownCache:
    """최근에 본 미식별자 임베딩을 보관 — 같은 미식별자 반복 처리 방지."""

    def __init__(self, ttl=120.0, sim_threshold=0.62, max_items=80):
        """TTL과 유사도 임계값, 최대 보관 개수 설정."""
        self._items = []
        self._lock = threading.Lock()
        self.ttl = ttl
        self.sim_threshold = sim_threshold
        self.max_items = max_items

    def hit(self, emb, now):
        """캐시 적중 여부와 최대 유사도 반환."""
        if emb is None: return False, 0.0
        with self._lock:
            self._gc(now)
            best_sim = 0.0
            for e, _ in self._items:
                s = cosine_sim(emb, e)
                if s > best_sim:
                    best_sim = s
            return (best_sim >= self.sim_threshold), best_sim

    def add(self, emb, now):
        """미식별자 임베딩 추가."""
        if emb is None: return
        with self._lock:
            self._items.append((np.asarray(emb, dtype=np.float32), now))
            self._gc(now)
            if len(self._items) > self.max_items:
                self._items = self._items[-self.max_items:]

    def _gc(self, now):
        """TTL 지난 항목 제거."""
        cutoff = now - self.ttl
        self._items = [(e, t) for e, t in self._items if t >= cutoff]

    def size(self, now):
        """현재 캐시 크기."""
        with self._lock:
            self._gc(now)
            return len(self._items)


recent_unknown_cache = RecentUnknownCache(ttl=RECENT_UNKNOWN_CACHE_TTL, sim_threshold=RECENT_UNKNOWN_CACHE_SIM)


# ═════════════════════════════════════════════════════════════════
# UnknownPersonRegistry — 미식별자에게 일관된 ID 부여
# ═════════════════════════════════════════════════════════════════
UNKNOWN_REGISTRY_SIM_THR = 0.55
UNKNOWN_MAX_EMBS_PER_PERSON = 8
UNKNOWN_PRESENCE_WINDOW = 3.0


class UnknownPersonRegistry:
    """미식별자들을 임베딩 기반으로 묶어 #1, #2 … 식의 고유 ID 부여."""

    def __init__(self, sim_threshold=0.55, max_embs=8, ttl=None,
                 presence_window=3.0):
        """파라미터 초기화 (TTL=None이면 영구 보관)."""
        self._records = []
        self._next_id = 1
        self._lock = threading.Lock()
        self.sim_threshold = sim_threshold
        self.max_embs = max_embs
        self.ttl = ttl
        self.presence_window = presence_window

    def _gc(self, now):
        """TTL 사용 시 오래된 레코드 제거."""
        if self.ttl is None: return
        cutoff = now - self.ttl
        self._records = [r for r in self._records if r["last_seen"] >= cutoff]

    def _best_match(self, embedding):
        """임베딩과 가장 유사한 기존 레코드 검색."""
        if embedding is None: return None, 0.0
        best_sim = 0.0
        best_rec = None
        for r in self._records:
            if not r["embs"]: continue
            for e in r["embs"]:
                sim = cosine_sim(embedding, e)
                if sim > best_sim:
                    best_sim = sim
                    best_rec = r
        if best_rec is not None and best_sim >= self.sim_threshold:
            return best_rec, best_sim
        return None, best_sim

    def assign_id(self, embedding, now):
        """기존 레코드와 매칭되면 그 ID, 아니면 새 ID 발급."""
        with self._lock:
            self._gc(now)
            rec, _sim = self._best_match(embedding)
            if rec is not None:
                if embedding is not None:
                    rec["embs"].append(np.asarray(embedding, dtype=np.float32))
                    if len(rec["embs"]) > self.max_embs:
                        rec["embs"] = rec["embs"][-self.max_embs:]
                rec["last_seen"] = now
                return rec["id"]
            uid = f"미식별자#{self._next_id}"
            self._next_id += 1
            self._records.append({
                "id": uid,
                "embs": ([np.asarray(embedding, dtype=np.float32)]
                         if embedding is not None else []),
                "last_seen": now,
                "logged_at": 0.0,
                "log_count": 0,
            })
            return uid

    def touch(self, unknown_id, now, embedding=None):
        """해당 ID의 last_seen 갱신, 새 임베딩 있으면 추가."""
        with self._lock:
            for r in self._records:
                if r["id"] == unknown_id:
                    r["last_seen"] = now
                    if embedding is not None and len(r["embs"]) < self.max_embs:
                        r["embs"].append(np.asarray(embedding, dtype=np.float32))
                    return True
        return False

    def is_currently_logged(self, unknown_id, now):
        """해당 미식별자가 현재 화면에 있으면서 이미 로그된 상태인지."""
        with self._lock:
            for r in self._records:
                if r["id"] != unknown_id: continue
                if r["logged_at"] <= 0: return False
                return (now - r["last_seen"]) < self.presence_window
        return False

    def mark_logged(self, unknown_id, now, embedding=None):
        """위반 로그된 시점 기록."""
        with self._lock:
            for r in self._records:
                if r["id"] == unknown_id:
                    r["logged_at"] = now
                    r["last_seen"] = now
                    r["log_count"] = r.get("log_count", 0) + 1
                    if embedding is not None:
                        r["embs"].append(np.asarray(embedding, dtype=np.float32))
                        if len(r["embs"]) > self.max_embs:
                            r["embs"] = r["embs"][-self.max_embs:]
                    return True
        return False

    def status(self, now):
        """디버그용 — 전체/로그됨/현재화면 카운트와 최근 ID 목록."""
        with self._lock:
            self._gc(now)
            present = sum(1 for r in self._records
                          if (now - r["last_seen"]) < self.presence_window)
            return {
                "total": len(self._records),
                "logged": sum(1 for r in self._records if r["logged_at"] > 0),
                "currently_present": present,
                "ids": [r["id"] for r in self._records[-20:]],
            }


unknown_registry = UnknownPersonRegistry(
    sim_threshold=UNKNOWN_REGISTRY_SIM_THR,
    max_embs=UNKNOWN_MAX_EMBS_PER_PERSON,
    ttl=None,
    presence_window=UNKNOWN_PRESENCE_WINDOW,
)


# ───────── 한글 텍스트 렌더링 ─────────
_KR_FONT_PATHS = [
    r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
_kr_font_cache = {}
_kr_font_available = None


def _load_kr_font(size: int):
    """OS별 한글 폰트를 찾아 PIL ImageFont로 로드. 캐싱됨."""
    global _kr_font_available
    if size in _kr_font_cache: return _kr_font_cache[size]
    for p in _KR_FONT_PATHS:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _kr_font_cache[size] = f
                if _kr_font_available is None:
                    _kr_font_available = True
                    print(f"🔤 한글 폰트: {p}")
                return f
            except Exception:
                continue
    if _kr_font_available is None:
        _kr_font_available = False
    f = ImageFont.load_default()
    _kr_font_cache[size] = f
    return f


def put_text_kr(img, text, pos_bl, font_size=16, color=(255, 255, 255)):
    """OpenCV 이미지 위에 PIL로 한글 텍스트를 그림. 실패 시 ASCII 폴백."""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = _load_kr_font(font_size)
        x, y_bl = pos_bl
        y_top = max(0, y_bl - int(font_size * 0.95))
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        draw.text((x, y_top), text, font=font, fill=rgb)
        np.copyto(img, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))
    except Exception:
        cv2.putText(img, text, pos_bl, cv2.FONT_HERSHEY_SIMPLEX, font_size / 32.0, color, 2)


def get_text_size_kr(text, font_size=16):
    """한글 텍스트의 (width, height) 추정."""
    try:
        font = _load_kr_font(font_size)
        if hasattr(font, "getbbox"):
            l, t, r, b = font.getbbox(text)
            return (r - l, b - t)
        else:
            return font.getsize(text)
    except Exception:
        return (len(text) * font_size // 2, font_size)


def laplacian_sharpness(gray):
    """Laplacian 분산값 — 이미지 선명도 지표."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def remove_overlapping_boxes(boxes_with_meta):
    """면적/신뢰도 우선 정렬 후 IoU NMS — YOLO 결과 중복 제거."""
    if len(boxes_with_meta) <= 1: return boxes_with_meta
    sorted_boxes = sorted(boxes_with_meta, key=lambda x: (x[0][2]*x[0][3], x[1]), reverse=True)
    kept = []
    for bbox, conf, ci in sorted_boxes:
        if not any(iou(bbox, kb) > DET_NMS_IOU for kb, _, _ in kept):
            kept.append((bbox, conf, ci))
    return kept


def enhance_image(img, face_size=None):
    """얼굴 크롭 향상 — 작은 얼굴은 ESPCN, CLAHE로 명암 보정."""
    if img is None or img.size == 0: return img
    try:
        if (USE_ESPCN_SR and sr is not None and face_size is not None
                and face_size < ESPCN_MIN_FACE_SIZE):
            img = sr.upsample(img)
            h, w = img.shape[:2]
            if max(h, w) > 600:
                scale = 600 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        if ENHANCE_LIGHTWEIGHT:
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
            cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            img_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            gaussian = cv2.GaussianBlur(img_clahe, (0, 0), 2.0)
            return cv2.addWeighted(img_clahe, 1.5, gaussian, -0.5, 0)
    except Exception:
        return img


def estimate_face_pose(landmarks_5):
    """5점 랜드마크에서 yaw/pitch/roll 각도 추정."""
    try:
        if landmarks_5 is None or len(landmarks_5) < 5:
            return 0.0, 0.0, 0.0
        le, re, nose, lm, rm = landmarks_5[0], landmarks_5[1], landmarks_5[2], landmarks_5[3], landmarks_5[4]
        dy = re[1] - le[1]
        dx = re[0] - le[0]
        roll = math.degrees(math.atan2(dy, dx))
        eye_cx = (le[0] + re[0]) / 2.0
        eye_dist = max(abs(re[0] - le[0]), 1.0)
        yaw = (nose[0] - eye_cx) / eye_dist * 90.0
        mouth_cy = (lm[1] + rm[1]) / 2.0
        eye_cy = (le[1] + re[1]) / 2.0
        face_h = max(abs(mouth_cy - eye_cy), 1.0)
        pitch_center = eye_cy + face_h * 0.5
        pitch = (nose[1] - pitch_center) / face_h * 90.0
        return float(yaw), float(pitch), float(roll)
    except Exception:
        return 0.0, 0.0, 0.0


def pose_ok(yaw, pitch, roll):
    """정면 얼굴인지 여부 (각도가 모두 허용 범위 안)."""
    return (abs(yaw) <= FACE_MAX_YAW and
            abs(pitch) <= FACE_MAX_PITCH and
            abs(roll) <= FACE_MAX_ROLL)


class ExtractStatus:
    """임베딩 추출 결과 상태 라벨."""
    OK = "ok"
    NO_FACE = "no_face"
    POOR_QUALITY = "poor"


def _is_far_size(face_size_hint):
    """얼굴이 멀리(작게) 검출되었는지."""
    if face_size_hint is None: return False
    return face_size_hint < FAR_FACE_THRESHOLD_PX


def _get_quality_gates(face_size_hint):
    """근거리/원거리별 (det_score, 얼굴크기, 선명도) 게이트 반환."""
    if _is_far_size(face_size_hint):
        return (INSIGHT_DET_SCORE_MIN_FAR, EMB_MIN_FACE_SIZE_FAR, EMB_MIN_SHARPNESS_FAR)
    return (INSIGHT_DET_SCORE_MIN_NEAR, EMB_MIN_FACE_SIZE_NEAR, EMB_MIN_SHARPNESS_NEAR)


def _insight_extract_one(crop, app=None, face_size_hint=None):
    """InsightFace 단일 추출 — 검출/품질/포즈 게이트 통과한 임베딩만 반환."""
    try:
        if app is None:
            app = face_pool.acquire()
        faces = app.get(crop)
        if not faces: return None, 0.0, 0.0, ExtractStatus.NO_FACE

        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        fw_ = f.bbox[2] - f.bbox[0]
        fh_ = f.bbox[3] - f.bbox[1]

        det_min, size_min, _ = _get_quality_gates(face_size_hint)

        if fw_ < size_min or fh_ < size_min:
            return None, 0.0, 0.0, ExtractStatus.POOR_QUALITY

        det_score = float(getattr(f, "det_score", 1.0))
        if det_score < det_min:
            return None, 0.0, det_score, ExtractStatus.POOR_QUALITY

        kps = getattr(f, "kps", None)
        if kps is not None:
            yaw, pitch, roll = estimate_face_pose(kps)
            if not pose_ok(yaw, pitch, roll):
                return None, 0.0, det_score, ExtractStatus.POOR_QUALITY

        emb = np.asarray(f.normed_embedding, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        size_factor = min(min(fw_, fh_) / 100.0, 1.0)
        face_quality = det_score * size_factor
        return emb, face_quality, det_score, ExtractStatus.OK
    except Exception:
        return None, 0.0, 0.0, ExtractStatus.NO_FACE


def extract_face_embedding(crop, face_size_hint=None, allow_multiscale=None):
    """얼굴 크롭에서 임베딩 추출 (선택적으로 좌우반전·확대 TTA)."""
    if crop is None or crop.size == 0:
        return None, ExtractStatus.NO_FACE
    h, w = crop.shape[:2]

    is_far = _is_far_size(face_size_hint)

    target = 500 if is_far else 400
    if max(h, w) < target and not (USE_ESPCN_SR and sr is not None):
        s = target / max(h, w)
        crop = cv2.resize(crop, (int(w*s), int(h*s)), interpolation=cv2.INTER_LANCZOS4)

    crop = enhance_image(crop, face_size=face_size_hint)
    use_multiscale = EMB_MULTISCALE if allow_multiscale is None else allow_multiscale

    if USE_INSIGHTFACE:
        app = face_pool.acquire()
        if is_far and USE_FAR_TTA and use_multiscale:
            h2, w2 = crop.shape[:2]
            crop_x15 = cv2.resize(crop, (int(w2*1.3), int(h2*1.3)), interpolation=cv2.INTER_LANCZOS4)
            variants = [crop, cv2.flip(crop, 1), crop_x15]
        elif use_multiscale:
            variants = [crop, cv2.flip(crop, 1)]
        else:
            emb, _, _, st = _insight_extract_one(crop, app=app, face_size_hint=face_size_hint)
            return emb, st

        embs, qs, statuses = [], [], []
        for v in variants:
            e, q, _ds, st = _insight_extract_one(v, app=app, face_size_hint=face_size_hint)
            statuses.append(st)
            if e is not None:
                embs.append(e)
                qs.append(q)
        if not embs:
            if all(s == ExtractStatus.NO_FACE for s in statuses):
                return None, ExtractStatus.NO_FACE
            return None, ExtractStatus.POOR_QUALITY
        if len(embs) == 1: return embs[0], ExtractStatus.OK
        weights = np.array(qs, dtype=np.float32) + 1e-3
        weights /= weights.sum()
        avg = sum(e * w for e, w in zip(embs, weights))
        return avg / (np.linalg.norm(avg) + 1e-8), ExtractStatus.OK
    else:
        for img_v in [crop, cv2.flip(crop, 1)]:
            for det in ["retinaface", "yunet", "opencv"]:
                try:
                    r = DeepFace.represent(img_path=img_v, model_name="Facenet512",
                                           enforce_detection=True, detector_backend=det)
                    if r:
                        emb = np.asarray(r[0]["embedding"], dtype=np.float32)
                        return emb / (np.linalg.norm(emb) + 1e-8), ExtractStatus.OK
                except Exception: continue
        return None, ExtractStatus.NO_FACE


def get_face_thr(entry):
    """개인별 임계값 — 자동보정값 → 자기유사도 하한 → 기본값 순으로."""
    return clamp_thr(entry.get("auto_thr") or
                     entry.get("self_sim_lower") or
                     base_face_thr())


def match_face(emb, db):
    """DB 전체와 임베딩을 비교해 1순위 매칭/혼동 여부 판정. 마진 부스트와 3순위 검사 포함."""
    scores, thrs, top_sims_map = {}, {}, {}
    for e in db:
        n = e["name"]
        thrs[n] = get_face_thr(e)
        feats = e.get("features", [])
        if feats:
            all_sims = sorted([cosine_sim(emb, f) for f in feats], reverse=True)
            top_k = all_sims[:min(3, len(all_sims))]
            top_sims_map[n] = top_k
            if len(top_k) >= 3:
                scores[n] = float(np.median(top_k))
            elif len(top_k) == 2:
                scores[n] = float(np.mean(top_k))
            else:
                scores[n] = float(top_k[0])
        else:
            scores[n] = 0.0
            top_sims_map[n] = []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked: return "unknown", 0.0, 0.0, False

    best, bs = ranked[0]
    second_score = ranked[1][1] if len(ranked) >= 2 else 0.0
    third_score  = ranked[2][1] if len(ranked) >= 3 else 0.0
    margin = bs - second_score

    best_sims = top_sims_map.get(best, [])
    internal_var = (max(best_sims) - min(best_sims)) if len(best_sims) >= 2 else 0.0
    unstable = internal_var > FACE_TOP3_VAR_MAX * 2

    is_confused = (len(ranked) >= 2 and
                   bs >= thrs[best] * 0.85 and
                   margin < FACE_CONFUSION_MARGIN)

    # 1·2위 점수 차이가 크면 약간의 부스트로 임계값 통과
    boosted = False
    if (0.42 <= bs < thrs[best]
            and margin >= 0.15
            and second_score < 0.45
            and not is_confused
            and not unstable):
        boost = min(0.10, margin * 0.5)
        bs_new = bs + boost
        if bs_new >= thrs[best]:
            if DEBUG_MATCH:
                print(f"  🚀 마진부스트: {best} {bs:.3f}→{bs_new:.3f} "
                      f"(thr={thrs[best]:.2f}, m={margin:.2f}, 2nd={second_score:.2f})")
            bs = bs_new
            boosted = True

    # 3위까지 점수가 다 붙어있으면 혼동으로 간주
    margin_3rd = bs - third_score
    if len(ranked) >= 3 and margin_3rd < FACE_CONFUSION_MARGIN * 1.5 and bs < FACE_STRONG_ABS:
        is_confused = True

    required_margin = (FACE_MARGIN * 0.7 if bs >= FACE_STRONG_ABS else
                       FACE_MARGIN if bs >= 0.55 else
                       FACE_MARGIN + 0.04)

    strong_match = (bs >= FACE_STRONG_ABS and margin >= FACE_STRONG_MARG and not unstable)
    matched = ((bs >= thrs[best] and margin >= required_margin
                and not is_confused and not unstable)
               or strong_match)

    if DEBUG_MATCH and bs > 0.30:
        top3 = ranked[:3]
        top3_str = ", ".join(f"{n}={s:.3f}" for n, s in top3)
        flags = []
        if strong_match: flags.append("STRONG")
        if is_confused: flags.append("CONFUSED")
        if unstable: flags.append("UNSTABLE")
        if boosted: flags.append("BOOSTED")
        flag_str = ('[' + ','.join(flags) + ']') if flags else ''
        status = ('✓' + best if matched else '✗') + flag_str
        print(f"  🔍 [{top3_str}] thr={thrs[best]:.2f} m={margin:.3f} → {status}", flush=True)

    return (best if matched else "unknown"), bs, margin, is_confused


def decide_face_only(face_name, face_score, face_margin, is_confused=False):
    """match_face 결과를 최종 (이름, 신뢰도, 점수)로 정리. 혼동이면 unknown."""
    if face_name == "unknown": return "unknown", 0.0, face_score
    if is_confused: return "unknown", 0.0, face_score
    return face_name, face_score, face_score


def enforce_name_uniqueness(tracks):
    """동일 이름으로 확정된 트랙이 둘 이상이면 약한 쪽을 kill 대상으로 표시."""
    name_map = {}
    kill_ids = set()
    for t in tracks:
        if not t.confirmed or t.name == "unknown" or t.marked_for_kill: continue
        if t.name not in name_map:
            name_map[t.name] = t
        else:
            cur = name_map[t.name]
            if cur.miss > 10 and t.miss == 0:
                kill_ids.add(cur.id)
                name_map[t.name] = t
            elif t.miss > 10 and cur.miss == 0:
                kill_ids.add(t.id)
            elif cur.identified_at > 0 and t.identified_at > cur.identified_at + 2.0:
                kill_ids.add(t.id)
            elif t.identified_at > 0 and cur.identified_at > t.identified_at + 2.0:
                kill_ids.add(cur.id)
                name_map[t.name] = t
            else:
                t_better = ((t.face > cur.face + 0.03) or
                            (abs(t.face - cur.face) <= 0.03 and
                             t.bbox[2]*t.bbox[3] > cur.bbox[2]*cur.bbox[3]))
                if t_better:
                    kill_ids.add(cur.id)
                    name_map[t.name] = t
                else:
                    kill_ids.add(t.id)
    return kill_ids


# ═════════════════════════════════════════════════════════════════
# 위반 쿨다운 — 식별자/미식별자 분리 관리
# ═════════════════════════════════════════════════════════════════
class ViolationCooldownGuard:
    """이름별/공간별 쿨다운을 추적해 위반 로그가 폭증하는 것을 방지."""

    def __init__(self):
        """이름별/미식별자 기록용 자료구조 초기화."""
        self._lock = threading.Lock()
        self._by_name = {}
        self._unknown_records = []

    def _gc(self, now):
        """미식별자 기록에서 만료된 항목 정리."""
        unk_cutoff = now - UNKNOWN_LOG_COOLDOWN * 1.2
        self._unknown_records = [r for r in self._unknown_records if r[1] >= unk_cutoff]

    def _is_blocked_unknown(self, embedding, bbox, now):
        """동일 미식별자(임베딩 유사 or 공간 근접)가 쿨다운 중인지 검사."""
        cutoff = now - UNKNOWN_LOG_COOLDOWN
        for emb, ts, b in self._unknown_records:
            if ts < cutoff: continue
            if embedding is not None and emb is not None:
                sim = cosine_sim(embedding, emb)
                if sim > UNKNOWN_SIM_THRESHOLD:
                    return True, f"임베딩 유사 ({sim:.2f})"
            if (embedding is None or emb is None) and b is not None and bbox is not None:
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
                fcx = b[0] + b[2] / 2
                fcy = b[1] + b[3] / 2
                if (cx - fcx) ** 2 + (cy - fcy) ** 2 < UNKNOWN_SPATIAL_RADIUS ** 2:
                    return True, f"공간 근접"
        return False, ""

    def try_acquire(self, name, embedding, bbox, now):
        """안전모 위반 로그 가능 여부 — 쿨다운 미경과면 False, 통과 시 기록."""
        with self._lock:
            self._gc(now)
            if name and name != "unknown":
                last = self._by_name.get(name, 0)
                if (now - last) < PERSON_LOG_COOLDOWN:
                    return False
                self._by_name[name] = now
                return True
            else:
                blocked, reason = self._is_blocked_unknown(embedding, bbox, now)
                if blocked:
                    if DEBUG_MATCH:
                        print(f"  🚫 미식별자 쿨다운 차단: {reason}", flush=True)
                    return False
                self._unknown_records.append((embedding, now, bbox))
                if DEBUG_MATCH:
                    print(f"  ✅ 미식별자 등록 (총 {len(self._unknown_records)}건 활성)", flush=True)
                return True

    def try_acquire_motion(self, name, embedding, bbox, now, cooldown=POSE_LOG_COOLDOWN):
        """쓰러짐/넘어짐/떨어짐 로그 가능 여부 — 별도 쿨다운 적용."""
        with self._lock:
            self._gc(now)
            key = f"motion:{name}" if name and name != "unknown" else None
            if key:
                last = self._by_name.get(key, 0)
                if (now - last) < cooldown:
                    return False
                self._by_name[key] = now
                return True
            else:
                cutoff = now - cooldown
                for emb, ts, b in self._unknown_records:
                    if ts < cutoff: continue
                    if b is not None and bbox is not None:
                        cx = bbox[0] + bbox[2] / 2
                        cy = bbox[1] + bbox[3] / 2
                        fcx = b[0] + b[2] / 2
                        fcy = b[1] + b[3] / 2
                        if (cx - fcx) ** 2 + (cy - fcy) ** 2 < UNKNOWN_SPATIAL_RADIUS ** 2:
                            return False
                self._unknown_records.append((embedding, now, bbox))
                return True

    def time_since_log(self, name):
        """해당 이름의 마지막 로그 이후 경과 시간."""
        with self._lock:
            last = self._by_name.get(name, 0)
        return time.time() - last if last > 0 else float('inf')

    def is_in_cooldown(self, name, embedding, bbox, now):
        """쿨다운 중인지 (UI 표시용 — 기록은 하지 않음)."""
        with self._lock:
            if name and name != "unknown":
                last = self._by_name.get(name, 0)
                return (now - last) < PERSON_LOG_COOLDOWN
            blocked, _ = self._is_blocked_unknown(embedding, bbox, now)
            return blocked


violation_guard = ViolationCooldownGuard()


# ═════════════════════════════════════════════════════════════════
# PoseEngine — YOLO-Pose 비동기 분석 (쓰러짐/넘어짐/떨어짐 감지)
# ═════════════════════════════════════════════════════════════════
class PoseEngine:
    """별도 스레드에서 YOLO-Pose 모델을 돌려 키포인트 기반으로 자세 상태를 판단."""

    def __init__(self, model_path=YOLO_POSE_MODEL, gpu_id=None):
        """큐/결과 dict 초기화 후 워커 스레드 시작."""
        self.queue = deque(maxlen=32)
        self.results = {}
        self._lock = threading.Lock()
        self.running = True
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.model_loaded = False
        self._debug_count = 0
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"🤸 [PoseEngine] YOLO-Pose 비동기 분석기 초기화 (모델: {model_path})")

    def submit(self, tid, frame_crop, global_x_offset, global_y_offset):
        """트랙 ID와 크롭 영상을 큐에 등록 (offset은 전역 좌표 변환용)."""
        with self._lock:
            self.queue.append((tid, frame_crop, global_x_offset, global_y_offset, time.time()))

    def consume(self, tid):
        """해당 트랙의 최신 분석 결과 한 번 가져오고 제거."""
        with self._lock:
            return self.results.pop(tid, None)

    def _loop(self):
        """포즈 모델을 로드하고 큐에서 꺼내 키포인트 분석을 수행하는 워커."""
        try:
            model = YOLO(self.model_path)
            if self.gpu_id is not None and USE_CUDA:
                try:
                    model.to(f'cuda:{self.gpu_id}')
                    print(f"✅ [PoseEngine] 모델 로드 완료 (GPU {self.gpu_id})", flush=True)
                except Exception as e:
                    print(f"⚠️ [PoseEngine] GPU 할당 실패, CPU로 동작: {e}", flush=True)
            else:
                print(f"✅ [PoseEngine] 모델 로드 완료 (CPU)", flush=True)
            self.model_loaded = True
        except Exception as e:
            print(f"❌ [PoseEngine] 모델 로드 실패: {e}", flush=True)
            print(f"   → '{self.model_path}' 파일이 없으면 ultralytics가 자동 다운로드합니다", flush=True)
            self.running = False
            return

        NOSE = 0
        L_EYE, R_EYE = 1, 2
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        L_ANKLE, R_ANKLE = 15, 16

        device_arg = self.gpu_id if (self.gpu_id is not None and USE_CUDA) else 'cpu'

        while self.running:
            item = None
            with self._lock:
                if self.queue: item = self.queue.popleft()

            if item is None:
                time.sleep(0.01)
                continue

            tid, crop, gx_offset, gy_offset, req_time = item
            h, w = crop.shape[:2]

            default_result = {
                "state": "정상", "tilt": None, "g_nose_y": None,
                "g_body_cx": None, "g_body_cy": None,
                "n_visible": 0, "valid": False, "req_time": req_time
            }

            try:
                results = model(crop, verbose=False, max_det=1, imgsz=640, device=device_arg)
                if not results:
                    with self._lock:
                        self.results[tid] = default_result
                    continue
                res = results[0]

                kp_obj = getattr(res, "keypoints", None)
                if (kp_obj is None or kp_obj.xy is None or len(kp_obj.xy) == 0):
                    with self._lock:
                        self.results[tid] = default_result
                    continue

                kpts = kp_obj.xy[0].cpu().numpy()
                if kp_obj.conf is not None and len(kp_obj.conf) > 0:
                    confs = kp_obj.conf[0].cpu().numpy()
                else:
                    confs = np.ones(len(kpts), dtype=np.float32)

                if len(kpts) < 17 or len(confs) < 17:
                    with self._lock:
                        self.results[tid] = default_result
                    continue

                # 키포인트 보조 함수 (가시성/전역좌표/정규화좌표)
                def _vis(i): return confs[i] > POSE_KPT_CONF_THRESH
                def _gy(i): return gy_offset + kpts[i][1]
                def _gx(i): return gx_offset + kpts[i][0]
                def _ny(i): return kpts[i][1] / h if h > 0 else 0
                def _nx(i): return kpts[i][0] / w if w > 0 else 0

                n_visible = int((confs > POSE_KPT_CONF_THRESH).sum())

                if n_visible < POSE_MIN_KPT_VISIBLE:
                    self._debug_count += 1
                    if DEBUG_POSE and self._debug_count % 100 == 0:
                        print(f"  🤸 PoseEngine Track#{tid}: kpt_vis={n_visible}/17 (부족), crop={w}x{h}", flush=True)
                    with self._lock:
                        self.results[tid] = default_result
                    continue

                has_shoulder = _vis(L_SHOULDER) or _vis(R_SHOULDER)
                has_both_shoulders = _vis(L_SHOULDER) and _vis(R_SHOULDER)
                has_hip = _vis(L_HIP) or _vis(R_HIP)
                has_both_hips = _vis(L_HIP) and _vis(R_HIP)
                has_ankle = _vis(L_ANKLE) or _vis(R_ANKLE)
                has_knee = _vis(L_KNEE) or _vis(R_KNEE)
                has_nose = _vis(NOSE)

                if not (has_shoulder and has_hip):
                    self._debug_count += 1
                    if DEBUG_POSE and self._debug_count % 100 == 0:
                        print(f"  🤸 PoseEngine Track#{tid}: 어깨/엉덩이 미검출", flush=True)
                    with self._lock:
                        self.results[tid] = default_result
                    continue

                g_nose_y = _gy(NOSE) if has_nose else None

                # 어깨/엉덩이 중심 좌표 (정규화 및 전역)
                if has_both_shoulders:
                    sx_n = (_nx(L_SHOULDER) + _nx(R_SHOULDER)) / 2
                    sy_n = (_ny(L_SHOULDER) + _ny(R_SHOULDER)) / 2
                    sx_g = (_gx(L_SHOULDER) + _gx(R_SHOULDER)) / 2
                    sy_g = (_gy(L_SHOULDER) + _gy(R_SHOULDER)) / 2
                else:
                    s_idx = L_SHOULDER if _vis(L_SHOULDER) else R_SHOULDER
                    sx_n, sy_n = _nx(s_idx), _ny(s_idx)
                    sx_g, sy_g = _gx(s_idx), _gy(s_idx)

                if has_both_hips:
                    hipx_n = (_nx(L_HIP) + _nx(R_HIP)) / 2
                    hipy_n = (_ny(L_HIP) + _ny(R_HIP)) / 2
                    hipx_g = (_gx(L_HIP) + _gx(R_HIP)) / 2
                    hipy_g = (_gy(L_HIP) + _gy(R_HIP)) / 2
                else:
                    h_idx = L_HIP if _vis(L_HIP) else R_HIP
                    hipx_n, hipy_n = _nx(h_idx), _ny(h_idx)
                    hipx_g, hipy_g = _gx(h_idx), _gy(h_idx)

                g_body_cx = (sx_g + hipx_g) / 2
                g_body_cy = (sy_g + hipy_g) / 2

                # 어깨-엉덩이 기울기로 몸통 각도 산출
                dx1 = hipx_n - sx_n
                dy1 = hipy_n - sy_n
                tilt_sh = math.degrees(math.atan2(abs(dx1), abs(dy1) + 1e-6))

                tilt_ank = None
                if has_ankle:
                    if _vis(L_ANKLE) and _vis(R_ANKLE):
                        ax_n = (_nx(L_ANKLE) + _nx(R_ANKLE)) / 2
                        ay_n = (_ny(L_ANKLE) + _ny(R_ANKLE)) / 2
                    else:
                        a_idx = L_ANKLE if _vis(L_ANKLE) else R_ANKLE
                        ax_n, ay_n = _nx(a_idx), _ny(a_idx)
                    dx2 = ax_n - sx_n
                    dy2 = ay_n - sy_n
                    tilt_ank = math.degrees(math.atan2(abs(dx2), abs(dy2) + 1e-6))

                if tilt_ank is not None:
                    tilt = (tilt_sh + tilt_ank) / 2
                else:
                    tilt = tilt_sh

                state = "정상"

                # 기울기가 크고 머리-발 수직거리가 짧으면 "쓰러짐", 아니면 "넘어짐"
                if tilt > BODY_TILT_THRESH_DEG:
                    ankle_ys_n = []
                    if _vis(L_ANKLE): ankle_ys_n.append(_ny(L_ANKLE))
                    if _vis(R_ANKLE): ankle_ys_n.append(_ny(R_ANKLE))

                    if ankle_ys_n and has_nose:
                        avg_ankle_y = sum(ankle_ys_n) / len(ankle_ys_n)
                        nose_y_n = _ny(NOSE)
                        head_foot_diff = abs(avg_ankle_y - nose_y_n)

                        if head_foot_diff < HEAD_FOOT_FLAT_THRESH:
                            state = "쓰러짐"
                        else:
                            state = "넘어짐"
                    elif tilt > 70:
                        state = "쓰러짐"
                    else:
                        state = "넘어짐"

                self._debug_count += 1
                if DEBUG_POSE and (self._debug_count % 100 == 0 or state != "정상"):
                    tilt_str = f"{tilt:.1f}°"
                    print(f"  🤸 PoseEngine Track#{tid}: state={state}, tilt={tilt_str}, "
                          f"kpt={n_visible}/17, crop={w}x{h}", flush=True)

                with self._lock:
                    self.results[tid] = {
                        "state": state, "tilt": tilt, "g_nose_y": g_nose_y,
                        "g_body_cx": g_body_cx, "g_body_cy": g_body_cy,
                        "n_visible": n_visible, "valid": True, "req_time": req_time
                    }
            except Exception as e:
                if DEBUG_POSE:
                    self._debug_count += 1
                    if self._debug_count % 50 == 0:
                        print(f"  ⚠️ PoseEngine 예외 Track#{tid}: {e}", flush=True)
                with self._lock:
                    self.results[tid] = default_result

    def stop(self):
        """워커 종료 플래그 설정."""
        self.running = False


# ═════════════════════════════════════════════════════════════════
# HiResVerifyEngine — 안전모 위반 재검증을 별도 GPU/스레드에서 비동기 수행
# ═════════════════════════════════════════════════════════════════
class HiResVerifyEngine:
    """확대된 영상에 대해 YOLO를 한번 더 돌려 미착용 여부를 재확인."""

    def __init__(self, yolo_hires_model, gpu_id):
        """모델/큐/결과 dict 초기화 후 워커 시작."""
        self.model = yolo_hires_model
        self.gpu_id = gpu_id
        self._lock = threading.Lock()
        self.results = {}
        self.pending = set()
        self.queue = deque(maxlen=16)
        self.on = True
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, track_id, frame, bbox):
        """트랙의 재검증 요청을 큐에 추가 (중복 차단)."""
        with self._lock:
            if track_id in self.pending: return False
            if track_id in self.results: return False
            self.pending.add(track_id)
            self.queue.append((track_id, frame.copy(), bbox))
            return True

    def peek(self, track_id):
        """결과 존재 여부만 확인(소비하지 않음)."""
        with self._lock:
            return self.results.get(track_id)

    def consume(self, track_id):
        """결과를 꺼내고 제거."""
        with self._lock:
            r = self.results.pop(track_id, None)
            return r

    def has_result(self, track_id):
        """결과 보유 여부."""
        with self._lock:
            return track_id in self.results

    def is_pending(self, track_id):
        """요청 처리 중인지."""
        with self._lock:
            return track_id in self.pending

    def clear(self, track_id):
        """해당 트랙의 결과/대기 상태 모두 제거 (트랙 종료 시)."""
        with self._lock:
            self.results.pop(track_id, None)
            self.pending.discard(track_id)

    def _verify(self, frame, bbox):
        """확대 영상에서 YOLO를 돌려 헬멧 미착용(class 0) 여부 판정."""
        try:
            zoomed, nbbox = digital_zoom_crop(frame, bbox, scale=2)
            if zoomed is None or nbbox is None: return None
            results = list(self.model(zoomed, imgsz=1280, half=True,
                                       device=self.gpu_id, conf=0.30,
                                       stream=True, verbose=False))
            best_box = None
            best_iou = 0.0
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bb = (x1, y1, x2-x1, y2-y1)
                    iou_val = iou(bb, nbbox)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_box = (int(box.cls[0]), float(box.conf[0]))
            if best_box is None or best_iou < 0.20: return None
            return True if best_box[0] == 0 else False
        except Exception:
            return None

    def _loop(self):
        """큐에서 요청을 꺼내 _verify 실행 후 결과 저장."""
        while self.on:
            item = None
            with self._lock:
                if self.queue:
                    item = self.queue.popleft()
            if item is None:
                time.sleep(0.01)
                continue
            tid, frame, bbox = item
            decision = self._verify(frame, bbox)
            with self._lock:
                self.results[tid] = (decision, time.time())
                self.pending.discard(tid)

    def stop(self):
        """워커 종료 플래그 설정."""
        self.on = False


hires_engine = None


# ═════════════════════════════════════════════════════════════════
# 위반 기록 및 증거 저장
# ═════════════════════════════════════════════════════════════════
violations_lock = threading.Lock()
recent_violations = deque(maxlen=30)
VIOLATION_ID_COUNTER = [0]


def add_violation(name, vtype, image_url):
    """위반 1건을 recent_violations 큐에 추가하고 ID 반환."""
    VIOLATION_ID_COUNTER[0] += 1
    vid = VIOLATION_ID_COUNTER[0]
    with violations_lock:
        recent_violations.appendleft({
            "id": vid,
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "name": name,
            "type": vtype,
            "image_url": image_url
        })
    return vid


def _void_unknown_for_track(track, replaced_by=None):
    """미식별자로 기록되었다가 나중에 식별된 트랙의 기존 위반을 취소(void)."""
    if not track or not getattr(track, "unknown_violation_ids", None):
        return
    vids = list(track.unknown_violation_ids)
    track.unknown_violation_ids = []
    with violations_lock:
        kept = deque(maxlen=30)
        for v in recent_violations:
            if v.get("id") in vids:
                continue
            kept.append(v)
        recent_violations.clear()
        recent_violations.extend(kept)
    if NOTIFY_MAIN_SERVER:
        try:
            _notify_queue.put_nowait({
                "__void__": True,
                "track_id": track.id,
                "logged_at_unix": track.unknown_logged_at if track.unknown_logged_at else time.time(),
                "replaced_by": replaced_by or (track.locked_name or track.ever_identified_as or ""),
            })
        except queue.Full:
            pass
    print(f"  🧹 Track#{track.id}: 이전 미식별자 위반 {len(vids)}건 void (확정 → {replaced_by})",
          flush=True)


def save_single_evidence(track, name, reason="안전모 미착용", motion_frame=None, motion_bbox=None):
    """위반 증거 이미지를 파일로 저장하고 웹에서 접근 가능한 경로 반환."""
    is_motion = reason in ("쓰러짐", "넘어짐", "떨어짐")

    if is_motion and motion_frame is not None:
        frame = motion_frame
        bbox = motion_bbox if motion_bbox is not None else track.bbox
    elif track.best_evidence is not None:
        frame, bbox, _quality = track.best_evidence
    elif motion_frame is not None:
        frame = motion_frame
        bbox = motion_bbox if motion_bbox is not None else track.bbox
    else:
        return None

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    day = datetime.date.today().isoformat()
    safe_name = "".join(c for c in name if c.isalnum() or c in "_-") or "unknown"

    if is_motion:
        d = os.path.join(EVIDENCE_DIR, "MotionAlerts", safe_name, day)
    else:
        d = os.path.join(EVIDENCE_DIR, safe_name, day)
    os.makedirs(d, exist_ok=True)

    try:
        hx, hy, hw, hh = bbox
        fh, fw = frame.shape[:2]

        # 동작 위반은 전신을 더 넓게, 안전모 위반은 상반신 위주로 크롭
        if is_motion:
            y1 = max(0, hy - hh)
            y2 = min(fh, hy + hh * 5)
            x1 = max(0, hx - hw * 3)
            x2 = min(fw, hx + hw * 4)
        else:
            y1 = max(0, hy - hh//2)
            y2 = min(fh, hy + int(hh*4))
            x1 = max(0, hx - hw)
            x2 = min(fw, hx + hw*2)

        filename = f"{ts}_{safe_name}_{reason}.jpg" if is_motion else f"{ts}_{safe_name}.jpg"
        p = os.path.join(d, filename)
        person = frame[y1:y2, x1:x2]

        # 위반 종류별 색상/테두리/라벨
        if reason == "떨어짐":
            display_text = f"⚠️ 추락 감지: {name}"
            color = (0, 0, 255)
            border = 6
        elif reason == "쓰러짐":
            display_text = f"🚨 쓰러짐 감지: {name}"
            color = (0, 0, 255)
            border = 6
        elif reason == "넘어짐":
            display_text = f"⚠️ 넘어짐 감지: {name}"
            color = (0, 80, 255)
            border = 5
        else:
            if name == "unknown" or name == "미식별자":
                display_text = "⚠️ 미식별자 - NO HELMET"
                color = (0, 140, 255)
                border = 5
            else:
                display_text = f"NO HELMET: {name}"
                color = (0, 0, 255)
                border = 4

        if person.size > 0 and person.shape[0] > 50 and person.shape[1] > 30:
            cropped = frame[y1:y2, x1:x2].copy()
            rel_x, rel_y = hx-x1, hy-y1
            cv2.rectangle(cropped, (rel_x, rel_y), (rel_x+hw, rel_y+hh), color, border)

            if is_motion:
                # 동작 위반은 상단 배너 + 하단 타임스탬프
                banner_h = 50
                overlay = cropped.copy()
                cv2.rectangle(overlay, (0, 0), (cropped.shape[1], banner_h), color, -1)
                cv2.addWeighted(overlay, 0.75, cropped, 0.25, 0, cropped)
                put_text_kr(cropped, display_text, (10, banner_h - 10),
                            font_size=28, color=(255, 255, 255))
                time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                put_text_kr(cropped, time_str, (10, cropped.shape[0] - 10),
                            font_size=18, color=(255, 255, 0))
            else:
                put_text_kr(cropped, display_text, (rel_x, max(20, rel_y-8)),
                            font_size=20, color=color)
            safe_imwrite(p, cropped)
        else:
            annotated = frame.copy()
            cv2.rectangle(annotated, (hx,hy), (hx+hw,hy+hh), color, border)
            put_text_kr(annotated, display_text, (hx, max(20,hy-8)),
                        font_size=22, color=color)
            safe_imwrite(p, annotated)

        if is_motion:
            return f"/evidence/MotionAlerts/{safe_name}/{day}/{filename}"
        else:
            return f"/evidence/{safe_name}/{day}/{filename}"
    except Exception as e:
        if DEBUG_POSE:
            print(f"  ⚠️ save_single_evidence 예외: {e}", flush=True)
        return None


def try_log_violation(track, name, frame, now, tracker_ref, yolo_hires_ref=None, reason="안전모 미착용"):
    """위반 시도 — HiRes 재검증, 쿨다운 통과 시 증거 저장 + 알림 전송."""
    effective_name = name if name != "unknown" else None
    if not effective_name:
        effective_name = track.locked_name or track.ever_identified_as
    embedding = track.last_good_embedding
    bbox = track.bbox

    is_motion = reason in ("쓰러짐", "넘어짐", "떨어짐")

    # 미식별자: 같은 사람이 화면에 남아있는 동안에는 한 번만 기록
    if not is_motion and not effective_name:
        if track.unknown_person_id is None:
            track.unknown_person_id = unknown_registry.assign_id(embedding, now)
        if unknown_registry.is_currently_logged(track.unknown_person_id, now):
            track.person_logged = True
            track.unknown_logged = True
            if DEBUG_MATCH:
                print(f"  🔁 미식별자 중복차단: Track#{track.id} → "
                      f"{track.unknown_person_id} (현재 화면에 있으며 이미 로그됨)",
                      flush=True)
            return False

    # 안전모 위반은 HiRes로 한번 더 검증 (비동기 or 동기)
    if (reason == "안전모 미착용" and USE_HIRES_VERIFY
            and yolo_hires_ref is not None and track.best_evidence is not None):
        if HIRES_ASYNC and hires_engine is not None:
            res = hires_engine.peek(track.id)
            if res is None:
                if not hires_engine.is_pending(track.id):
                    hires_engine.submit(track.id, track.best_evidence[0], track.bbox)
                return False
            else:
                decision, _ts = res
                hires_engine.consume(track.id)
                if decision is False:
                    return False
        else:
            hr = verify_violation_hires(track.best_evidence[0], track.bbox)
            if hr is False:
                return False

    # 쿨다운 체크
    if is_motion:
        if not violation_guard.try_acquire_motion(effective_name, embedding, bbox, now, POSE_LOG_COOLDOWN):
            return False
    else:
        if not violation_guard.try_acquire(effective_name, embedding, bbox, now):
            track.person_logged = True
            if not effective_name:
                track.unknown_logged = True
            return False

    if effective_name:
        display_name = effective_name
    else:
        display_name = track.unknown_person_id or "미식별자"

    if not is_motion and track.best_evidence is None:
        track.force_update_evidence(frame)

    if is_motion:
        ev_path = save_single_evidence(track, effective_name or "unknown", reason,
                                        motion_frame=frame, motion_bbox=bbox)
    else:
        ev_path = save_single_evidence(track, effective_name or "unknown", reason)

    if reason == "안전모 미착용":
        track.person_logged = True
        track.unknown_logged_at = now
        if not effective_name:
            track.unknown_logged = True
            if track.unknown_person_id:
                unknown_registry.mark_logged(track.unknown_person_id, now, embedding)

    if effective_name and not is_motion:
        tracker_ref.mark_logged(effective_name, now)
    else:
        if USE_RECENT_UNKNOWN_CACHE and embedding is not None and not is_motion:
            recent_unknown_cache.add(embedding, now)
    if ev_path:
        vid = add_violation(display_name, reason, ev_path)
        if (not effective_name) and reason == "안전모 미착용":
            try:
                track.unknown_violation_ids.append(vid)
            except Exception:
                pass
        print(f"  📸 위반 기록: {display_name} ({reason}) → {ev_path}", flush=True)
        notify_main_server(display_name, reason, ev_path, track_id=track.id)
    else:
        if DEBUG_MATCH:
            print(f"  ⚠️ 위반 기록 실패 (evidence 생성 불가): {display_name}", flush=True)
    return True


# ═════════════════════════════════════════════════════════════════
# Track — 추적 중인 한 사람의 모든 상태 보관
# ═════════════════════════════════════════════════════════════════
class Track:
    """한 명의 추적 대상에 대한 위치/속도/얼굴/투표/자세 등 전체 상태."""

    def __init__(self, tid, bbox, now):
        """초기 위치로 트랙 생성. 모든 식별/투표/자세 상태를 기본값으로 셋업."""
        self.id = tid
        self.bbox = bbox
        self.vx = 0.0
        self.vy = 0.0
        self.first_seen = now
        self.last_seen = now
        self.miss = 0
        self.hits = 1
        self.age_frames = 1
        self.name = "unknown"
        self.score = 0.0
        self.face = 0.0
        self.confirmed = False
        self.identified_at = 0.0
        self.unknown_logged_at = 0.0
        self.person_logged = False
        self.ever_identified_as = None
        self.identification_abandoned = False
        self.helmet_history = deque(maxlen=HELMET_HISTORY_LEN)
        self.face_candidates = []
        self.last_request = 0.0
        self.check_count = 0
        self.response_count = 0
        self.successful_extractions = 0
        self.no_face_count = 0
        self.best_evidence = None
        self.evidence_saved = False
        self.suspect_no_helmet_count = 0
        self.last_zoomed_at = 0.0
        self.vote_history = []
        self.last_verify_at = 0.0
        self.disagree_streak = 0
        self.confirm_score = 0.0
        self.locked_name = None
        self.marked_for_kill = False
        self.last_good_embedding = None
        self.cache_lookup_done = False
        self.consecutive_unknown_with_face = 0
        self.unknown_cache_checked = False

        self.unknown_person_id = None
        self.unknown_logged = False

        # 트랙 swap 감지용 anchor 임베딩
        self.anchor_embedding = None
        self.anchor_set_at = 0.0
        self.swap_suspect_count = 0
        self.last_swap_suspect_at = 0.0
        self.unknown_violation_ids = []

        # 자세 상태
        self.pose_state = "정상"
        self.pose_pending_state = "정상"
        self.last_pose_check = 0.0
        self.pose_state_history = deque(maxlen=POSE_HISTORY_LEN)
        self.pose_event_started_at = 0.0
        self.pose_detected_at = 0.0
        self.pose_logged_at = 0.0
        self.tilt_deg = None
        self.tilt_history = deque(maxlen=POSE_HISTORY_LEN)
        self.nose_y_history = deque(maxlen=8)
        self.body_cy_history = deque(maxlen=8)
        self.pose_time_history = deque(maxlen=8)
        self.last_valid_pose_at = 0.0

    def predict(self):
        """단순 등속 모델로 다음 프레임 위치 예측."""
        return (int(self.bbox[0]+self.vx), int(self.bbox[1]+self.vy),
                self.bbox[2], self.bbox[3])

    def update_bbox(self, new_bbox, now):
        """새 검출과 매칭 — 속도 갱신, miss 리셋, hits 증가."""
        self.vx = self.vx*0.5 + (new_bbox[0]-self.bbox[0])*0.5
        self.vy = self.vy*0.5 + (new_bbox[1]-self.bbox[1])*0.5
        self.bbox = new_bbox
        self.last_seen = now
        self.miss = 0
        self.hits += 1

    def update_helmet(self, has_helmet):
        """안전모 착용 여부를 히스토리에 push, 의심 카운트 갱신."""
        self.helmet_history.append(bool(has_helmet))
        self.suspect_no_helmet_count = ((self.suspect_no_helmet_count+1)
                                        if not has_helmet
                                        else max(0, self.suspect_no_helmet_count-1))

    def box_size(self):
        """박스의 큰 쪽 변 길이 — 거리 판단에 사용."""
        return max(self.bbox[2], self.bbox[3])

    def is_far(self):
        """원거리 트랙 판정."""
        return self.box_size() < FAR_BOX_THRESHOLD

    def is_fast_suspect(self):
        """짧은 시간에 미착용이 일정 횟수 잡혔는지 (빠른 의심 판정)."""
        return sum(1 for x in self.helmet_history if not x) >= HELMET_VIOLATION_FAST

    def is_violation(self):
        """위반 확정 — 히스토리 길이와 미착용 횟수 모두 충족."""
        need = HELMET_VIOLATION_FAR if self.is_far() else HELMET_VIOLATION_NEED
        return (len(self.helmet_history) >= need and
                sum(1 for x in self.helmet_history if not x) >= need)

    def is_stable(self):
        """일정 횟수 매칭되어 안정화된 트랙인지 (한 번 안정되면 유지)."""
        if getattr(self, '_ever_stable', False):
            return True
        if self.hits >= MIN_TRACK_HITS and self.age_frames >= MIN_TRACK_AGE:
            self._ever_stable = True
            return True
        return False

    def add_candidate(self, frame, bbox, now):
        """얼굴 추출용 크롭(여유 padding 포함)을 후보 목록에 추가."""
        hx, hy, hw, hh = bbox
        fh, fw = frame.shape[:2]

        # 원거리/근거리에 따라 다른 padding 비율 적용
        is_far_local = hw < FAR_FACE_THRESHOLD_PX
        if is_far_local:
            pad_l = 2.0; pad_r = 3.0; pad_t = 1.5; pad_b = 3.5
        else:
            pad_l = 1.5; pad_r = 2.5; pad_t = 1.0; pad_b = 3.0

        px1 = max(0, hx - int(hw * pad_l))
        py1 = max(0, hy - int(hh * pad_t))
        px2 = min(fw, hx + int(hw * pad_r))
        py2 = min(fh, hy + int(hh * pad_b))

        if px2 - px1 < 20 or py2 - py1 < 20: return

        large_crop = frame[py1:py2, px1:px2].copy()
        gray = cv2.cvtColor(large_crop, cv2.COLOR_BGR2GRAY)
        sharpness = laplacian_sharpness(gray)

        if is_far_local:
            min_sharp = EMB_MIN_SHARPNESS_FAR * 0.5
        elif hw < 50:
            min_sharp = EMB_MIN_SHARPNESS_NEAR * 0.5
        else:
            min_sharp = EMB_MIN_SHARPNESS_NEAR

        if sharpness < min_sharp: return

        # 크기 35% + 선명도 65%로 품질 점수 산출
        quality = (min((px2-px1)*(py2-py1)/10000, 1.0)*0.35 + min(sharpness/250, 1.0)*0.65)
        local_bbox = (hx - px1, hy - py1, hw, hh)
        self.face_candidates.append((large_crop, local_bbox, quality, now, hw))
        self.face_candidates.sort(key=lambda x: -x[2])
        self.face_candidates = self.face_candidates[:BEST_SHOT_MAX]

    def prune_candidates(self, now):
        """오래된 후보 제거."""
        self.face_candidates = [c for c in self.face_candidates
                                if (now-c[3]) < CANDIDATE_KEEP_TIME]

    def maybe_update_evidence(self, frame, has_face_candidate):
        """증거 영상이 아직 없으면 첫 적합 시점에 한 번만 저장."""
        if self.best_evidence is not None: return
        hx, hy, hw, hh = self.bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, hx), max(0, hy)
        x2, y2 = min(fw, hx+hw), min(fh, hy+hh)
        if x2-x1 < 10 or y2-y1 < 10: return

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        quality = (min((x2-x1)*(y2-y1)/15000, 1.0)*0.4
                 + min(laplacian_sharpness(gray)/250, 1.0)*0.3
                 + (0.3 if has_face_candidate else 0.0))
        self.best_evidence = (frame.copy(), self.bbox, quality)

    def force_update_evidence(self, frame):
        """현재 프레임을 강제로 증거 영상으로 덮어쓰기."""
        self.best_evidence = (frame.copy(), self.bbox, 1.0)

    def no_helmet_count(self):
        """히스토리에서 미착용으로 잡힌 횟수."""
        return sum(1 for x in self.helmet_history if not x)

    def process_pose_result(self, pose_res, frame_h, now):
        """PoseEngine 결과를 받아 떨어짐/쓰러짐/넘어짐 상태를 확정 → 이벤트 반환."""
        if pose_res is None:
            return None

        state = pose_res.get("state", "정상")
        tilt = pose_res.get("tilt")
        g_nose_y = pose_res.get("g_nose_y")
        g_body_cy = pose_res.get("g_body_cy")
        valid = pose_res.get("valid", False)
        req_time = pose_res.get("req_time", now)

        if not valid:
            return None

        self.last_valid_pose_at = now
        self.tilt_deg = tilt
        if tilt is not None:
            self.tilt_history.append(tilt)

        if g_nose_y is not None:
            self.nose_y_history.append(g_nose_y)
            self.pose_time_history.append(req_time)
        if g_body_cy is not None:
            self.body_cy_history.append(g_body_cy)

        # 떨어짐 감지 — 머리·몸통이 함께 빠르게 하강해야 인정 (고개숙임 오탐 방지)
        fall_detected = False
        if len(self.nose_y_history) >= 3 and len(self.pose_time_history) >= 3:
            dt = self.pose_time_history[-1] - self.pose_time_history[-3]
            if dt > 0.05:
                vel_nose = (self.nose_y_history[-1] - self.nose_y_history[-3]) / dt
                vel_body = None
                if len(self.body_cy_history) >= 3:
                    vel_body = (self.body_cy_history[-1] - self.body_cy_history[-3]) / dt

                thr = frame_h * FALL_VELOCITY_THRESH
                if vel_nose > thr:
                    if vel_body is None or vel_body > thr * 0.5:
                        fall_detected = True
                        if DEBUG_POSE:
                            body_str = f"{vel_body:.0f}" if vel_body is not None else "N/A"
                            print(f"  ⚠️ Track#{self.id}: 떨어짐 감지 "
                                  f"(nose_vel={vel_nose:.0f}, body_vel={body_str}, "
                                  f"thr={thr:.0f}px/s)", flush=True)

        if fall_detected:
            state = "떨어짐"

        self.pose_state_history.append((state, now))

        recent_results = list(self.pose_state_history)
        abnormal_results = [(s, t) for s, t in recent_results if s != "정상"]
        normal_count = len(recent_results) - len(abnormal_results)

        if not abnormal_results:
            self.pose_event_started_at = 0.0
            self.pose_pending_state = "정상"

            if self.pose_state != "정상" and (now - self.pose_detected_at) > POSE_DISPLAY_DURATION:
                self.pose_state = "정상"
            return None

        state_counts = Counter(s for s, _ in abnormal_results)
        dominant_state, dominant_count = state_counts.most_common(1)[0]

        if self.pose_event_started_at == 0.0:
            self.pose_event_started_at = abnormal_results[0][1]

        duration = now - self.pose_event_started_at
        ratio = len(abnormal_results) / max(1, len(recent_results))

        self.pose_pending_state = dominant_state

        # 떨어짐은 즉시 확정
        if state == "떨어짐":
            self.pose_state = "떨어짐"
            self.pose_detected_at = now
            return "떨어짐"

        # 쓰러짐/넘어짐은 일정 프레임/지속시간/비율 충족 시 확정
        if (dominant_count >= POSE_CONFIRM_FRAMES and
            duration >= POSE_MIN_DURATION and
            ratio >= 0.6):
            if self.pose_state != dominant_state:
                if DEBUG_POSE:
                    print(f"  🚨 Track#{self.id}: 동작 확정 → {dominant_state} "
                          f"(count={dominant_count}, dur={duration:.2f}s, ratio={ratio:.0%})",
                          flush=True)
                self.pose_state = dominant_state
                self.pose_detected_at = now
                return dominant_state

        return None

    def add_vote(self, name, score, quality=1.0, margin=0.0, now=None):
        """워커 결과 1건을 투표 히스토리에 누적 (윈도우 안의 것만 유지)."""
        if now is None: now = time.time()
        if name == "unknown":
            return
        self.vote_history.append((name, score, quality, margin, now))
        self.vote_history = [v for v in self.vote_history if now - v[4] < FACE_VOTE_WINDOW]
        if len(self.vote_history) > 25:
            self.vote_history = self.vote_history[-25:]

    def get_majority_vote(self, person_thresholds=None):
        """투표 히스토리로 다수결 후보를 산출. 강한 점수는 빠른 경로로 확정."""
        total_known = len(self.vote_history)
        if total_known == 0:
            return None, 0.0, 0.0

        name_info = {}
        for n, s, q, m, _ts in self.vote_history:
            if n not in name_info:
                name_info[n] = {"count": 0, "weighted": 0.0, "scores": [], "margins": []}
            name_info[n]["count"] += 1
            name_info[n]["weighted"] += q
            name_info[n]["scores"].append(s)
            name_info[n]["margins"].append(m)

        best_name = max(name_info, key=lambda n: name_info[n]["weighted"])
        info = name_info[best_name]
        ratio = info["count"] / total_known
        avg_score = sum(info["scores"]) / len(info["scores"])
        avg_margin = sum(info["margins"]) / len(info["margins"]) if info["margins"] else 0.0

        person_thr = (person_thresholds or {}).get(best_name, FACE_THR_INSIGHT)
        person_vote_avg_min = max(0.45, person_thr * 0.93)

        # 빠른 경로 — 한두 표여도 점수가 충분히 강하면 확정
        if total_known == 1 and avg_score >= (FACE_STRONG_ABS + 0.06):
            return best_name, avg_score, ratio
        if total_known == 2 and ratio == 1.0 and avg_score >= FACE_STRONG_ABS:
            return best_name, avg_score, ratio
        if total_known == 2 and ratio == 1.0 and avg_score >= person_thr:
            return best_name, avg_score, ratio

        if total_known < FACE_VOTE_MIN:
            return None, avg_score, ratio

        if avg_score >= person_vote_avg_min and ratio >= FACE_VOTE_RATIO:
            return best_name, avg_score, ratio

        if total_known >= 3 and ratio >= FACE_VOTE_STRONG and avg_score >= person_thr:
            return best_name, avg_score, ratio

        # 마진 폴백 — 점수가 낮아도 다른 후보와 충분히 벌어졌다면 인정
        if (total_known >= 4
                and ratio >= 0.85
                and avg_margin >= FACE_VOTE_MARGIN_MIN + 0.05
                and avg_score >= 0.45):
            if DEBUG_MATCH:
                print(f"  🎯 마진폴백 확정: {best_name} avg={avg_score:.3f} m={avg_margin:.3f} "
                      f"(thr={person_thr:.2f}, ratio={ratio:.0%})")
            return best_name, avg_score, ratio

        return None, avg_score, ratio


# ═════════════════════════════════════════════════════════════════
# Tracker — 검출 박스를 트랙에 할당 (Hungarian, swap 방지)
# ═════════════════════════════════════════════════════════════════
class Tracker:
    """전체 트랙 목록과 검출↔트랙 매칭, kill 처리 담당."""

    def __init__(self):
        """트랙 리스트와 인물별 마지막 본/로그 시간 초기화."""
        self.tracks = []
        self.next_id = 1
        self.person_last_seen = {}
        self.person_last_logged = {}
        self.kill_zones = []

    def _match(self, det, prefer_confirmed=None):
        """greedy 폴백용 — 검출 1개에 가장 잘 맞는 트랙 인덱스 반환."""
        best_i, best_score = None, -1
        d_cx, d_cy = det[0]+det[2]/2, det[1]+det[3]/2
        d_size = max(det[2], det[3])

        for i, t in enumerate(self.tracks):
            if t.marked_for_kill: continue
            if prefer_confirmed is True and not t.confirmed: continue
            if prefer_confirmed is False and t.confirmed: continue

            iou_val = max(iou(det, t.bbox), iou(det, t.predict()))
            t_cx, t_cy = t.bbox[0]+t.bbox[2]/2, t.bbox[1]+t.bbox[3]/2
            dist = np.sqrt((d_cx-t_cx)**2 + (d_cy-t_cy)**2)

            if t.confirmed:
                iou_thr = CONFIRMED_TRACK_IOU_THR
                dist_ratio = CONFIRMED_TRACK_DIST_RATIO
            else:
                iou_thr = TRACK_IOU_THR
                dist_ratio = TRACK_DIST_RATIO

            if iou_val >= iou_thr or dist < d_size * dist_ratio:
                combined = iou_val + (1.0 - min(dist/(d_size*2), 1.0)) * 0.3
                if combined > best_score:
                    best_score, best_i = combined, i
        return best_i

    # Hungarian 매칭에서 게이팅 탈락 쌍에 부여하는 큰 비용
    _COST_BLOCK = 1e6
    _COST_GATE  = 1e5

    def _pair_cost(self, t, det):
        """(track, det) 쌍의 비용 — IoU 클수록·거리 작을수록 낮음. 게이팅 미통과 시 _COST_BLOCK."""
        if t.marked_for_kill:
            return self._COST_BLOCK

        t_pred = t.predict()
        iou_val = max(iou(det, t.bbox), iou(det, t_pred))

        d_cx = det[0] + det[2] / 2.0
        d_cy = det[1] + det[3] / 2.0
        t_cx = t_pred[0] + t_pred[2] / 2.0
        t_cy = t_pred[1] + t_pred[3] / 2.0
        d_size = max(det[2], det[3]) + 1e-6
        dist_norm = math.hypot(d_cx - t_cx, d_cy - t_cy) / d_size

        if t.confirmed:
            iou_thr    = CONFIRMED_TRACK_IOU_THR
            dist_ratio = CONFIRMED_TRACK_DIST_RATIO
        else:
            iou_thr    = TRACK_IOU_THR
            dist_ratio = TRACK_DIST_RATIO

        if iou_val < iou_thr and dist_norm > dist_ratio:
            return self._COST_BLOCK

        cost = (1.0 - iou_val) + 0.3 * min(dist_norm, 2.0)
        if t.confirmed:
            cost -= 0.15  # 확정 트랙 보호
        return cost

    def update(self, detections, now):
        """검출들을 트랙에 할당하고, 매칭 안 된 검출은 신규 트랙으로 생성."""
        for t in self.tracks:
            t.miss += 1
            t.age_frames += 1

        matched = {}
        matched_track_ids = {}
        used_d = set()

        if detections and self.tracks:
            N = len(self.tracks)
            M = len(detections)

            if _HUNGARIAN_OK:
                # Hungarian — 전체 비용 합이 최소가 되도록 일괄 할당
                cost = np.full((N, M), self._COST_BLOCK, dtype=np.float32)
                for i, t in enumerate(self.tracks):
                    for j, det in enumerate(detections):
                        cost[i, j] = self._pair_cost(t, det)

                try:
                    row_ind, col_ind = linear_sum_assignment(cost)
                except Exception:
                    row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

                for i, j in zip(row_ind, col_ind):
                    if cost[i, j] >= self._COST_GATE:
                        continue
                    self.tracks[i].update_bbox(detections[j], now)
                    matched_track_ids[j] = self.tracks[i].id
                    used_d.add(j)
            else:
                # 폴백 — confirmed 우선 greedy 두 패스
                used_t = set()
                for d_idx, det in enumerate(detections):
                    mi = self._match(det, prefer_confirmed=True)
                    if mi is not None and mi not in used_t:
                        self.tracks[mi].update_bbox(det, now)
                        used_t.add(mi)
                        matched_track_ids[d_idx] = self.tracks[mi].id
                        used_d.add(d_idx)
                for d_idx, det in enumerate(detections):
                    if d_idx in used_d:
                        continue
                    mi = self._match(det, prefer_confirmed=False)
                    if mi is not None and mi not in used_t:
                        self.tracks[mi].update_bbox(det, now)
                        used_t.add(mi)
                        matched_track_ids[d_idx] = self.tracks[mi].id
                        used_d.add(d_idx)

        # 매칭 안 된 검출은 신규 트랙
        for d_idx, det in enumerate(detections):
            if d_idx in used_d:
                continue
            new_track = Track(self.next_id, det, now)
            self.tracks.append(new_track)
            matched_track_ids[d_idx] = new_track.id
            self.next_id += 1

        # 오래된 트랙/kill 표시된 트랙 제거
        lost = [t for t in self.tracks
                if t.miss > TRACK_MAX_MISS or t.marked_for_kill]
        self.tracks = [t for t in self.tracks
                       if t.miss <= TRACK_MAX_MISS and not t.marked_for_kill]
        self.kill_zones = [(b, ex) for b, ex in self.kill_zones if ex > now]

        # det_idx → 최종 tracks 리스트의 위치로 재인덱싱
        id_to_idx = {t.id: i for i, t in enumerate(self.tracks)}
        for d_idx, tid in matched_track_ids.items():
            if tid in id_to_idx:
                matched[d_idx] = id_to_idx[tid]

        return matched, lost

    def add_kill_zone(self, bbox, now):
        """swap 직후 같은 위치에서 새 트랙이 곧바로 생기는 것을 잠시 막음."""
        self.kill_zones.append((bbox, now + KILL_COOLDOWN_AFTER_SWAP))

    def remove_ids(self, ids):
        """주어진 ID 집합에 해당하는 트랙들을 강제 제거."""
        if ids: self.tracks = [t for t in self.tracks if t.id not in ids]

    def mark_seen(self, name, now):
        """해당 이름을 방금 봤음을 기록."""
        self.person_last_seen[name] = now

    def can_log(self, key, now, cd=PERSON_LOG_COOLDOWN):
        """해당 키의 마지막 로그가 쿨다운을 지났는지."""
        return (now - self.person_last_logged.get(key, 0)) >= cd

    def mark_logged(self, key, now):
        """해당 키의 마지막 로그 시각 기록."""
        self.person_last_logged[key] = now


# ═════════════════════════════════════════════════════════════════
# Worker — 얼굴 후보를 임베딩으로 변환하고 DB와 매칭
# ═════════════════════════════════════════════════════════════════
class Worker:
    """별도 스레드 풀에서 얼굴 임베딩 추출과 DB 매칭을 비동기로 수행."""

    def __init__(self, db, n=8):
        """DB 참조 보관 후 워커 스레드 n개 시작."""
        self.db = db
        self.q = deque(maxlen=24)
        self.r = {}
        self.lock = threading.Lock()
        self.on = True
        self.in_flight = set()
        self.threads = [self._start() for _ in range(n)]
        print(f"⚙️ Worker {n}개 시작")

    def _start(self):
        """워커 스레드 1개 생성/시작."""
        t = threading.Thread(target=self._w, daemon=True)
        t.start()
        return t

    def add(self, tid, candidates):
        """트랙 ID의 얼굴 후보를 큐에 추가 (중복은 무시)."""
        with self.lock:
            for t, _ in self.q:
                if t == tid: return False
            if tid in self.in_flight: return False
            if len(self.q) >= 24: self.q.pop()
            self.q.appendleft((tid, list(candidates)))
            return True

    def pop(self, tid):
        """해당 트랙의 결과를 꺼냄 (없으면 None)."""
        with self.lock:
            return self.r.pop(tid, None)

    def check(self):
        """죽은 워커 스레드를 재시작."""
        for i, t in enumerate(self.threads):
            if not t.is_alive():
                self.threads[i] = self._start()

    def _extract_from_crop(self, large_crop, local_bbox, face_size_hint):
        """후보 크롭에서 임베딩 추출 — 실패 시 더 타이트한 영역으로 재시도."""
        emb, st = extract_face_embedding(large_crop, face_size_hint=face_size_hint)
        if emb is not None: return emb, st

        hx, hy, hw, hh = local_bbox
        lh, lw = large_crop.shape[:2]
        x1 = max(0, hx - int(hw*0.30))
        y1 = max(0, hy - int(hh*0.15))
        x2 = min(lw, hx+hw + int(hw*0.30))
        y2 = min(lh, hy+hh + int(hh*1.00))

        if x2-x1 >= 20 and y2-y1 >= 20:
            tight = large_crop[y1:y2, x1:x2]
            emb2, st2 = extract_face_embedding(tight, face_size_hint=face_size_hint)
            return emb2, st2
        return None, st

    def _w(self):
        """워커 루프 — 후보 추출 → 개별 매칭 → 이름별 투표 → 평균 임베딩 폴백 매칭."""
        while self.on:
            item = None
            with self.lock:
                if self.q:
                    item = self.q.popleft()
                    if item is not None:
                        self.in_flight.add(item[0])
            if not item:
                time.sleep(0.005)
                continue

            tid = item[0]
            try:
                _, cands = item
                cands = cands[:WORKER_MAX_CANDIDATES_PER_CALL]

                embeddings = []
                qualities = []
                statuses = []

                for cand in cands:
                    if len(cand) >= 5:
                        large_crop, local_bbox, q, _t, hw_hint = cand
                    else:
                        large_crop, local_bbox, q, _t = cand
                        hw_hint = local_bbox[2]

                    emb, st = self._extract_from_crop(large_crop, local_bbox, hw_hint)
                    statuses.append(st)

                    if emb is not None:
                        embeddings.append(emb)
                        qualities.append(q)
                        if len(embeddings) >= EMB_MULTI_EXTRACT: break

                if not embeddings:
                    overall_status = (ExtractStatus.NO_FACE
                                      if all(s == ExtractStatus.NO_FACE for s in statuses)
                                      else ExtractStatus.POOR_QUALITY)
                    if DEBUG_MATCH:
                        print(f"  👤 Track#{tid}: 추출 실패 [{overall_status}]", flush=True)
                    with self.lock:
                        self.r[tid] = ("unknown", 0.0, 0.0, False, 0.0,
                                       overall_status, None, 0.0)
                        self.in_flight.discard(tid)
                    continue

                # 개별 임베딩별로 매칭한 뒤 이름별로 표 집계
                votes_per_name = {}
                score_per_name = {}
                margin_per_name = {}
                for emb, qw in zip(embeddings, qualities):
                    fn, fs, fm, is_confused = match_face(emb, self.db)
                    if fn != "unknown" and not is_confused:
                        votes_per_name[fn] = votes_per_name.get(fn, 0) + 1
                        if fn not in score_per_name or fs > score_per_name[fn]:
                            score_per_name[fn] = fs
                            margin_per_name[fn] = fm

                # 품질 가중 평균 임베딩
                if len(embeddings) == 1:
                    avg_emb = embeddings[0]
                    avg_q = qualities[0]
                else:
                    weights = np.array(qualities, dtype=np.float32) + 1e-3
                    weights /= weights.sum()
                    avg_emb = sum(e * w for e, w in zip(embeddings, weights))
                    avg_q = float(np.mean(qualities))
                avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)

                # 다수 이름이 있으면 그 결과를 우선
                if votes_per_name:
                    winner = max(votes_per_name, key=lambda k: (votes_per_name[k], score_per_name[k]))
                    win_count = votes_per_name[winner]
                    if win_count * 2 >= len(embeddings):
                        fn2, fs2, fm2, conf2 = match_face(avg_emb, self.db)
                        if fn2 == winner and not conf2:
                            with self.lock:
                                self.r[tid] = (winner, max(fs2, score_per_name[winner]),
                                               max(fs2, score_per_name[winner]),
                                               False, avg_q, ExtractStatus.OK, avg_emb,
                                               max(fm2, margin_per_name.get(winner, 0)))
                                self.in_flight.discard(tid)
                            continue
                        elif fn2 != "unknown" and fn2 != winner:
                            with self.lock:
                                self.r[tid] = ("unknown", 0.0, 0.0, True, avg_q,
                                               ExtractStatus.OK, avg_emb, 0.0)
                                self.in_flight.discard(tid)
                            continue
                        else:
                            with self.lock:
                                self.r[tid] = (winner, score_per_name[winner],
                                               score_per_name[winner], False, avg_q,
                                               ExtractStatus.OK, avg_emb,
                                               margin_per_name.get(winner, 0))
                                self.in_flight.discard(tid)
                            continue

                # 폴백 — 평균 임베딩으로 한 번 더 매칭
                fn, fs, fm, is_confused = match_face(avg_emb, self.db)
                n, c, fc_ = decide_face_only(fn, fs, fm, is_confused)
                with self.lock:
                    self.r[tid] = (n, c, fc_, is_confused, avg_q,
                                   ExtractStatus.OK, avg_emb, fm)
                    self.in_flight.discard(tid)
            except Exception as _e:
                import traceback
                print(f"  💥 Worker 예외 (tid={tid}): {_e}", flush=True)
                traceback.print_exc()
                with self.lock:
                    self.r[tid] = ("unknown", 0.0, 0.0, False, 0.0,
                                   ExtractStatus.NO_FACE, None, 0.0)
                    self.in_flight.discard(tid)

    def stop(self):
        """워커 전체 종료."""
        self.on = False
        for t in self.threads:
            t.join(timeout=2)


# ═════════════════════════════════════════════════════════════════
# YoloEngine — YOLO 본 모델을 비동기 루프로 돌려 최신 결과만 보관
# ═════════════════════════════════════════════════════════════════
class YoloEngine:
    """별도 스레드에서 YOLO 추론을 돌리고 가장 최근 결과만 외부에 노출."""

    def __init__(self, model_path, gpu_id, imgsz=YOLO_IMGSZ):
        """모델 로드 후 추론 루프 시작."""
        self.model = YOLO(model_path)
        self.gpu_id = gpu_id
        self.imgsz = imgsz
        if USE_CUDA:
            try:
                self.model.to(f'cuda:{gpu_id}')
                print(f"✅ [YoloEngine] GPU {gpu_id}")
            except Exception as e:
                print(f"⚠️ YOLO GPU 할당 실패: {e}")

        self._input_lock = threading.Lock()
        self._latest_input = None
        self._output_lock = threading.Lock()
        self._latest_output = None
        self._frame_id = 0
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, frame):
        """추론할 프레임 제출 (덮어쓰기 방식 — 오래된 입력은 자동 폐기)."""
        with self._input_lock:
            self._frame_id += 1
            self._latest_input = (frame, self._frame_id)

    def get_latest(self):
        """가장 최근 결과 반환 (없으면 None)."""
        with self._output_lock:
            return self._latest_output

    def _loop(self):
        """추론 루프 — 새 입력이 들어왔을 때만 YOLO 실행, 박스 필터링 후 결과 갱신."""
        last_processed = -1
        while self.running:
            with self._input_lock:
                inp = self._latest_input
            if inp is None or inp[1] == last_processed:
                time.sleep(0.005)
                continue
            frame, fid = inp
            last_processed = fid

            try:
                kw = {"stream": True, "conf": YOLO_CONF_MIN, "imgsz": self.imgsz, "verbose": False}
                if USE_CUDA:
                    kw["half"] = True
                    kw["device"] = self.gpu_id

                yr = self.model(frame, **kw)
                raw_boxes = []
                for r in yr:
                    for box in r.boxes:
                        ci = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w_b = x2 - x1
                        h_b = y2 - y1

                        # 박스 필터 — 크기/종횡비/거리별 신뢰도 컷
                        if w_b < YOLO_MIN_SIZE or h_b < YOLO_MIN_SIZE: continue

                        ar = w_b / h_b
                        if ar < YOLO_ASPECT_MIN or ar > YOLO_ASPECT_MAX: continue
                        if max(w_b, h_b) >= FAR_BOX_THRESHOLD and conf < YOLO_CONF_NEAR: continue
                        if max(w_b, h_b) < FAR_BOX_THRESHOLD and conf < YOLO_CONF_FAR: continue

                        raw_boxes.append(((x1, y1, w_b, h_b), conf, ci))

                raw_boxes = remove_overlapping_boxes(raw_boxes)
                with self._output_lock:
                    self._latest_output = (raw_boxes, fid, time.time())
            except Exception as e:
                if DEBUG_MATCH:
                    print(f"⚠️ YoloEngine: {e}")

    def stop(self):
        """추론 루프 종료."""
        self.running = False


# ═════════════════════════════════════════════════════════════════
# FastAPI 앱 + DB 로드 + 엔진 초기화
# ═════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 시작 시 메인 처리 루프 스레드 기동."""
    threading.Thread(target=process_camera_loop, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")

_db_lock = threading.Lock()
_db_mtime_cache = {}
database = []
person_thresholds = {}


def load_face_db():
    """face_db 폴더의 인물별 JSON + legacy face_db.json을 로드해 임계값까지 자동 산출."""
    global database, person_thresholds
    new_db = []
    changed = False
    seen_names = set()

    # 새 포맷 — 인물별 JSON 파일
    if os.path.isdir(FACE_DB_DIR):
        for fn in sorted(f for f in os.listdir(FACE_DB_DIR) if f.endswith(".json")):
            path = os.path.join(FACE_DB_DIR, fn)
            try:
                mtime = os.path.getmtime(path)
                if _db_mtime_cache.get(path) != mtime:
                    changed = True
                    _db_mtime_cache[path] = mtime
                with open(path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if not isinstance(entry, dict): continue
                name = entry.get("name")
                feats = entry.get("features")

                if not name or not feats: continue

                valid = [np.asarray(ft, dtype=np.float32).ravel() for ft in feats
                         if np.asarray(ft, dtype=np.float32).ravel().shape[0] > 0]
                if not valid: continue

                valid = [v / (np.linalg.norm(v) + 1e-8) for v in valid]
                entry["features"] = valid
                if name not in seen_names:
                    new_db.append(entry)
                    seen_names.add(name)
            except Exception as e:
                print(f"⚠️ DB 로드 실패 {fn}: {e}")

    # 레거시 단일 파일 face_db.json도 함께 흡수
    if os.path.exists(DB_PATH):
        try:
            mtime = os.path.getmtime(DB_PATH)
            if _db_mtime_cache.get(DB_PATH) != mtime:
                changed = True
                _db_mtime_cache[DB_PATH] = mtime
            with open(DB_PATH, "r", encoding="utf-8") as f:
                legacy = json.load(f)
            for e in legacy:
                if e.get("name") and e.get("features") and e["name"] not in seen_names:
                    valid = [np.asarray(ft, dtype=np.float32).ravel() for ft in e["features"]
                             if np.asarray(ft, dtype=np.float32).ravel().shape[0] > 0]
                    if valid:
                        valid = [v / (np.linalg.norm(v) + 1e-8) for v in valid]
                        e["features"] = valid
                        new_db.append(e)
                        seen_names.add(e["name"])
        except Exception as e:
            print(f"⚠️ legacy DB: {e}")

    # 개인별 자동 임계값 산출
    new_thresholds = {}
    if AUTO_CALIBRATE_THRESHOLDS:
        for entry in new_db:
            name = entry["name"]
            if "self_sim_lower" in entry and entry["self_sim_lower"] is not None:
                new_thresholds[name] = clamp_thr(entry["self_sim_lower"])
                continue

            auto_thr, stats = auto_calibrate_threshold(entry["features"])
            if auto_thr is not None:
                entry["auto_thr"] = auto_thr
                new_thresholds[name] = auto_thr
                if changed:
                    print(f"  🎯 {name}: 자동 임계값 {auto_thr:.3f} "
                          f"(자기유사도 mean={stats['mean']:.3f}, p10={stats['p10']:.3f}, "
                          f"min={stats['min']:.3f}, n={stats['n_pairs']})")
            else:
                new_thresholds[name] = base_face_thr()

    with _db_lock:
        database = new_db
        person_thresholds = new_thresholds
        try:
            worker.db = new_db
        except NameError:
            pass

    if changed:
        print(f"📚 face_db: {len(new_db)}명")
    return new_db


load_face_db()
print(f"🚀 등록 인원: {len(database)}명")

# ───────── HiRes 재검증 모델 ─────────
yolo_hires = None
if USE_HIRES_VERIFY and USE_CUDA and GPU_HIRES != GPU_YOLO_MAIN:
    try:
        yolo_hires = YOLO(YOLO_MODEL)
        yolo_hires.to(f'cuda:{GPU_HIRES}')
        print(f"✅ [YOLO-HiRes] GPU {GPU_HIRES}")
        if HIRES_ASYNC:
            hires_engine = HiResVerifyEngine(yolo_hires, GPU_HIRES)
            print(f"✅ [HiRes 비동기 엔진] 시작")
    except Exception as e:
        print(f"⚠️ HiRes 초기화 실패: {e}")
        yolo_hires = None
        hires_engine = None
else:
    print(f"ℹ️ [HiRes] 비활성화 (USE_HIRES_VERIFY={USE_HIRES_VERIFY})")

worker = Worker(database, n=WORKER_COUNT)
tracker = Tracker()

pose_engine = PoseEngine(model_path=YOLO_POSE_MODEL, gpu_id=GPU_HIRES)


def _db_watcher():
    """주기적으로 face_db를 다시 로드해 추가/수정된 인물을 반영."""
    while True:
        time.sleep(10)
        try:
            load_face_db()
        except Exception as e:
            print(f"⚠️ DB watcher: {e}")


threading.Thread(target=_db_watcher, daemon=True).start()

# ───────── 송출용 최신 JPEG 공유 변수 ─────────
latest_jpeg_lock = threading.Lock()
latest_jpeg = None
latest_jpeg_cond = threading.Condition(latest_jpeg_lock)
latest_jpeg_seq = 0
fps_value = 0.0


# ═════════════════════════════════════════════════════════════════
# 메인 처리 루프 — 카메라 → YOLO → Tracker → Worker → 렌더링 → JPEG
# ═════════════════════════════════════════════════════════════════
def process_camera_loop():
    """카메라 프레임을 반복 처리 — 트래킹/식별/위반판정/렌더링/JPEG 인코딩."""
    global latest_jpeg, fps_value, latest_jpeg_seq
    grabber = None
    fc_local = 0
    heads_cache = []
    wearing_cache = set()
    zoom_active = False
    zoom_start = 0.0
    zoom_target_tid = None
    zoom_last_end = 0.0
    fpsc = deque(maxlen=30)
    server_start_time = time.time()
    last_yolo_fid = -1

    yolo_engine = YoloEngine(YOLO_MODEL, GPU_YOLO_MAIN, imgsz=YOLO_IMGSZ)

    # 카메라 연결 시도 (실패 시 5초 후 재시도)
    while grabber is None:
        try:
            grabber = CameraGrabber(0)
            time.sleep(0.3)
            server_start_time = time.time()
            print("✅ 카메라 시작")
        except Exception as e:
            print(f"❌ 카메라 실패: {e}")
            time.sleep(5.0)

    while True:
        loop_start = time.time()
        frame = grabber.read()

        # 카메라 단선 → 재연결
        if frame is None:
            if not grabber.is_alive():
                try:
                    grabber.stop()
                except:
                    pass
                grabber = None
                heads_cache = []
                wearing_cache = set()

                with latest_jpeg_cond:
                    latest_jpeg = None
                    latest_jpeg_seq += 1
                    latest_jpeg_cond.notify_all()
                tracker.tracks.clear()
                while grabber is None:
                    try:
                        grabber = CameraGrabber(0)
                        time.sleep(0.3)
                        server_start_time = time.time()
                    except:
                        time.sleep(3.0)
                continue
            time.sleep(0.01)
            continue

        now = time.time()
        if now - server_start_time < 2.0: continue

        fc_local += 1
        if fc_local % 30 == 0:
            worker.check()

        fh, fw = frame.shape[:2]

        # YOLO 비동기 — 가장 최신 결과 가져옴
        yolo_engine.submit(frame)
        yolo_result = yolo_engine.get_latest()
        new_yolo_data = (yolo_result is not None and yolo_result[1] != last_yolo_fid)

        if new_yolo_data:
            raw_boxes, fid, _ts = yolo_result
            last_yolo_fid = fid
            heads_cache = [b[0] for b in raw_boxes]
            wearing_cache = set(i for i, b in enumerate(raw_boxes) if b[2] == 1)

            matched, lost_tracks = tracker.update(heads_cache, now)

            # 사라진 트랙 중 위반 카운트 누적된 것은 마지막 증거를 저장
            if SAVE_ON_TRACK_LOST:
                for lt in lost_tracks:
                    if lt.no_helmet_count() < LOST_TRACK_MIN_VIOLATIONS or lt.best_evidence is None:
                        continue
                    ename = lt.locked_name or lt.ever_identified_as
                    elapsed_alive = lt.last_seen - lt.first_seen
                    if not ename and not lt.identification_abandoned and elapsed_alive < 3.0:
                        continue
                    try_log_violation(lt, ename or "unknown", frame, now, tracker, yolo_hires)
                    if hires_engine is not None:
                        hires_engine.clear(lt.id)

            for d_idx, t_idx in matched.items():
                if t_idx < len(tracker.tracks):
                    tracker.tracks[t_idx].update_helmet(d_idx in wearing_cache)

            # 각 트랙에 얼굴 후보 추가 + 워커 요청 (확정된 트랙은 주기적 재검증만)
            for t in tracker.tracks:
                t.prune_candidates(now)
                if t.miss > 0 or not t.is_stable(): continue

                if t.confirmed and t.locked_name is not None:
                    if (now - t.last_verify_at) < RECONFIRM_INTERVAL:
                        continue
                    t.add_candidate(frame, t.bbox, now)
                    if len(t.face_candidates) >= 2:
                        t.last_verify_at = now
                        worker.add(t.id, t.face_candidates[:3])
                    continue

                t.add_candidate(frame, t.bbox, now)

                if not t.confirmed:
                    req_interval = RE_REQUEST_INITIAL
                else:
                    req_interval = RE_REQUEST_INTERVAL if not t.is_far() else RE_REQUEST_INTERVAL*2

                if t.identification_abandoned:
                    req_interval = 1.5

                if (now-t.last_request) >= req_interval and len(t.face_candidates) > 0:
                    t.last_request = now
                    t.check_count += 1
                    worker.add(t.id, t.face_candidates)

        # ────── Worker 결과 처리 ──────
        for t in tracker.tracks:
            res = worker.pop(t.id)
            if res is None: continue

            if len(res) >= 8:
                n, c, fc_, is_confused, avg_q, status, embedding, vmargin = res
            elif len(res) >= 7:
                n, c, fc_, is_confused, avg_q, status, embedding = res
                vmargin = 0.0
            else:
                n, c, fc_, is_confused, avg_q = res
                status = ExtractStatus.OK
                embedding = None
                vmargin = 0.0

            t.response_count += 1
            if status == ExtractStatus.OK:
                t.successful_extractions += 1
                if embedding is not None:
                    t.last_good_embedding = embedding
            else:
                t.no_face_count += 1

            # ── 이미 확정된 트랙 — 재검증/swap 감지 ──
            if t.confirmed and t.locked_name is not None:
                if n == t.locked_name and fc_ >= RECONFIRM_SCORE_MIN:
                    t.disagree_streak = 0
                    t.swap_suspect_count = 0
                    t.face = max(t.face, fc_)
                    if USE_RECENT_ID_CACHE and embedding is not None and fc_ >= 0.60:
                        recent_id_cache.add(t.locked_name, embedding, now)
                    if t.anchor_embedding is None and embedding is not None:
                        t.anchor_embedding = embedding.copy()
                        t.anchor_set_at = now
                elif (n != "unknown" and n != t.locked_name
                      and not is_confused and fc_ >= RECONFIRM_OTHER_MIN):
                    # 다른 사람이 같은 이름으로 이미 확정돼 있으면 이 트랙은 swap된 것 → 즉시 kill
                    _other_has_n = any(
                        (ot.id != t.id and ot.confirmed
                         and ot.name == n and not ot.marked_for_kill
                         and ot.miss <= 5)
                        for ot in tracker.tracks
                    )
                    if _other_has_n:
                        print(f"  ❗ Track#{t.id}: '{t.locked_name}' 고정인데 '{n}'({fc_:.3f}) "
                              f"식별 — '{n}'은 다른 트랙에 이미 확정됨 → 즉시 kill (트랙 swap)",
                              flush=True)
                        t.marked_for_kill = True
                        tracker.add_kill_zone(t.bbox, now)
                        continue

                    # anchor 임베딩과 어긋남이 누적되면 swap으로 간주
                    if (t.anchor_embedding is not None and embedding is not None):
                        _anchor_sim = cosine_sim(embedding, t.anchor_embedding)
                        if _anchor_sim < ANCHOR_EMB_SWAP_THRESHOLD:
                            if (now - getattr(t, 'last_swap_suspect_at', 0)) > ANCHOR_EMB_SWAP_WINDOW:
                                t.swap_suspect_count = 0
                            t.swap_suspect_count += 1
                            t.last_swap_suspect_at = now
                            print(f"  🔄 Track#{t.id}: anchor 임베딩 어긋남 "
                                  f"(sim={_anchor_sim:.3f} < {ANCHOR_EMB_SWAP_THRESHOLD}) "
                                  f"[{t.swap_suspect_count}/{ANCHOR_EMB_SWAP_FAST_KILL}]",
                                  flush=True)
                            if t.swap_suspect_count >= ANCHOR_EMB_SWAP_FAST_KILL:
                                t.marked_for_kill = True
                                tracker.add_kill_zone(t.bbox, now)
                                continue
                        else:
                            if t.swap_suspect_count > 0:
                                t.swap_suspect_count = max(0, t.swap_suspect_count - 1)

                    t.disagree_streak += 1
                    print(f"  ⚠️ Track#{t.id}: '{t.locked_name}' 고정인데 '{n}'({fc_:.3f}) "
                          f"[{t.disagree_streak}/{RECONFIRM_DISAGREE_MAX}]", flush=True)
                    if t.disagree_streak >= RECONFIRM_DISAGREE_MAX:
                        t.marked_for_kill = True
                        tracker.add_kill_zone(t.bbox, now)
                continue

            t.score = c
            t.face = fc_

            # ── 재진입 캐시 매칭 — 같은 사람이 다시 등장한 경우 즉시 확정 ──
            if (USE_RECENT_ID_CACHE
                    and not t.cache_lookup_done
                    and embedding is not None
                    and status == ExtractStatus.OK):
                t.cache_lookup_done = True
                cached_name, cached_sim = recent_id_cache.lookup(embedding, now)
                if cached_name is not None:
                    if n == cached_name or n == "unknown":
                        threshold = max(0.60, person_thresholds.get(cached_name, 0.55))
                        if cached_sim >= threshold:
                            print(f"  ⚡ Track#{t.id}: 재진입 캐시 매치 → "
                                  f"{cached_name} (sim={cached_sim:.3f})", flush=True)
                            t.confirmed = True
                            t.name = cached_name
                            t.locked_name = cached_name
                            t.ever_identified_as = cached_name
                            t.identified_at = now
                            t.face = cached_sim
                            t.confirm_score = cached_sim
                            t.last_verify_at = now
                            t.disagree_streak = 0
                            t.anchor_embedding = embedding.copy()
                            t.anchor_set_at = now
                            t.swap_suspect_count = 0
                            tracker.mark_seen(cached_name, now)
                            recent_id_cache.add(cached_name, embedding, now)
                            _void_unknown_for_track(t, replaced_by=cached_name)
                            if t.is_violation():
                                try_log_violation(t, cached_name, frame, now, tracker, yolo_hires)
                            continue

            # ── 미식별자 재진입 캐시 — 이미 본 미식별자면 즉시 포기 ──
            if (USE_RECENT_UNKNOWN_CACHE
                    and not t.unknown_cache_checked
                    and embedding is not None
                    and status == ExtractStatus.OK
                    and n == "unknown"
                    and not is_confused):
                t.unknown_cache_checked = True
                hit, sim = recent_unknown_cache.hit(embedding, now)
                if hit:
                    t.identification_abandoned = True
                    print(f"  ♻️ Track#{t.id}: 이전 미식별자와 유사 (sim={sim:.3f}) "
                          f"→ 즉시 확정", flush=True)
                    if t.is_violation():
                        try_log_violation(t, "unknown", frame, now, tracker, yolo_hires)
                    continue

            if status == ExtractStatus.OK and embedding is not None:
                if n == "unknown" and not is_confused:
                    t.consecutive_unknown_with_face += 1
                else:
                    t.consecutive_unknown_with_face = 0

            # ── 조기 미식별자 확정 — unknown이 연속으로 나오면 빠르게 포기 ──
            _track_elapsed = now - t.first_seen
            if (not t.identification_abandoned
                    and t.consecutive_unknown_with_face >= EARLY_ABANDON_UNKNOWN_WITH_FACE
                    and _track_elapsed >= EARLY_ABANDON_MIN_ELAPSED
                    and t.response_count >= EARLY_ABANDON_MIN_RESPONSES):
                t.identification_abandoned = True
                print(f"  ⚡ Track#{t.id}: 조기 미식별자 확정 "
                      f"(unknown {t.consecutive_unknown_with_face}회, "
                      f"elapsed {_track_elapsed:.1f}s, 응답 {t.response_count}회)",
                      flush=True)
                if t.is_violation():
                    try_log_violation(t, "unknown", frame, now, tracker, yolo_hires)
                continue

            t.add_vote(n, fc_, avg_q, vmargin, now)

            # ── 다수결로 신원 확정 시도 ──
            if n != "unknown" and not is_confused:
                majority_name, majority_score, majority_ratio = t.get_majority_vote(person_thresholds)
                if majority_name is not None:
                    # 같은 트랙이 다른 이름으로 바뀌려 하면 거부
                    if t.ever_identified_as and t.ever_identified_as != majority_name:
                        print(f"  🛡️ Track#{t.id}: '{t.ever_identified_as}' → "
                              f"'{majority_name}' 거부", flush=True)
                        t.vote_history.clear()
                        t.marked_for_kill = True
                        continue

                    t.confirmed = True
                    t.name = majority_name
                    t.locked_name = majority_name
                    t.ever_identified_as = majority_name
                    t.identified_at = now
                    t.face = majority_score
                    t.confirm_score = majority_score
                    t.last_verify_at = now
                    t.disagree_streak = 0
                    if t.last_good_embedding is not None:
                        t.anchor_embedding = t.last_good_embedding.copy()
                        t.anchor_set_at = now
                    t.swap_suspect_count = 0
                    tracker.mark_seen(majority_name, now)
                    person_thr = person_thresholds.get(majority_name, FACE_THR_INSIGHT)
                    print(f"  🔒 Track#{t.id}: 확정 → {majority_name} "
                          f"(투표 {len(t.vote_history)}개, ratio={majority_ratio:.0%}, "
                          f"avg={majority_score:.3f}, thr={person_thr:.2f})", flush=True)
                    if USE_RECENT_ID_CACHE and embedding is not None:
                        recent_id_cache.add(majority_name, embedding, now)
                    _void_unknown_for_track(t, replaced_by=majority_name)
                    if t.is_violation():
                        try_log_violation(t, majority_name, frame, now, tracker, yolo_hires)

        # ── 동일 이름 중복 트랙 정리 ──
        kill_ids = enforce_name_uniqueness(tracker.tracks)
        for t in tracker.tracks:
            if t.marked_for_kill:
                kill_ids.add(t.id)
        if hires_engine is not None:
            for kid in kill_ids:
                hires_engine.clear(kid)
        tracker.remove_ids(kill_ids)

        # 살아있는 미식별자 ID는 last_seen 갱신
        for t in tracker.tracks:
            if t.miss == 0 and t.unknown_person_id is not None:
                unknown_registry.touch(t.unknown_person_id, now)

        # ── 디지털 줌 대상 선정 (의심 트랙 우선) ──
        if not zoom_active and (now-zoom_last_end) > ZOOM_COOLDOWN:
            best_pri = 0
            zoom_cand = None
            for t in tracker.tracks:
                if t.miss > 0 or t.box_size() >= ZOOM_MIN_HEAD: continue
                if (now-t.last_zoomed_at) < ZOOM_COOLDOWN*2: continue
                if t.identification_abandoned and t.unknown_logged:
                    continue
                if t.confirmed and t.name != "unknown":
                    if violation_guard.is_in_cooldown(t.name, None, None, now):
                        continue
                pri = (3 if (t.is_violation() and not t.confirmed) else
                       (2 if t.is_fast_suspect() else
                        (1 if t.suspect_no_helmet_count >= ZOOM_SUSPECT_FRAMES else 0)))
                if pri > best_pri:
                    best_pri = pri
                    zoom_cand = t
            if zoom_cand:
                zoom_active = True
                zoom_start = now
                zoom_target_tid = zoom_cand.id
                zoom_cand.last_zoomed_at = now

        if zoom_active and (now-zoom_start > ZOOM_DURATION):
            zoom_active = False
            zoom_last_end = now
            zoom_target_tid = None

        # ────── 렌더링 ──────
        display = frame.copy()

        # 안전모 착용자(녹색 얇은 박스) 표시
        for i, hb in enumerate(heads_cache):
            if i in wearing_cache:
                cv2.rectangle(display, (hb[0],hb[1]), (hb[0]+hb[2],hb[1]+hb[3]), (0,200,0), 1)

        for t in tracker.tracks:
            if t.miss > 5:
                continue
            # 잠시 놓친 트랙은 흐릿하게 그리기
            if t.miss > 0:
                if t.is_stable():
                    hx, hy, hw, hh = t.bbox
                    if t.confirmed and t.name != "unknown":
                        col = (120, 160, 120)
                        lb = f"{t.name}"
                    else:
                        col = (140, 140, 140)
                        lb = ""
                    cv2.rectangle(display, (hx, hy), (hx+hw, hy+hh), col, 1)
                    if lb:
                        put_text_kr(display, lb, (hx+2, max(20, hy-3)),
                                    font_size=16, color=(220, 220, 220))
                continue
            hx,hy,hw,hh = t.bbox
            if not t.is_stable(): continue

            # ── 자세 분석 요청 ──
            if now - t.last_pose_check > POSE_CHECK_INTERVAL:
                px1 = max(0, hx - int(hw * 2.5))
                py1 = max(0, hy - int(hh * 0.6))
                px2 = min(fw, hx + int(hw * 3.5))
                py2 = min(fh, hy + int(hh * 9.0))
                if px2 - px1 > 80 and py2 - py1 > 120:
                    crop = frame[py1:py2, px1:px2].copy()
                    pose_engine.submit(t.id, crop, px1, py1)
                    t.last_pose_check = now

            pose_res = pose_engine.consume(t.id)
            confirmed_event = t.process_pose_result(pose_res, fh, now)

            if confirmed_event is not None and (now - t.pose_logged_at) > POSE_LOG_COOLDOWN:
                known_name = t.locked_name or t.ever_identified_as or "unknown"
                t.force_update_evidence(frame)
                logged = try_log_violation(t, known_name, frame, now, tracker, yolo_hires,
                                            reason=confirmed_event)
                if logged:
                    t.pose_logged_at = now

            # 비정상 자세 표시 (확정 후 일정 시간 동안)
            if t.pose_state != "정상" and (now - t.pose_detected_at) < POSE_DISPLAY_DURATION:
                col = (0, 0, 255)
                known_name = t.locked_name or t.ever_identified_as or "미식별자"
                lb = f"🚨 {t.pose_state} ({known_name})"

                cv2.rectangle(display, (hx,hy), (hx+hw,hy+hh), col, 4)
                tw_,th_ = get_text_size_kr(lb, font_size=22)
                cv2.rectangle(display, (hx,hy-th_-6), (hx+tw_+4,hy), col, -1)
                put_text_kr(display, lb, (hx+2,hy-3), font_size=22, color=(255,255,255))

                if t.tilt_deg is not None:
                    tilt_str = f"tilt={t.tilt_deg:.0f}°"
                    put_text_kr(display, tilt_str, (hx, hy+hh+20),
                                font_size=14, color=(0,255,255))
                continue

            violating = t.is_violation()
            fast_susp = t.is_fast_suspect()

            # 정상 상태 — 확정자는 이름, 아니면 녹색 얇은 박스
            if not violating and not fast_susp:
                if t.confirmed and t.name != "unknown":
                    col = (0,180,0)
                    lb = f"{t.name} 🔒"
                    cv2.rectangle(display, (hx,hy), (hx+hw,hy+hh), col, 2)
                    tw_,th_ = get_text_size_kr(lb, font_size=18)
                    cv2.rectangle(display, (hx,hy-th_-6), (hx+tw_+4,hy), col, -1)
                    put_text_kr(display, lb, (hx+2,hy-3), font_size=18, color=(255,255,255))
                else:
                    cv2.rectangle(display, (hx,hy), (hx+hw,hy+hh), (0,180,0), 1)
                continue

            t.maybe_update_evidence(frame, len(t.face_candidates)>0)

            # 의심 단계 (확정 전)
            if fast_susp and not violating:
                cv2.rectangle(display, (hx,hy), (hx+hw,hy+hh), (0,140,255), 2)
                cv2.putText(display, f"NO HELMET? ({t.no_helmet_count()})",
                            (hx,hy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,140,255), 2)
                continue

            elapsed = now - t.first_seen

            # ── 위반 확정 상태 — 라벨/색상 결정 + 로그 시도 ──
            if t.confirmed and t.name != "unknown":
                if violation_guard.is_in_cooldown(t.name, None, None, now):
                    remaining = max(0, PERSON_LOG_COOLDOWN - violation_guard.time_since_log(t.name))
                    col = (128,128,128)
                    lb = f"{t.name} (IGNORED {int(remaining)}s)"
                else:
                    col = (0,0,255)
                    lb = f"{t.name} ({t.score:.2f})"
                    if not t.person_logged:
                        try_log_violation(t, t.name, frame, now, tracker, yolo_hires)
            else:
                worker_pending = (t.check_count > 0 and (now-t.last_request) < RE_REQUEST_INITIAL*4)
                known_name = t.locked_name or t.ever_identified_as

                ready_to_abandon = (
                    not t.identification_abandoned and (
                        (elapsed >= MAX_IDENTIFICATION_TIME and
                         t.successful_extractions >= MIN_SUCCESSFUL_RESPONSES) or
                        t.response_count >= MAX_WORKER_RESPONSES
                    )
                )
                if ready_to_abandon:
                    t.identification_abandoned = True
                    print(f"  ⏰ Track#{t.id}: 식별 포기 ({elapsed:.1f}s, "
                          f"응답 {t.response_count}, 성공 {t.successful_extractions})", flush=True)

                is_still = (not t.identification_abandoned and
                            (elapsed < IDENTIFICATION_GRACE or
                             t.successful_extractions < MIN_SUCCESSFUL_RESPONSES or
                             worker_pending))

                if is_still:
                    col = (0,200,255)
                    lb = f"IDENTIFYING ({elapsed:.1f}s, {t.successful_extractions}/{MIN_SUCCESSFUL_RESPONSES})"
                elif known_name:
                    if violation_guard.is_in_cooldown(known_name, None, None, now):
                        remaining = max(0, PERSON_LOG_COOLDOWN - violation_guard.time_since_log(known_name))
                        col = (128,128,128)
                        lb = f"{known_name} (IGNORED {int(remaining)}s)"
                    else:
                        col = (0,0,255)
                        lb = f"{known_name} ({t.face:.2f})"
                        try_log_violation(t, known_name, frame, now, tracker, yolo_hires)
                else:
                    if t.identification_abandoned:
                        unk_label = t.unknown_person_id or "미식별자"

                        if t.unknown_logged:
                            col = (128,128,128)
                            lb = f"{unk_label} (LOGGED)"
                        elif violation_guard.is_in_cooldown(None, t.last_good_embedding, t.bbox, now):
                            col = (128,128,128)
                            lb = f"{unk_label} (IGNORED)"
                        else:
                            col = (0,100,255)
                            lb = f"{unk_label} - NO HELMET"
                            if t.best_evidence is None:
                                t.force_update_evidence(frame)
                            try_log_violation(t, "unknown", frame, now, tracker, yolo_hires)
                    else:
                        col = (0,200,255)
                        lb = f"IDENTIFYING ({elapsed:.1f}s)"

            cv2.rectangle(display, (hx,hy), (hx+hw,hy+hh), col, 3)
            tw_,th_ = get_text_size_kr(lb, font_size=20)
            cv2.rectangle(display, (hx,hy-th_-6), (hx+tw_+4,hy), col, -1)
            put_text_kr(display, lb, (hx+2,hy-3), font_size=20, color=(255,255,255))

        # ── 줌 오버레이 ──
        if zoom_active and zoom_target_tid is not None:
            target = next((t for t in tracker.tracks
                           if t.id==zoom_target_tid and t.miss==0), None)
            if target:
                zoomed, nbbox = digital_zoom_crop(frame, target.bbox, ZOOM_SCALE)
                if zoomed is not None:
                    nhx,nhy,nhw,nhh = nbbox
                    col = (0,0,255) if target.is_violation() else (0,140,255)
                    cv2.rectangle(zoomed, (nhx,nhy), (nhx+nhw,nhy+nhh), col, 3)
                    cv2.putText(zoomed, f"ZOOM x{ZOOM_SCALE}",
                                (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                    display = zoomed
            else:
                zoom_active = False
                zoom_last_end = now
                zoom_target_tid = None

        # ── FPS 측정 ──
        fpsc.append(time.time()-loop_start)
        if fpsc:
            avg = sum(fpsc)/len(fpsc)
            fps_value = 1.0/avg if avg > 0 else 0.0

        # ── 송출용 다운스케일 + 빠른 JPEG 인코딩 ──
        try:
            send_img = cv2.resize(display, (STREAM_WIDTH, STREAM_HEIGHT),
                                  interpolation=cv2.INTER_LINEAR)
            ret_enc, buf = cv2.imencode('.jpg', send_img,
                                         [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_Q,
                                          cv2.IMWRITE_JPEG_OPTIMIZE, 0,
                                          cv2.IMWRITE_JPEG_PROGRESSIVE, 0])
            if ret_enc:
                with latest_jpeg_cond:
                    latest_jpeg = buf.tobytes()
                    latest_jpeg_seq += 1
                    latest_jpeg_cond.notify_all()
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════
# MJPEG 송출
# ═════════════════════════════════════════════════════════════════
def generate_frames():
    """MJPEG generator — 처리 FPS와 송출 FPS를 분리해 TCP 버퍼 누적 방지."""
    last_seq = -1
    last_sent_time = 0.0
    min_interval = 1.0 / STREAM_SEND_FPS
    boundary_prefix = b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
    try:
        while True:
            cur = None
            with latest_jpeg_cond:
                # 새 프레임이 없으면 push될 때까지 대기 (최대 1초) — busy loop 방지
                if latest_jpeg_seq == last_seq:
                    latest_jpeg_cond.wait(timeout=1.0)
                if latest_jpeg is None or latest_jpeg_seq == last_seq:
                    continue
                now_t = time.time()
                # 송출 페이스 조절: min_interval 미달이면 이 프레임은 스킵
                if (now_t - last_sent_time) < min_interval:
                    last_seq = latest_jpeg_seq
                    sleep_time = min_interval - (now_t - last_sent_time)
                    pass_through = True
                else:
                    pass_through = False
                    cur = latest_jpeg
                    last_seq = latest_jpeg_seq
                    last_sent_time = now_t

            if pass_through:
                # Condition을 풀고 sleep해야 다른 스레드의 notify를 막지 않음
                time.sleep(max(0.005, min(sleep_time, min_interval)))
                continue

            yield (boundary_prefix
                   + str(len(cur)).encode()
                   + b'\r\n\r\n'
                   + cur
                   + b'\r\n')
    except (GeneratorExit, Exception):
        # 클라이언트 연결이 끊겨도 메인 루프는 계속 돌아야 함
        pass


# ═════════════════════════════════════════════════════════════════
# HTTP 엔드포인트
# ═════════════════════════════════════════════════════════════════
@app.get("/face_db_status")
def face_db_status():
    """등록된 얼굴 DB 스냅샷과 개인별 임계값 반환."""
    with _db_lock:
        db_snap = list(database)
        thrs = dict(person_thresholds)
    entries = []
    for e in db_snap:
        feats = e.get("features",[])
        entries.append({
            "name": e.get("name","?"),
            "features_count": len(feats),
            "auto_thr": e.get("auto_thr"),
            "applied_thr": thrs.get(e.get("name","?")),
        })
    return JSONResponse({"loaded_count":len(db_snap), "entries":entries,
                         "thresholds": thrs})


@app.get("/cooldown_status")
def cooldown_status():
    """현재 쿨다운 상태 + 캐시/레지스트리 스냅샷 (디버그용)."""
    now = time.time()
    with violation_guard._lock:
        names = {n: round(PERSON_LOG_COOLDOWN - (now - ts), 1)
                 for n, ts in violation_guard._by_name.items()
                 if (now - ts) < PERSON_LOG_COOLDOWN}
        unk = len([r for r in violation_guard._unknown_records
                   if (now - r[1]) < UNKNOWN_LOG_COOLDOWN])
    cache = recent_id_cache.status(now)
    unk_cache_size = recent_unknown_cache.size(now)
    unk_registry = unknown_registry.status(now)
    return JSONResponse({"by_name": names, "unknowns_active": unk,
                         "unknown_cooldown": UNKNOWN_LOG_COOLDOWN,
                         "recent_id_cache": cache,
                         "recent_unknown_cache_size": unk_cache_size,
                         "unknown_registry": unk_registry})


@app.get("/violations")
def get_violations():
    """최근 위반 목록."""
    with violations_lock:
        return JSONResponse(list(recent_violations))


@app.get("/stats")
def get_stats():
    """현재 FPS와 위반 누적 수."""
    with violations_lock:
        return {"fps": round(fps_value,1), "total": len(recent_violations)}


# /video_feed 헤더 — 캐시 완전 차단 + nginx 같은 리버스프록시 버퍼링 OFF
@app.get("/video_feed")
def video_feed():
    """MJPEG 스트림 응답 (multipart/x-mixed-replace)."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
def realtime():
    """루트 — HTML 페이지 서빙."""
    if os.path.exists(HTML_FILE):
        return FileResponse(HTML_FILE)
    return JSONResponse({"error": f"{HTML_FILE} not found"}, status_code=500)


# ═════════════════════════════════════════════════════════════════
# uvicorn 실행
#   TCP_NODELAY 패치: MJPEG는 작은 boundary chunk가 끊임없이 나가는데
#   Nagle이 켜져있으면 그걸 모아 보내려고 기다려서 송출 지연이 누적됨.
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import socket as _socket
    _orig_socket_init = _socket.socket.__init__

    def _patched_socket_init(self, *args, **kwargs):
        """모든 새 TCP 소켓에 TCP_NODELAY 적용 (Nagle 끄기)."""
        _orig_socket_init(self, *args, **kwargs)
        try:
            # TCP 스트림 소켓에만 적용 — UDP/UNIX 소켓은 무시
            if self.type == _socket.SOCK_STREAM:
                self.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
        except Exception:
            pass

    _socket.socket.__init__ = _patched_socket_init

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        http="h11",
        timeout_keep_alive=5,
        log_level="warning",
        access_log=False,
    )