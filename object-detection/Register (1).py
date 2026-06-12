import os, sys, json, cv2, numpy as np

IMAGE_FOLDER = "known_faces"
FACE_DB_DIR  = "face_db"    
VIDEO_EXTS   = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
IMG_EXTS     = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_FACES    = 60

DUP_THR_INS  = 0.985   # InsightFace
DUP_THR_DF   = 0.95    # DeepFace/Facenet512

# ── 얼굴 엔진 ──────────────────────────────────────────────
USE_INSIGHTFACE = True
try:
    from insightface.app import FaceAnalysis
    _face_app = FaceAnalysis(name='buffalo_l',
                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    _face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("[얼굴엔진] InsightFace (buffalo_l) ✓")
    DUP_THR = DUP_THR_INS
except Exception as e:
    # InsightFace 로드 실패 시 DeepFace로 폴백
    print(f"[얼굴엔진] InsightFace 실패 → DeepFace 폴백: {e}")
    USE_INSIGHTFACE = False
    from deepface import DeepFace
    DUP_THR = DUP_THR_DF

# ── ReID ──────────────────────────────────────────────────
USE_REID = True
try:
    import torch, torchreid
    _reid = torchreid.models.build_model('osnet_x0_25', num_classes=1000, pretrained=True)
    _reid.eval()
    _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    _reid = _reid.to(_dev)
    print(f"[ReID] OSNet ({_dev}) ✓")
except Exception as e:
    # ReID 못 띄우면 HSV 컬러 히스토그램만으로 사람 매칭
    print(f"[ReID] 로드 실패 → HSV만 사용: {e}")
    USE_REID = False

# ── 유틸 ──────────────────────────────────────────────────
def resize(img, mx=640):
    h, w = img.shape[:2]
    if max(h, w) > mx:
        s = mx / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)))
    return img

def enhance(img):
    """CLAHE로 L 채널 대비 보정 — 어두운 영상 얼굴 검출률 올리는 용도."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def cosine_sim(v1, v2):
    a = np.asarray(v1, dtype=np.float32).ravel()
    b = np.asarray(v2, dtype=np.float32).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 1e-8 else 0.0

def extract_face(img):
    """얼굴 임베딩 반환. 여러 명이 잡히면 bbox 가장 큰 것 하나만."""
    if USE_INSIGHTFACE:
        try:
            faces = _face_app.get(img)
            if not faces: return None
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            return np.asarray(f.normed_embedding, dtype=np.float32).tolist()
        except:
            return None
    else:
        # DeepFace는 detector 백엔드별로 잘 잡는 경우가 달라서 순차 시도
        for det in ["retinaface", "mtcnn", "ssd"]:
            try:
                r = DeepFace.represent(img_path=img, model_name="Facenet512",
                                       enforce_detection=True, detector_backend=det)
                if r: return r[0]["embedding"]
            except: continue
        return None

# ── ReID ──────────────────────────────────────────────────
def get_reid_embedding(body):
    if not USE_REID or body is None or body.size == 0: return None
    try:
        # 입력 규격: 128x256 RGB + ImageNet 정규화
        img = cv2.cvtColor(body, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 256))
        t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = (t - mean) / std
        t = t.to(_dev)
        with torch.no_grad():
            feat = _reid(t)
        return feat.cpu().numpy().flatten().astype(np.float32).tolist()
    except Exception as e:
        print(f"  [ReID오류] {e}")
        return None

# ── 옷 HSV ────────────────────────────────────────────────
def get_color_hist(img):
    small = cv2.resize(img, (150, 150)) if max(img.shape[:2]) > 150 else img.copy()
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # 채도/명도 낮은 픽셀(배경, 그림자) 제외하고 H-S 2D 히스토그램
    mask = cv2.inRange(hsv, (0, 25, 25), (180, 255, 255))
    if cv2.countNonZero(mask) < 30: return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().tolist()

def extract_outfit_from_dir(d, skip=None):
    """폴더 내 이미지에서 상의 HSV + 전신 ReID 임베딩 평균 반환."""
    if skip is None: skip = set()
    upper_hists = []
    reid_embs   = []
    proc = []
    for fn in sorted(os.listdir(d)):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in IMG_EXTS or fn in skip: continue
        img = cv2.imread(os.path.join(d, fn))
        if img is None: continue
        img = resize(img, 640); h, w = img.shape[:2]; asp = h / w

        # 세로로 긴 사진(asp>=1.3)은 전신으로 보고 상의 영역만 잘라냄
        if asp >= 1.3:
            upper = img[int(h*0.15):int(h*0.45), :]
            body  = img[int(h*0.10):int(h*0.85), :]
            print(f"    전신: {fn}")
        else:
            upper = img
            body  = img
            print(f"    상의: {fn}")

        hi = get_color_hist(upper)
        if hi: upper_hists.append(hi)
        rid = get_reid_embedding(body)
        if rid: reid_embs.append(rid)
        proc.append(fn)

    # 여러 장의 평균을 대표값으로 저장 (단일 사진 노이즈 줄이려고)
    hist_avg = np.mean(upper_hists, axis=0).tolist() if upper_hists else None
    reid_avg = np.mean(reid_embs, axis=0).tolist() if reid_embs else None
    outfit = {"color_hist": hist_avg} if hist_avg else None
    return outfit, reid_avg, proc

# ── 얼굴 영상 추출 ─────────────────────────────────────────
def extract_faces_from_video(path):
    """영상에서 얼굴 임베딩 최대 MAX_FACES개 추출. 정/반전 둘 다 시도해 측면 보강."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    영상: {total}프레임 {fps:.0f}fps")
    embs = []; fc = 0; ok = 0; orig_c = 0; flip_c = 0
    interval = 2   # 2프레임마다 1장만 처리 (연속 프레임은 거의 같아서 낭비)
    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        if fc % interval != 0: continue
        frame = resize(frame); frame = enhance(frame)
        # 같은 프레임을 좌우반전해서 한 번 더 → 반대쪽 측면 얼굴 임베딩 확보
        for variant, img in [("orig", frame), ("flip", cv2.flip(frame, 1))]:
            if ok >= MAX_FACES: break
            emb = extract_face(img)
            if emb is None: continue
            # DUP_THR 이상으로 비슷한 임베딩은 중복으로 보고 스킵
            if any(cosine_sim(emb, e) > DUP_THR for e in embs): continue
            embs.append(emb); ok += 1
            if variant == "orig": orig_c += 1
            else: flip_c += 1
            print(f"    [{ok}/{MAX_FACES}] ({variant})")
        if ok >= MAX_FACES: break
    cap.release()
    print(f"    → {ok}개 추출 (정방향 {orig_c}, 반전 {flip_c})")
    return embs

def compute_self_sim(embs):
    """
    수집한 임베딩들끼리 유사도를 봐서 인식 임계값을 사람마다 자동 결정.
    동일 인물 내부 유사도의 하위 10퍼센타일에서 약간 더 내린 값을 임계값으로 씀.
    → 너무 빡빡하면 본인도 못 알아보고, 너무 느슨하면 남을 본인으로 착각함.
    """
    if len(embs) < 2:
        return 0.45 if USE_INSIGHTFACE else 0.72
    sims = []
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            sims.append(cosine_sim(embs[i], embs[j]))
    sims.sort()
    idx = max(0, int(len(sims) * 0.10))
    p10 = float(sims[idx])
    adjusted = p10 - 0.06
    # 엔진별로 유사도 분포가 달라서 clamp 범위도 다르게
    if USE_INSIGHTFACE:
        result = max(0.35, min(adjusted, 0.48))
    else:
        result = max(0.65, min(adjusted, 0.75))
    print(f"    자기유사도: 10th={p10:.3f} → thr={result:.3f}")
    return result

# ── DB ────────────────────────────────────────────────────
def load_db(folder=None):
    """face_db/ 폴더의 {name}.json 들을 읽어 db, processed_files 두 딕셔너리로 반환."""
    if folder is None:
        folder = FACE_DB_DIR
    db = {}; proc = {}
    if not os.path.isdir(folder):
        return db, proc
    for fn in sorted(f for f in os.listdir(folder) if f.endswith(".json")):
        path = os.path.join(folder, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                continue
            db[name] = entry
            proc[name] = set(entry.get("processed_files", []))
        except Exception as e:
            print(f"[load_db] {fn} 읽기 실패: {e}")
    return db, proc


def _save_entry(entry: dict):
    """
    한 사람 entry를 face_db/{name}.json 으로 저장.
    임시 파일에 먼저 쓰고 os.replace 로 원자적 교체 → 쓰는 도중 끊겨도 파일 안 깨짐.
    """
    os.makedirs(FACE_DB_DIR, exist_ok=True)
    safe_name = entry["name"].replace(" ", "_")
    file_path = os.path.join(FACE_DB_DIR, f"{safe_name}.json")
    import tempfile as _tmpmod
    fd, tmp = _tmpmod.mkstemp(suffix=".json", dir=FACE_DB_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
        os.replace(tmp, file_path)
    except Exception:
        # 실패 시 임시 파일 청소
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass
        raise


# ══════════════════════════════════════════════════════════
#  배치 실행 (CLI 전용)
# ══════════════════════════════════════════════════════════
def run_batch():
    # CLI 인자 파싱 (import 시점엔 호출 안 되므로 sys.argv 건드려도 안전)
    reset_mode = "--reset" in sys.argv
    reset_person = None
    for i, a in enumerate(sys.argv):
        if a == "--reset-person" and i + 1 < len(sys.argv):
            reset_person = sys.argv[i + 1]

    print("=" * 50)
    print("  얼굴 + 옷 + ReID 등록 v2")
    print("=" * 50)
    if reset_person: print(f"  ★ {reset_person} 재등록")

    # --reset 이면 기존 DB 무시하고 처음부터 / 아니면 기존 거 이어서
    if reset_mode:
        edb = {}; eproc = {}
    else:
        edb, eproc = load_db(FACE_DB_DIR)

    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print("폴더 생성됨. 예: known_faces/홍길동/video.mp4 + front.jpg")
        return

    database = []
    for pn in sorted(os.listdir(IMAGE_FOLDER)):
        pd = os.path.join(IMAGE_FOLDER, pn)
        if not os.path.isdir(pd): continue

        # 이미 처리한 파일은 건너뛰고 새로 추가된 것만 처리 (증분 등록)
        ap = eproc.get(pn, set())
        af = set(os.listdir(pd))
        nf = af - ap
        nv = [f for f in nf if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
        ni = [f for f in nf if os.path.splitext(f)[1].lower() in IMG_EXTS]

        force_reset = (reset_person == pn)
        # 새 파일 없고 강제 재등록도 아니면 스킵
        if pn in edb and not nv and not ni and not force_reset:
            print(f"[{pn}] 스킵")
            database.append(edb[pn]); continue

        print(f"\n[{pn}]")
        # 기존 entry에 이어붙일지, 새로 만들지
        if pn in edb and not reset_mode and not force_reset:
            entry = edb[pn].copy()
            ef = list(entry.get("features", []))
            ep = set(entry.get("processed_files", []))
        else:
            entry = {"name": pn, "features": [], "outfit_upper": None,
                     "reid_embedding": None, "processed_files": [],
                     "self_sim_lower": 0.0,
                     "engine": "insightface" if USE_INSIGHTFACE else "deepface"}
            ef = []; ep = set()

        # 옷 + ReID (이미지에서)
        outfit, reid_emb, op = extract_outfit_from_dir(pd, skip=ep)
        if outfit: entry["outfit_upper"] = outfit
        if reid_emb: entry["reid_embedding"] = reid_emb
        print(f"    옷={'✓' if outfit else '✗'}  ReID={'✓' if reid_emb else '✗'}")

        # 얼굴 (영상에서)
        ne = []
        for fn in nv:
            print(f"  영상: {fn}")
            try:
                ne.extend(extract_faces_from_video(os.path.join(pd, fn)))
            except Exception as e:
                print(f"    ⚠ 오류: {e}")

        all_f = ef + ne
        entry["features"] = all_f
        entry["engine"] = "insightface" if USE_INSIGHTFACE else "deepface"
        # 임베딩 2개 이상이면 자동 임계값, 부족하면 보수적 기본값
        entry["self_sim_lower"] = compute_self_sim(all_f) if len(all_f) >= 2 else (0.45 if USE_INSIGHTFACE else 0.72)
        entry["processed_files"] = sorted((ep | set(op) | set(nv)) & af)
        database.append(entry)

    for p in database:
        _save_entry(p)

    print("\n" + "=" * 40)
    for p in database:
        ho = "✓" if p.get("outfit_upper") else "✗"
        rd = "✓" if p.get("reid_embedding") else "✗"
        print(f"  {p['name']}: 얼굴{len(p.get('features', []))} thr={p.get('self_sim_lower',0):.3f} 옷={ho} ReID={rd}")
    print("완료!")


if __name__ == "__main__":
    run_batch()