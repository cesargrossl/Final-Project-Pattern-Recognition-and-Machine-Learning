# test01.py — Detector de falhas por textura (Gabor + LBP + FFT) com OC-SVM
# Versão com salvamento robusto em Windows (imwrite_unicode)

import warnings
warnings.filterwarnings("ignore", message="Applying `local_binary_pattern`")

from pathlib import Path
import numpy as np
import cv2

from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage import img_as_ubyte, exposure
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ======================== Paths & Params ========================
ROOT        = Path(__file__).resolve().parent
DIR_PADRAO  = ROOT / "dataset" / "padrao"      # treino preferido (normais)
DIR_FALHAS  = ROOT / "dataset" / "falhas"      # fallback (falhas)
DIR_ANALIS  = ROOT / "dataset" / "analisar"    # imagens grandes para analisar
OUT_DIR     = ROOT / "outputs"

PATCH_SIZE  = 128
STRIDE      = 64
NU          = 0.10
GAMMA       = "scale"

GABOR_THETAS = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
GABOR_FREQS  = [0.05, 0.10, 0.20, 0.30]

THR_PERC = 85
MIN_BOX  = 0.05
NMS_IOU  = 0.25
VERBOSE  = True

# ======================== I/O Helpers ===========================
def imread_unicode(p: Path):
    # leitura robusta (Windows com acentos)
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size == 0: return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(p), cv2.IMREAD_COLOR)

def imwrite_unicode(path: Path, image) -> bool:
    """
    Salva imagem em caminhos com acento usando imencode + tofile (Windows-safe).
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        ext = ".png"
        path = path.with_suffix(ext)
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        buf.tofile(str(path))  # suporta unicode
        return True
    except Exception:
        return cv2.imwrite(str(path), image)

def list_images(d: Path):
    return [*d.glob("*.jpg"), *d.glob("*.jpeg"), *d.glob("*.png"), *d.glob("*.bmp"),
            *d.glob("*.JPG"), *d.glob("*.JPEG"), *d.glob("*.PNG"), *d.glob("*.BMP")]

# ======================== Features ==============================
def radial_power_spectrum(img_gray, nbins=16):
    h, w = img_gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fftshift(np.fft.fft2(img_gray * win))
    psd2D = np.abs(f) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - cx, y - cy)
    r = r / r.max() * np.pi
    hist, _ = np.histogram(r, bins=nbins, range=(0, np.pi), weights=psd2D)
    return (hist / (hist.sum() + 1e-9)).astype(np.float32)

def feats_for_patch(img_rgb):
    g_float = rgb2gray(img_rgb).astype(np.float32)
    g_norm  = (g_float - np.median(g_float)) / (np.std(g_float) + 1e-6)

    gabor_feats = []
    for f in GABOR_FREQS:
        for t in GABOR_THETAS:
            real, imag = gabor(g_norm, frequency=f, theta=t)
            gabor_feats += [real.mean(), real.std(), imag.mean(), imag.std()]

    g_u8 = img_as_ubyte(exposure.rescale_intensity(g_float, in_range='image'))
    lbp = local_binary_pattern(g_u8, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp, bins=np.arange(0, 8+3), range=(0, 8+2), density=True)

    rps = radial_power_spectrum(g_norm, nbins=16)
    stats = np.array([g_norm.mean(), g_norm.std(), np.median(g_norm)], dtype=np.float32)

    return np.hstack([np.float32(gabor_feats), np.float32(hist_lbp), rps, stats])

def extract_patches(img_rgb, patch_size=PATCH_SIZE, stride=STRIDE):
    H, W, _ = img_rgb.shape
    if H < patch_size or W < patch_size:
        return np.empty((0, patch_size, patch_size, 3), dtype=np.uint8), (0,0), []
    ny = max(1, 1 + (H - patch_size) // stride)
    nx = max(1, 1 + (W - patch_size) // stride)
    patches, coords = [], []
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy*stride, ix*stride
            patch = img_rgb[y0:y0+patch_size, x0:x0+patch_size]
            if patch.shape[:2] == (patch_size, patch_size):
                patches.append(patch); coords.append((y0, x0))
    patches = np.stack(patches, axis=0) if patches else np.empty((0, patch_size, patch_size, 3), dtype=np.uint8)
    return patches, (ny, nx), coords

# ======================== Heatmap & Boxes =======================
def heatmap_to_image(score_grid, img, patch_size=PATCH_SIZE, stride=STRIDE, alpha=0.45):
    H, W = img.shape[:2]
    ny, nx = score_grid.shape
    heat = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            s = score_grid[iy, ix]
            y0, x0 = iy*stride, ix*stride
            heat[y0:y0+patch_size, x0:x0+patch_size] += s
            count[y0:y0+patch_size, x0:x0+patch_size] += 1.0
    heat = heat / np.maximum(count, 1.0)
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat_color + (1 - alpha) * img).astype(np.uint8)
    return heat, overlay, heat_color

def nms(boxes, scores, iou_thr=0.25):
    if not boxes: return []
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    x1, y1 = boxes[:,0], boxes[:,1]
    x2, y2 = boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1); h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep

def grid_to_boxes(score_grid, img_shape, thr_perc=THR_PERC, min_box_ratio=MIN_BOX, nms_iou=NMS_IOU):
    H, W = img_shape[:2]
    heat = cv2.resize(score_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    thr = np.percentile(heat, thr_perc)
    mask = (heat >= thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((7,7), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, scores = [], []
    min_area = min_box_ratio * (H*W)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area: 
            continue
        s = float(heat[y:y+h, x:x+w].mean())
        boxes.append((x,y,w,h)); scores.append(s)
    keep = nms(boxes, scores, iou_thr=nms_iou)
    return [boxes[i] for i in keep], [scores[i] for i in keep], mask, heat

# ======================== Train & Score =========================
def load_training_patches_from_dir(img_dir: Path):
    files = sorted(list_images(img_dir))
    print(f"[TREINO] {img_dir} -> {len(files)} arquivo(s)")
    X_feats = []
    for p in files:
        bgr = imread_unicode(p)
        if bgr is None:
            print(f"[AVISO] não li {p}"); 
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        patches, _, _ = extract_patches(rgb, PATCH_SIZE, max(STRIDE, PATCH_SIZE//2))
        if len(patches) == 0:
            patches = np.array([cv2.resize(rgb, (PATCH_SIZE, PATCH_SIZE))])
        feats = np.vstack([feats_for_patch(px) for px in patches])
        X_feats.append(feats)
    return np.vstack(X_feats) if X_feats else None

def train_model():
    X = None; trained_on = None
    if DIR_PADRAO.exists():
        X = load_training_patches_from_dir(DIR_PADRAO)
        if X is not None: trained_on = "PADRÃO (normais)"
    if X is None and DIR_FALHAS.exists():
        X = load_training_patches_from_dir(DIR_FALHAS)
        if X is not None: trained_on = "FALHAS (fallback)"
    if X is None:
        raise SystemExit("[ERRO] Sem dados de treino em dataset/padrao/ e dataset/falhas/")
    print(f"[TREINO] patches={X.shape[0]} | dim={X.shape[1]} | origem={trained_on}")
    scaler = StandardScaler().fit(X)
    oc = OneClassSVM(kernel="rbf", nu=NU, gamma=GAMMA).fit(scaler.transform(X))
    return scaler, oc, trained_on

def score_image(scaler, oc, rgb, trained_on):
    patches, grid, _ = extract_patches(rgb, PATCH_SIZE, STRIDE)
    if len(patches) == 0: 
        return None, None, None
    X = np.vstack([feats_for_patch(px) for px in patches])
    d = oc.decision_function(scaler.transform(X)).ravel()
    a = -d if "PADRÃO" in trained_on else d
    mn, mx = np.percentile(a, 1), np.percentile(a, 99)
    score = np.clip((a - mn) / (mx - mn + 1e-6), 0, 1)
    ny, nx = grid
    return score.reshape(ny, nx), ny, nx

# ======================== Main ================================
def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # teste de escrita
    test_ok = imwrite_unicode(OUT_DIR / "teste.png", np.zeros((10,10,3), np.uint8))
    print(f"[TESTE] write teste.png -> {test_ok}")

    scaler, oc, trained_on = train_model()
    analis = sorted(list_images(DIR_ANALIS))
    print(f"[RUN] imagens para analisar: {len(analis)}")
    if not analis:
        print(f"[ERRO] Coloque JPG/PNG em {DIR_ANALIS}")
        return

    for p in analis:
        bgr = imread_unicode(p)
        if bgr is None:
            print(f"[AVISO] não li {p.name}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        score_grid, ny, nx = score_image(scaler, oc, rgb, trained_on)
        if score_grid is None:
            print(f"[AVISO] {p.name}: sem patches (PATCH_SIZE={PATCH_SIZE} grande?)")
            continue

        heat, overlay, heat_color = heatmap_to_image(score_grid, rgb, PATCH_SIZE, STRIDE, alpha=0.45)
        boxes, scores, mask, heat_up = grid_to_boxes(score_grid, rgb.shape)

        boxed = rgb.copy()
        for (x,y,w,h), s in zip(boxes, scores):
            cv2.rectangle(boxed, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(boxed, f"{s:.2f}", (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        def save(name, arr):
            ok = imwrite_unicode(OUT_DIR / name, arr)
            print(f"[SAVE] {name} -> {ok}")

        save(f"{p.stem}_orig.png",    cv2.cvtColor(rgb,        cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_overlay.png", cv2.cvtColor(overlay,    cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_boxes.png",   cv2.cvtColor(boxed,      cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_heat.png",    cv2.cvtColor(heat_color, cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_mask.png",    mask)

        print(f"[OK] {p.name}: ny={ny} nx={nx} boxes={len(boxes)}")

if __name__ == "__main__":
    run()
