# test01.py — Detector de falhas (Gabor + LBP + FFT) com OC-SVM
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

# ========================
# Caminhos e parâmetros
# ========================
ROOT        = Path(__file__).resolve().parent
DIR_PADRAO  = ROOT / "dataset" / "padrao"
DIR_FALHAS  = ROOT / "dataset" / "falhas"
DIR_ANALIS  = ROOT / "dataset" / "analisar"
OUT_DIR     = ROOT / "outputs"

PATCH_SIZE  = 128
STRIDE      = 64
NU          = 0.10
GAMMA       = "scale"

GABOR_THETAS = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
GABOR_FREQS  = [0.05, 0.10, 0.20, 0.30]

THR_PERC = 85
MIN_BOX  = 0.05    # deixei menor para não filtrar demais
NMS_IOU  = 0.25

VERBOSE = True

# ========================
# Utilidades
# ========================
def imread_unicode(path: Path):
    # leitura robusta (Windows com acentos)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

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
    hist = hist.astype(np.float32)
    s = hist.sum()
    return (hist / s) if s > 0 else hist

def feats_for_patch(img_rgb):
    g_float = rgb2gray(img_rgb).astype(np.float32)
    g_norm  = (g_float - np.median(g_float)) / (np.std(g_float) + 1e-6)

    gabor_feats = []
    for f in GABOR_FREQS:
        for t in GABOR_THETAS:
            real, imag = gabor(g_norm, frequency=f, theta=t)
            gabor_feats += [real.mean(), real.std(), imag.mean(), imag.std()]

    g_rescaled = exposure.rescale_intensity(g_float, in_range='image')
    g_u8 = img_as_ubyte(g_rescaled)
    P, R = 8, 1
    lbp = local_binary_pattern(g_u8, P=P, R=R, method='uniform')
    hist_lbp, _ = np.histogram(lbp, bins=np.arange(0, P + 3), range=(0, P + 2), density=True)

    rps = radial_power_spectrum(g_norm, nbins=16)
    stats = np.array([g_norm.mean(), g_norm.std(), np.median(g_norm)], dtype=np.float32)

    return np.hstack([np.array(gabor_feats, dtype=np.float32), hist_lbp.astype(np.float32), rps, stats])

def extract_patches(img_rgb, patch_size=PATCH_SIZE, stride=STRIDE):
    H, W, _ = img_rgb.shape
    if H < patch_size or W < patch_size:
        return np.empty((0, patch_size, patch_size, 3), dtype=np.uint8), (0,0), []
    ny = max(1, 1 + (H - patch_size) // stride)
    nx = max(1, 1 + (W - patch_size) // stride)
    patches, coords = [], []
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * stride, ix * stride
            patch = img_rgb[y0:y0+patch_size, x0:x0+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
                coords.append((y0, x0))
    patches = np.stack(patches, axis=0) if patches else np.empty((0, patch_size, patch_size, 3), dtype=np.uint8)
    return patches, (ny, nx), coords

def heatmap_to_image(score_grid, img, patch_size=PATCH_SIZE, stride=STRIDE, alpha=0.45):
    H, W = img.shape[:2]
    ny, nx = score_grid.shape
    heat = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            s = score_grid[iy, ix]
            y0, x0 = iy * stride, ix * stride
            heat[y0:y0+patch_size, x0:x0+patch_size] += s
            count[y0:y0+patch_size, x0:x0+patch_size] += 1.0
    count[count == 0] = 1.0
    heat /= count
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat_color + (1 - alpha) * img).astype(np.uint8)
    return heat, overlay, heat_color  # retorna também a versão colorida

def nms(boxes, scores, iou_thr=0.25):
    if len(boxes) == 0: return []
    boxes = np.array(boxes).astype(float)
    scores = np.array(scores).astype(float)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,0]+boxes[:,2]; y2 = boxes[:,1]+boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds+1]
    return keep

# ========================
# Treino
# ========================
def list_images(d):
    return [*d.glob("*.jpg"), *d.glob("*.jpeg"), *d.glob("*.png"), *d.glob("*.bmp")]

def load_training_patches_from_dir(img_dir: Path):
    files = sorted(list_images(img_dir))
    if VERBOSE: print(f"[TREINO] {img_dir} -> {len(files)} arquivos")
    X_feats = []
    for p in files:
        bgr = imread_unicode(p)
        if bgr is None:
            print(f"[AVISO] Falha ao ler {p}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        patches, _, _ = extract_patches(rgb, PATCH_SIZE, max(STRIDE, PATCH_SIZE // 2))
        if len(patches) == 0:
            patches = np.array([cv2.resize(rgb, (PATCH_SIZE, PATCH_SIZE))])
        feats = np.vstack([feats_for_patch(px) for px in patches])
        X_feats.append(feats)
    return np.vstack(X_feats) if X_feats else None

def train_model():
    X = None
    treino = None

    if DIR_PADRAO.exists():
        X = load_training_patches_from_dir(DIR_PADRAO)
        if X is not None:
            treino = "PADRÃO (normais)"

    if X is None and DIR_FALHAS.exists():
        X = load_training_patches_from_dir(DIR_FALHAS)
        if X is not None:
            treino = "FALHAS (fallback)"

    if X is None:
        raise SystemExit(f"[ERRO] Sem dados de treino em {DIR_PADRAO} ou {DIR_FALHAS}")

    print(f"[TREINO] patches: {X.shape[0]} | dim: {X.shape[1]} | origem: {treino}")
    scaler = StandardScaler().fit(X)
    oc = OneClassSVM(kernel="rbf", nu=NU, gamma=GAMMA).fit(scaler.transform(X))
    return scaler, oc, treino

# ========================
# Inferência
# ========================
def score_image(scaler, oc, rgb, trained_on="PADRÃO (normais)"):
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
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]
    return boxes, scores, mask, heat

# ========================
# Execução
# ========================
def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if VERBOSE:
        print(f"[PATHS] padrao={DIR_PADRAO.exists()}  falhas={DIR_FALHAS.exists()}  analisar={DIR_ANALIS.exists()}")
    scaler, oc, trained_on = train_model()

    analis = sorted(list_images(DIR_ANALIS))
    if len(analis) == 0:
        print(f"[ERRO] Nenhuma imagem para analisar em {DIR_ANALIS}")
        return
    print(f"[RUN] analisando {len(analis)} imagem(ns) | modelo: {trained_on}")

    for p in analis:
        bgr = imread_unicode(p)
        if bgr is None:
            print(f"[AVISO] Falha ao ler {p.name}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        score_grid, ny, nx = score_image(scaler, oc, rgb, trained_on)
        if score_grid is None:
            print(f"[AVISO] {p.name}: sem patches (PATCH_SIZE grande demais?)")
            continue

        heat, overlay, heat_color = heatmap_to_image(score_grid, rgb, PATCH_SIZE, STRIDE, alpha=0.45)
        boxes, scores, mask, heat_up = grid_to_boxes(score_grid, rgb.shape)

        boxed = rgb.copy()
        for (x,y,w,h), s in zip(boxes, scores):
            cv2.rectangle(boxed, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(boxed, f"{s:.2f}", (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)

        # salvar (sempre 3 canais onde necessário)
        cv2.imwrite(str(OUT_DIR / f"{p.stem}_overlay.png"),       cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT_DIR / f"{p.stem}_overlay_boxes.png"), cv2.cvtColor(boxed,  cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT_DIR / f"{p.stem}_heat_legend.png"),   cv2.cvtColor(heat_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT_DIR / f"{p.stem}_mask.png"),          mask)

        if VERBOSE:
            print(f"[OK] {p.name}: ny={ny} nx={nx} boxes={len(boxes)} saved -> outputs/")

if __name__ == "__main__":
    run()
