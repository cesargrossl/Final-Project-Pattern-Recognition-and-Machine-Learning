# ---------------------------------------------------------------
# test01.py — Detector de falhas por textura (Gabor + LBP + FFT)
# OC-SVM + análise single-scale ou multi-escala
# Salvamento robusto no Windows (imwrite_unicode)
# ---------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", message="Applying `local_binary_pattern`")

from pathlib import Path
import argparse
import numpy as np
import cv2

from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage import img_as_ubyte, exposure
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ======================== Caminhos padrão ========================
ROOT        = Path(__file__).resolve().parent
DIR_PADRAO  = ROOT / "dataset" / "padrao"      # treino preferido (normais)
DIR_FALHAS  = ROOT / "dataset" / "falhas"      # fallback (falhas)
DIR_ANALIS  = ROOT / "dataset" / "analisar"    # imagens para analisar
OUT_DIR     = ROOT / "outputs"

# ======================== Hiperparâmetros ========================
# (valores padrão; podem ser alterados por CLI)
PATCH_SIZE  = 128
STRIDE      = 64
NU          = 0.10
GAMMA       = "scale"

# filtros de Gabor
GABOR_THETAS = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
GABOR_FREQS  = [0.05, 0.10, 0.20, 0.30]

# pós-processamento de caixas
THR_PERC = 85
MIN_BOX  = 0.05
NMS_IOU  = 0.25

VERBOSE  = True

# ======================== Utilidades I/O ========================
def imread_unicode(p: Path):
    """Leitura robusta (caminhos com acentos no Windows)."""
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size == 0: return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(p), cv2.IMREAD_COLOR)

def imwrite_unicode(path: Path, image) -> bool:
    """Salva imagem com suporte a Unicode (Windows-safe)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        ext = ".png"; path = path.with_suffix(ext)
    ok, buf = cv2.imencode(ext, image)
    if not ok: return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return cv2.imwrite(str(path), image)

def list_images(d: Path):
    return [*d.glob("*.jpg"), *d.glob("*.jpeg"), *d.glob("*.png"), *d.glob("*.bmp"),
            *d.glob("*.JPG"), *d.glob("*.JPEG"), *d.glob("*.PNG"), *d.glob("*.BMP")]

# ======================== Extração de features ===================
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
    # cinza + normalização robusta
    g_float = rgb2gray(img_rgb).astype(np.float32)
    g_norm  = (g_float - np.median(g_float)) / (np.std(g_float) + 1e-6)

    # Gabor: média e std dos canais real/imag
    gabor_feats = []
    for f in GABOR_FREQS:
        for t in GABOR_THETAS:
            real, imag = gabor(g_norm, frequency=f, theta=t)
            gabor_feats += [real.mean(), real.std(), imag.mean(), imag.std()]

    # LBP uniform (histograma normalizado)
    g_u8 = img_as_ubyte(exposure.rescale_intensity(g_float, in_range='image'))
    lbp = local_binary_pattern(g_u8, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp, bins=np.arange(0, 8+3), range=(0, 8+2), density=True)

    # FFT radial + estatísticas simples
    rps   = radial_power_spectrum(g_norm, nbins=16)
    stats = np.array([g_norm.mean(), g_norm.std(), np.median(g_norm)], dtype=np.float32)

    return np.hstack([np.float32(gabor_feats), np.float32(hist_lbp), rps, stats])

def extract_patches(img_rgb, patch_size, stride):
    """Divide a imagem em patches fixos (sem padding)."""
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

def spread_heat_from_grid(score_grid, img_shape, patch_size, stride):
    """Espalha a grade de scores para um mapa HxW (média nas sobreposições)."""
    H, W = img_shape[:2]
    ny, nx = score_grid.shape
    heat  = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            s = score_grid[iy, ix]
            y0, x0 = iy*stride, ix*stride
            heat[y0:y0+patch_size, x0:x0+patch_size] += s
            count[y0:y0+patch_size, x0:x0+patch_size] += 1.0
    heat = heat / np.maximum(count, 1.0)
    # normaliza 0..1 para visualização/comparação
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    return heat

def colorize_heat(heat, rgb, alpha=0.45):
    """Gera heat colorido (JET) e overlay com a imagem RGB."""
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat_color + (1 - alpha) * rgb).astype(np.uint8)
    return heat_color, overlay

# ======================== Caixas (Pós) ==========================
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

def boxes_from_pixel_heat(heat, min_box_ratio=MIN_BOX, thr_perc=THR_PERC, nms_iou=NMS_IOU):
    """Extrai caixas diretamente de um heat HxW (float 0..1)."""
    H, W = heat.shape[:2]
    thr = np.percentile(heat, thr_perc)
    mask = (heat >= thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((7,7), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, scores = [], []
    min_area = float(min_box_ratio) * (H * W)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area: 
            continue
        s = float(heat[y:y+h, x:x+w].mean())
        boxes.append((x,y,w,h)); scores.append(s)
    keep = nms(boxes, scores, iou_thr=nms_iou)
    return [boxes[i] for i in keep], [scores[i] for i in keep], mask

# ======================== Treino ================================
def load_training_patches_from_dir(img_dir: Path, patch_size, stride):
    files = sorted(list_images(img_dir))
    print(f"[TREINO] {img_dir} -> {len(files)} arquivo(s)")
    X_feats = []
    for p in files:
        bgr = imread_unicode(p)
        if bgr is None:
            print(f"[AVISO] não li {p}"); 
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        patches, _, _ = extract_patches(rgb, patch_size, max(stride, patch_size//2))
        if len(patches) == 0:
            patches = np.array([cv2.resize(rgb, (patch_size, patch_size))])
        feats = np.vstack([feats_for_patch(px) for px in patches])
        X_feats.append(feats)
    return np.vstack(X_feats) if X_feats else None

def train_model(patch_size, stride):
    X = None; trained_on = None
    if DIR_PADRAO.exists():
        X = load_training_patches_from_dir(DIR_PADRAO, patch_size, stride)
        if X is not None: trained_on = "PADRÃO (normais)"
    if X is None and DIR_FALHAS.exists():
        X = load_training_patches_from_dir(DIR_FALHAS, patch_size, stride)
        if X is not None: trained_on = "FALHAS (fallback)"
    if X is None:
        raise SystemExit("[ERRO] Sem dados de treino em dataset/padrao/ e dataset/falhas/")
    print(f"[TREINO] patches={X.shape[0]} | dim={X.shape[1]} | origem={trained_on}")
    scaler = StandardScaler().fit(X)
    oc = OneClassSVM(kernel="rbf", nu=NU, gamma=GAMMA).fit(scaler.transform(X))
    return scaler, oc, trained_on

# ======================== Scoring ===============================
def score_grid(scaler, oc, rgb, trained_on, patch_size, stride):
    patches, grid, _ = extract_patches(rgb, patch_size, stride)
    if len(patches) == 0:
        return None
    X = np.vstack([feats_for_patch(px) for px in patches])
    d = oc.decision_function(StandardScaler(with_mean=False, with_std=False).fit_transform(scaler.transform(X))).ravel()
    # interpretação do sinal
    a = -d if "PADRÃO" in trained_on else d
    mn, mx = np.percentile(a, 1), np.percentile(a, 99)
    score = np.clip((a - mn) / (mx - mn + 1e-6), 0, 1)
    ny, nx = grid
    return score.reshape(ny, nx)

def analyze_single_scale(scaler, oc, rgb, trained_on, patch_size, stride):
    grid = score_grid(scaler, oc, rgb, trained_on, patch_size, stride)
    if grid is None: return None
    return spread_heat_from_grid(grid, rgb.shape, patch_size, stride)

def analyze_multi_scale(scaler, oc, rgb, trained_on, scales):
    H, W = rgb.shape[:2]
    accum = np.zeros((H, W), dtype=np.float32)
    any_scale = False
    for ps, st in scales:
        heat = analyze_single_scale(scaler, oc, rgb, trained_on, ps, st)
        if heat is None: 
            continue
        any_scale = True
        accum = np.maximum(accum, heat)   # pega o mais suspeito entre escalas
    if not any_scale:
        return None
    # normaliza 0..1
    if accum.max() > accum.min():
        accum = (accum - accum.min()) / (accum.max() - accum.min())
    return accum

# ======================== CLI ===============================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=PATCH_SIZE, help="tamanho do patch (single-scale)")
    ap.add_argument("--stride", type=int, default=STRIDE, help="passo do sliding-window (single-scale)")
    ap.add_argument("--multiscale", action="store_true", help="usar análise multi-escala")
    ap.add_argument("--scales", type=str, default="96,48;64,32;48,24",
                    help="lista ps,st separada por ';' (ex: '96,48;64,32')")
    ap.add_argument("--thr", type=int, default=THR_PERC, help="percentil para threshold de caixas")
    ap.add_argument("--minbox", type=float, default=MIN_BOX, help="área mínima da caixa (proporção)")
    ap.add_argument("--out", type=str, default=str(OUT_DIR), help="pasta de saída")
    return ap.parse_args()

# ======================== Main ===============================
def run():
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TESTE] write teste.png ->", imwrite_unicode(out_dir / "teste.png", np.zeros((10,10,3), np.uint8)))

    # se for multi-escala, escolhemos o maior patch para treinar (estável)
    ps_train = args.patch if not args.multiscale else max(int(s.split(",")[0]) for s in args.scales.split(";"))
    st_train = args.stride if not args.multiscale else ps_train // 2

    scaler, oc, trained_on = train_model(ps_train, st_train)

    analis = sorted(list_images(DIR_ANALIS))
    print(f"[RUN] imagens para analisar: {len(analis)} | modelo: {trained_on}")
    if not analis:
        print(f"[ERRO] Coloque JPG/PNG em {DIR_ANALIS}")
        return

    # parse de escalas
    if args.multiscale:
        scales = []
        for pair in args.scales.split(";"):
            ps, st = pair.split(",")
            scales.append((int(ps), int(st)))
        print(f"[INFO] Multi-escala ON -> {scales}")
    else:
        print(f"[INFO] Single-scale -> patch={args.patch} stride={args.stride}")

    for p in analis:
        bgr = imread_unicode(p)
        if bgr is None:
            print(f"[AVISO] não li {p.name}"); 
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # heat por pixel
        if args.multiscale:
            heat = analyze_multi_scale(scaler, oc, rgb, trained_on, scales)
        else:
            heat = analyze_single_scale(scaler, oc, rgb, trained_on, args.patch, args.stride)

        if heat is None:
            print(f"[AVISO] {p.name}: não foi possível gerar heat (imagem menor que patch?)")
            continue

        # overlay e caixas
        heat_color, overlay = colorize_heat(heat, rgb, alpha=0.45)

        # thresholds mais permissivos para patches menores
        thr_perc  = args.thr if not args.multiscale else min(args.thr, 82)
        min_box_r = args.minbox if not args.multiscale else min(args.minbox, 0.03)

        boxes, scores, mask = boxes_from_pixel_heat(heat, min_box_ratio=min_box_r, thr_perc=thr_perc, nms_iou=NMS_IOU)

        boxed = rgb.copy()
        for (x,y,w,h), s in zip(boxes, scores):
            cv2.rectangle(boxed, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(boxed, f"{s:.2f}", (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # salvar
        def save(name, arr): print(f"[SAVE] {name} ->", imwrite_unicode(out_dir / name, arr))
        save(f"{p.stem}_orig.png",    cv2.cvtColor(rgb,        cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_overlay.png", cv2.cvtColor(overlay,    cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_boxes.png",   cv2.cvtColor(boxed,      cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_heat.png",    cv2.cvtColor(heat_color, cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_mask.png",    mask)

        print(f"[OK] {p.name}: caixas={len(boxes)}")

if __name__ == "__main__":
    run()
