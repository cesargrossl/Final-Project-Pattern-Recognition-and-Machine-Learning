# final.py — Detector de falhas por textura (Gabor+LBP+FFT) com OC-SVM + ROI + heatmap e caixas
# Pastas:
#   dataset/falhas/    -> exemplos (patches) de falha
#   dataset/analisar/  -> imagens a inspecionar
# Saídas (em outputs/):
#   - *_overlay.png
#   - *_overlay_boxes.png
#   - *_heat_legend.png
#   - *_overlay_roi.png      (overlay mostrando a ROI usada)
#   - resumo.csv

import warnings
warnings.filterwarnings("ignore", message="Applying `local_binary_pattern`")

from pathlib import Path
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # evita erro em ambiente sem display
import matplotlib.pyplot as plt

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
DIR_FALHAS  = ROOT / "dataset" / "falhas"
DIR_ANALIS  = ROOT / "dataset" / "analisar"
OUT_DIR     = ROOT / "outputs"

# Patches retangulares (bom para falhas horizontais)
PATCH_H, PATCH_W = 64, 128
STRIDE_Y, STRIDE_X = 32, 64

# OC-SVM
NU     = 0.10
GAMMA  = "scale"

# Banco de Gabor
GABOR_THETAS = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
GABOR_FREQS  = [0.05, 0.10, 0.20]

# Pós-processamento
THRESH  = 0.70          # score (0..1) para desenhar caixa
TOPK    = 12            # no máx. N caixas
NMS_IOU = 0.25          # NMS IoU

# ========================
# ROI (Região de Interesse)
# ========================
# MODOS:
#   ROI_MODE = None           -> não usa ROI (analisa a imagem toda)
#   ROI_MODE = "manual_px"    -> defina ROI_PX = (x1,y1,x2,y2) em pixels
#   ROI_MODE = "manual_rel"   -> defina ROI_REL = (x1,y1,x2,y2) em frações [0..1] da imagem
ROI_MODE = "manual_px"
ROI_PX   = (1, 1, 144, 421)         # <<---- sua ROI medida
ROI_REL  = (0.00, 0.00, 1.00, 1.00) # ignorado quando usamos manual_px

# ========================
# Utilidades
# ========================
def imread_unicode(path: Path):
    """Leitura robusta a caminhos com acentos (Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def radial_power_spectrum(img_gray, nbins=16):
    h, w = img_gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fftshift(np.fft.fft2(img_gray * win))
    psd2D = np.abs(f) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - cx, y - cy)
    r = r / (r.max() + 1e-6) * np.pi
    hist, _ = np.histogram(r, bins=nbins, range=(0, np.pi), weights=psd2D)
    hist = hist.astype(np.float32)
    s = hist.sum()
    return (hist / s) if s > 0 else hist

def feats_for_patch(img_rgb):
    g_float = rgb2gray(img_rgb).astype(np.float32)
    g_norm  = (g_float - np.median(g_float)) / (np.std(g_float) + 1e-6)

    # Gabor
    gabor_feats = []
    for f in GABOR_FREQS:
        for t in GABOR_THETAS:
            real, imag = gabor(g_norm, frequency=f, theta=t)
            gabor_feats += [real.mean(), real.std(), imag.mean(), imag.std()]

    # LBP
    g_rescaled = exposure.rescale_intensity(g_float, in_range='image')
    g_u8 = img_as_ubyte(g_rescaled)
    P, R = 8, 1
    lbp = local_binary_pattern(g_u8, P=P, R=R, method='uniform')
    hist_lbp, _ = np.histogram(lbp, bins=np.arange(0, P + 3),
                               range=(0, P + 2), density=True)

    # FFT radial + stats
    rps = radial_power_spectrum(g_norm, nbins=16)
    stats = np.array([g_norm.mean(), g_norm.std(), np.median(g_norm)], dtype=np.float32)

    return np.hstack([np.array(gabor_feats, dtype=np.float32),
                      hist_lbp.astype(np.float32), rps, stats])

def extract_patches_rect(img_rgb, ph=PATCH_H, pw=PATCH_W, sy=STRIDE_Y, sx=STRIDE_X):
    H, W, _ = img_rgb.shape
    ny = max(1, 1 + (H - ph) // sy) if H >= ph else 1
    nx = max(1, 1 + (W - pw) // sx) if W >= pw else 1
    patches, coords = [], []
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy*sy, ix*sx
            patch = img_rgb[y0:y0+ph, x0:x0+pw]
            if patch.shape[0] == ph and patch.shape[1] == pw:
                patches.append(patch)
                coords.append((y0, x0))
    if patches:
        patches = np.stack(patches, axis=0)
    else:
        patches = np.empty((0, ph, pw, 3), dtype=np.uint8)
    return patches, (ny, nx), coords

def heatmap_to_image(score_grid, img, ph=PATCH_H, pw=PATCH_W, sy=STRIDE_Y, sx=STRIDE_X, alpha=0.45):
    H, W = img.shape[:2]
    ny, nx = score_grid.shape
    heat = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            s = float(score_grid[iy, ix])
            y0, x0 = iy*sy, ix*sx
            heat[y0:y0+ph, x0:x0+pw] += s
            count[y0:y0+ph, x0:x0+pw] += 1.0
    count[count == 0] = 1.0
    heat /= count
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat_color + (1 - alpha) * img).astype(np.uint8)
    return heat, overlay

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def nms(boxes, scores, iou_thr=NMS_IOU, topk=TOPK):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0 and len(keep) < topk:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        remain = []
        for j in idxs[1:]:
            if _iou(boxes[i], boxes[j]) < iou_thr:
                remain.append(j)
        idxs = np.array(remain, dtype=int)
    return keep

# ---------- ROI helpers ----------
def build_roi_mask_for_image(img_rgb):
    """Cria máscara binária (uint8) com 1 dentro da ROI e 0 fora."""
    H, W = img_rgb.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    if ROI_MODE is None:
        mask[:] = 1
        return mask

    if ROI_MODE == "manual_px":
        x1, y1, x2, y2 = ROI_PX
        x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask

    if ROI_MODE == "manual_rel":
        rx1, ry1, rx2, ry2 = ROI_REL
        x1 = int(np.clip(rx1, 0, 1) * W)
        y1 = int(np.clip(ry1, 0, 1) * H)
        x2 = int(np.clip(rx2, 0, 1) * W)
        y2 = int(np.clip(ry2, 0, 1) * H)
        x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask

    mask[:] = 1
    return mask

def grid_valid_mask(mask, ny, nx, ph=PATCH_H, pw=PATCH_W, sy=STRIDE_Y, sx=STRIDE_X):
    """
    Marca como válido se o CENTRO do patch cair dentro da ROI.
    Mais robusto quando a ROI é estreita (como no seu caso).
    """
    H, W = mask.shape
    valid = np.zeros((ny, nx), dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            cy = iy*sy + ph//2
            cx = ix*sx + pw//2
            if 0 <= cy < H and 0 <= cx < W:
                valid[iy, ix] = (mask[cy, cx] == 1)
    return valid

# ========================
# Treino e inferência
# ========================
def train_model_from_falhas():
    falhas = sorted(list(DIR_FALHAS.glob("*.jpg")) + list(DIR_FALHAS.glob("*.png")))
    if not falhas:
        raise SystemExit(f"Nenhuma imagem de falha encontrada em {DIR_FALHAS}")

    X_feats = []
    print(f"Treinando com {len(falhas)} exemplo(s) em {DIR_FALHAS.name}/")
    for p in falhas:
        bgr = imread_unicode(p)
        if bgr is None:
            print(f"[AVISO] Falha ao ler {p}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        patches, _, _ = extract_patches_rect(rgb)
        if len(patches) == 0:
            patches = np.array([cv2.resize(rgb, (PATCH_W, PATCH_H))])

        feats = np.vstack([feats_for_patch(px) for px in patches])
        X_feats.append(feats)

    X = np.vstack(X_feats)
    print(f"Total de patches treino: {len(X)} | Dim(features): {X.shape[1]}")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    oc = OneClassSVM(kernel="rbf", nu=NU, gamma=GAMMA).fit(Xs)
    return scaler, oc

def score_image(scaler, oc, rgb, roi_mask):
    patches, grid, coords = extract_patches_rect(rgb)
    if len(patches) == 0:
        return None, None, None, None, None, None
    X = np.vstack([feats_for_patch(px) for px in patches])
    Xs = scaler.transform(X)
    d = oc.decision_function(Xs).ravel()  # >0 mais parecido com falhas

    # normalização robusta 0..1
    mn, mx = np.percentile(d, 1), np.percentile(d, 99)
    if mx - mn < 1e-9:
        score = np.zeros_like(d)
    else:
        score = (d - mn) / (mx - mn)
    score = np.clip(score, 0, 1)

    ny, nx = grid
    score_grid = score.reshape(ny, nx)

    # Aplica ROI: zera score fora da ROI (centro do patch fora)
    valid = grid_valid_mask(roi_mask, ny, nx)
    score_grid[~valid] = 0.0

    return score_grid, ny, nx, coords, score, valid

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scaler, oc = train_model_from_falhas()

    analis = sorted(list(DIR_ANALIS.glob("*.jpg")) + list(DIR_ANALIS.glob("*.png")))
    if not analis:
        print(f"Nenhuma imagem para analisar em {DIR_ANALIS}")
        return

    with open(OUT_DIR / "resumo.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["imagem", "detectadas", "thr", "patch", "stride", "roi_mode", "roi"])

        print(f"Analisando {len(analis)} imagem(ns) em {DIR_ANALIS.name}/")
        for p in analis:
            bgr = imread_unicode(p)
            if bgr is None:
                print(f"[AVISO] Falha ao ler {p}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # ROI para esta imagem
            roi_mask = build_roi_mask_for_image(rgb)

            score_grid, ny, nx, coords, flat_scores, valid = score_image(scaler, oc, rgb, roi_mask)
            if score_grid is None:
                print(f"[AVISO] {p.name}: sem patches extraídos.")
                continue

            heat, overlay = heatmap_to_image(score_grid, rgb)

            # desenha contorno da ROI no overlay (debug)
            overlay_roi = overlay.copy()
            cnts, _ = cv2.findContours((roi_mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_roi, cnts, -1, (0, 255, 255), 2)  # amarelo

            # Caixas onde score >= THRESH (apenas onde valid=True)
            boxes, scores = [], []
            for iy in range(ny):
                for ix in range(nx):
                    if not valid[iy, ix]:
                        continue
                    s = float(score_grid[iy, ix])
                    if s >= THRESH:
                        y0, x0 = iy * STRIDE_Y, ix * STRIDE_X
                        boxes.append([x0, y0, x0 + PATCH_W, y0 + PATCH_H])
                        scores.append(s)

            keep_idx = nms(boxes, scores, iou_thr=NMS_IOU, topk=TOPK) if boxes else []
            overlay_boxes = overlay_roi.copy()
            for i in keep_idx:
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(overlay_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(overlay_boxes, f"{scores[i]:.2f}", (x1 + 4, y1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

            # salvar imagens
            out_overlay       = OUT_DIR / f"{p.stem}_overlay.png"
            out_overlay_boxes = OUT_DIR / f"{p.stem}_overlay_boxes.png"
            out_overlay_roi   = OUT_DIR / f"{p.stem}_overlay_roi.png"
            out_heat_legend   = OUT_DIR / f"{p.stem}_heat_legend.png"

            cv2.imwrite(str(out_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_overlay_roi), cv2.cvtColor(overlay_roi, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_overlay_boxes), cv2.cvtColor(overlay_boxes, cv2.COLOR_RGB2BGR))

            # Heatmap com barra de cores
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1); plt.title(p.name); plt.imshow(rgb); plt.axis("off")
            plt.subplot(1, 2, 2); plt.title("Similaridade com 'falha' (0→1)")
            im = plt.imshow(score_grid, vmin=0, vmax=1, cmap="jet")
            plt.axis("off"); plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout(); plt.savefig(out_heat_legend, dpi=150); plt.close()

            print(f"Salvo: {out_overlay} | {out_overlay_roi} | {out_overlay_boxes} | {out_heat_legend}")
            w.writerow([p.name, len(keep_idx), THRESH,
                        f"{PATCH_H}x{PATCH_W}", f"{STRIDE_Y}x{STRIDE_X}",
                        ROI_MODE, (ROI_PX if ROI_MODE=='manual_px' else ROI_REL)])

if __name__ == "__main__":
    run()
