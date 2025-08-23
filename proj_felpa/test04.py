# detector.py — Detector de falhas com SVM e visualização
import warnings
warnings.filterwarnings("ignore", message="Applying `local_binary_pattern`")

import csv
from datetime import datetime
from pathlib import Path
import joblib

import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage import img_as_ubyte, exposure
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ========================
# Configurações
# ========================
ROOT        = Path(__file__).resolve().parent
DIR_FALHAS  = ROOT / "dataset" / "falhas"
DIR_ANALIS  = ROOT / "dataset" / "analisar"
OUT_DIR     = ROOT / "outputs"
MODEL_PATH  = ROOT / "modelo_ocsvm.pkl"

PATCH_SIZE  = 128
STRIDE      = 64
THRESH      = 0.75
MIN_REGION_PATCHES = 3
POST_MORPH_KERNEL = 3

GABOR_THETAS = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
GABOR_FREQS  = [0.05, 0.10, 0.20]

# ========================
# Utilidades
# ========================
def imread_unicode(path: Path):
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
    r = r / r.max() * np.pi
    hist, _ = np.histogram(r, bins=nbins, range=(0, np.pi), weights=psd2D)
    return hist.astype(np.float32) / (hist.sum() + 1e-6)

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
    hist_lbp, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10), density=True)

    rps = radial_power_spectrum(g_norm, nbins=16)
    stats = np.array([g_norm.mean(), g_norm.std(), np.median(g_norm)], dtype=np.float32)

    return np.hstack([gabor_feats, hist_lbp, rps, stats])

def extract_patches(img_rgb):
    H, W, _ = img_rgb.shape
    ny = max(1, 1 + (H - PATCH_SIZE) // STRIDE)
    nx = max(1, 1 + (W - PATCH_SIZE) // STRIDE)
    patches, coords = [], []
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * STRIDE, ix * STRIDE
            patch = img_rgb[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
            if patch.shape[0] == PATCH_SIZE and patch.shape[1] == PATCH_SIZE:
                patches.append(patch)
                coords.append((y0, x0))
    patches = np.stack(patches, axis=0) if patches else np.empty((0, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    return patches, (ny, nx), coords

def treinar_modelo():
    X_feats = []
    for p in sorted(DIR_FALHAS.glob("*.jpg")):
        bgr = imread_unicode(p)
        if bgr is None: continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        patches, _, _ = extract_patches(rgb)
        feats = np.vstack([feats_for_patch(px) for px in patches])
        X_feats.append(feats)
    X = np.vstack(X_feats)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = OneClassSVM(kernel="rbf", nu=0.10, gamma="scale").fit(Xs)
    joblib.dump((scaler, model), MODEL_PATH)
    return scaler, model

def score_image(scaler, model, rgb):
    patches, grid, _ = extract_patches(rgb)
    X = np.vstack([feats_for_patch(px) for px in patches])
    Xs = scaler.transform(X)
    score = (model.decision_function(Xs).ravel())
    score = (score - score.min()) / (score.max() - score.min() + 1e-6)
    return score.reshape(grid)

def contar_regioes(score_grid):
    mask = (score_grid >= THRESH).astype(np.uint8)
    if POST_MORPH_KERNEL > 0:
        k = np.ones((POST_MORPH_KERNEL, POST_MORPH_KERNEL), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    num_labels, labels = cv2.connectedComponents(mask, connectivity=4)
    return sum((labels == i).sum() >= MIN_REGION_PATCHES for i in range(1, num_labels)), mask

def salvar_resultados(rgb, score_grid, mask_grid, nome_base):
    H, W = rgb.shape[:2]
    ny, nx = score_grid.shape

    # Heatmap
    heat = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            s = score_grid[iy, ix]
            y0, x0 = iy * STRIDE, ix * STRIDE
            heat[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += s
            count[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] += 1.0
    count[count == 0] = 1.0
    heat /= count
    heat_u8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(OUT_DIR / f"{nome_base}_heatmap.png"), heat_color)

    # Máscara
    mask_resized = cv2.resize(mask_grid.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT_DIR / f"{nome_base}_mask.png"), mask_resized)

    # Overlay com bounding boxes
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = rgb.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imwrite(str(OUT_DIR / f"{nome_base}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        scaler, model = joblib.load(MODEL_PATH)
    else:
        scaler, model = treinar_modelo()

    imagens = sorted(DIR_ANALIS.glob("*.jpg")) + sorted(DIR_ANALIS.glob("*.png"))
    with open(OUT_DIR / "resumo.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "imagem", "falhas"])

        for p in imagens:
            bgr = imread_unicode(p)
            if bgr is None: continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            score_grid = score_image(scaler, model, rgb)
            qtd, mask_grid = contar_regioes(score_grid)
            salvar_resultados(rgb, score_grid, mask_grid, p.stem)
            writer.writerow([datetime.now().isoformat(), p.name, qtd])
            print(f"{p.name}: {qtd} falha(s)")

if __name__ == "__main__":
    main()
