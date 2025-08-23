# test02.py — Detector de falhas com SVM e visualização por bounding boxes
import warnings
warnings.filterwarnings("ignore", message="Applying `local_binary_pattern`")

import csv
from datetime import datetime
from pathlib import Path

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
# Caminhos e parâmetros
# ========================
ROOT        = Path(__file__).resolve().parent
DIR_FALHAS  = ROOT / "dataset" / "falhas"
DIR_ANALIS  = ROOT / "dataset" / "analisar"
OUT_DIR     = ROOT / "outputs"

PATCH_SIZE  = 128
STRIDE      = 64

NU          = 0.10
GAMMA       = "scale"

GABOR_THETAS = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
GABOR_FREQS  = [0.05, 0.10, 0.20]

THRESH              = 0.75
MIN_REGION_PATCHES  = 3
POST_MORPH_KERNEL   = 3

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

def heatmap_to_image(score_grid, img, alpha=0.45):
    H, W = img.shape[:2]
    ny, nx = score_grid.shape
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
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat_color + (1 - alpha) * img).astype(np.uint8)
    return heat, overlay

def desenhar_contornos(overlay_img, mask_grid):
    mask_img = cv2.resize(mask_grid.astype(np.uint8), (overlay_img.shape[1], overlay_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 0, 255), 2)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

def salvar_grafico_erros(rgb_img, mask_grid, output_path):
    mask_resized = cv2.resize(mask_grid.astype(np.uint8), (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_img)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', lw=2))
    plt.title("Regiões de Falha Detectadas")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ========================
# Modelo
# ========================
def treinar_modelo_falhas():
    falhas = sorted(list(DIR_FALHAS.glob("*.jpg")) + list(DIR_FALHAS.glob("*.png")))
    if not falhas:
        raise SystemExit(f"Nenhuma imagem de falha encontrada em {DIR_FALHAS}")
    X_feats = []
    for p in falhas:
        bgr = imread_unicode(p)
        if bgr is None:
            print(f"[AVISO] Falha ao ler {p}"); continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        patches, _, _ = extract_patches(rgb, PATCH_SIZE, max(STRIDE, PATCH_SIZE // 2))
        if len(patches) == 0:
            patches = np.array([cv2.resize(rgb, (PATCH_SIZE, PATCH_SIZE))])
        feats = np.vstack([feats_for_patch(px) for px in patches])
        X_feats.append(feats)
    X = np.vstack(X_feats)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    oc = OneClassSVM(kernel="rbf", nu=NU, gamma=GAMMA).fit(Xs)
    return scaler, oc

def score_image_grid(scaler, oc, rgb):
    patches, grid, _ = extract_patches(rgb, PATCH_SIZE, STRIDE)
    if len(patches) == 0:
        return None
    X = np.vstack([feats_for_patch(px) for px in patches])
    Xs = scaler.transform(X)
    d = oc.decision_function(Xs).ravel()
    mn, mx = np.percentile(d, 1), np.percentile(d, 99)
    score = (d - mn) / (mx - mn + 1e-6)
    score = np.clip(score, 0, 1)
    ny, nx = grid
    return score.reshape(ny, nx)

def contar_regioes(score_grid, thresh=THRESH, min_region_patches=MIN_REGION_PATCHES):
    mask = (score_grid >= thresh).astype(np.uint8)
    if POST_MORPH_KERNEL > 0:
        k = np.ones((POST_MORPH_KERNEL, POST_MORPH_KERNEL), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    num_labels, labels = cv2.connectedComponents(mask, connectivity=4)
    count = 0
    for lab in range(1, num_labels):
        area = (labels == lab).sum()
        if area >= min_region_patches:
            count += 1
    return int(count), mask, labels

# ========================
# Execução principal
# ========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scaler, oc = treinar_modelo_falhas()
    analisar = sorted(list(DIR_ANALIS.glob("*.jpg")) + list(DIR_ANALIS.glob("*.png")))
    if not analisar:
        print(f"Nenhuma imagem para analisar em {DIR_ANALIS}")
        return

    resumo_csv = OUT_DIR / "resumo.csv"
    with open(resumo_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "imagem", "detectadas", "threshold", "patch", "stride"])
        print(f"\n=== Resultados (THRESH={THRESH}, PATCH={PATCH_SIZE}, STRIDE={STRIDE}) ===")
        for p in analisar:
            bgr = imread_unicode(p)
            if bgr is None:
                print(f"[ERRO] Não foi possível ler {p.name}"); continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            score_grid = score_image_grid(scaler, oc, rgb)
            if score_grid is None:
                print(f"[AVISO] {p.name}: sem patches.")
                w.writerow([datetime.now().isoformat(), p.name, 0, THRESH, PATCH_SIZE, STRIDE])
                continue

            qtd, mask_grid, _ = contar_regioes(score_grid, THRESH, MIN_REGION_PATCHES)

            _, overlay = heatmap_to_image(score_grid, rgb, alpha=0.45)
            overlay = desenhar_contornos(overlay, mask_grid)

            cv2.imwrite(str(OUT_DIR / f"{p.stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            mask_img = cv2.resize((mask_grid * 255).astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(OUT_DIR / f"{p.stem}_mask.png"), mask_img)

            salvar_grafico_erros(rgb, mask_grid, str(OUT_DIR / f"{p.stem}_grafico.png"))

            print(f"{p.name}: {qtd} região(ões) semelhante(s) à 'falha'")
            w.writerow([datetime.now().isoformat(), p.name, qtd, THRESH, PATCH_SIZE, STRIDE])

    print(f"\nResumo salvo em: {resumo_csv}")

if __name__ == "__main__":
    main()
