# final_por_padrao_v2.py — Padrão bom + reforço de faixa horizontal + ROI
# Saídas (em outputs/):
#   *_overlay.png, *_overlay_boxes.png, *_overlay_roi.png, *_heat_legend.png, resumo.csv

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json, csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

# ========================
# Caminhos e parâmetros
# ========================
ROOT        = Path(__file__).resolve().parent
DIR_PADRAO  = ROOT / "dataset" / "padrao"      # patches "bons" (ex.: p01.jpg)
DIR_ANALIS  = ROOT / "dataset" / "analisar"
OUT_DIR     = ROOT / "outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DIR_PADRAO.mkdir(parents=True, exist_ok=True)
DIR_ANALIS.mkdir(parents=True, exist_ok=True)

# Sliding window fino (sensível a linhas horizontais)
PATCH_H, PATCH_W   = 16, 96
STRIDE_Y, STRIDE_X = 8,  32

# Multiescala leve
SCALES = [1.0, 0.9, 1.1]

# Limiar para marcar defeito (score 0..1)
THRESH_DEFECT = 0.45

# NMS
TOPK    = 12
NMS_IOU = 0.25

# ROI padrão caso não exista roi.json (defina ou deixe None p/ campo inteiro)
ROI_PX_DEFAULT = None  # ex.: (1, 1, 144, 421)

# ====== Auxiliar de faixa horizontal ======
LINE_MIN_PROM   = 1.3   # z-score mínimo para considerar “pico” (quanto menor, mais sensível)
LINE_BAND_HALF  = 5     # meia-altura da caixa (total ~ 2*LINE_BAND_HALF)
LINE_SMOOTH_SIG = 1.2   # suavização do gradiente

# ========================
# Utilidades
# ========================
def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0: return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def to_gray_u8(img_rgb):
    """
    Cinza com CLAHE para realçar contraste de linhas fracas.
    g  -> float32 0..1
    g8 -> uint8   0..255
    """
    g = rgb2gray(img_rgb.astype(np.float32) / 255.0)
    g8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g8 = clahe.apply(g8)
    g = g8.astype(np.float32) / 255.0
    return g, g8

def best_dissimilarity(win_g01, ref_g01, ref_u8):
    """SSIM + ZNCC (intensidade) + ZNCC do gradiente horizontal → dissimilaridade 0..1."""
    ssim_val = ssim(win_g01, ref_g01, data_range=1.0)
    win_u8 = (np.clip(win_g01, 0, 1) * 255).astype(np.uint8)
    zncc = cv2.matchTemplate(win_u8, ref_u8, cv2.TM_CCOEFF_NORMED)[0, 0]
    zncc01 = 0.5 * (zncc + 1.0)

    dy_win = cv2.Sobel(win_u8, cv2.CV_32F, 0, 1, ksize=3)
    dy_ref = cv2.Sobel(ref_u8,  cv2.CV_32F, 0, 1, ksize=3)
    dy_win_n = cv2.normalize(dy_win, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dy_ref_n = cv2.normalize(dy_ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    zncc_dy = cv2.matchTemplate(dy_win_n, dy_ref_n, cv2.TM_CCOEFF_NORMED)[0, 0]
    zncc_dy01 = 0.5 * (zncc_dy + 1.0)

    sim = 0.45 * ssim_val + 0.35 * zncc01 + 0.20 * zncc_dy01
    return float(np.clip(1.0 - sim, 0.0, 1.0))

def resize_keep_aspect(img, scale):
    if scale == 1.0: return img
    h, w = img.shape[:2]
    nh, nw = max(4, int(round(h * scale))), max(4, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

def load_reference_patches():
    paths = sorted(list(DIR_PADRAO.glob("*.jpg")) + list(DIR_PADRAO.glob("*.png")))
    if not paths:
        raise SystemExit(f"Nenhum patch de padrão encontrado em {DIR_PADRAO}. "
                         f"Coloque pelo menos p01.jpg.")
    refs = []
    for p in paths:
        bgr = imread_unicode(p)
        if bgr is None:
            print(f"[AVISO] não abriu {p}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        g01, g8 = to_gray_u8(rgb)
        refs.append((rgb, g01, g8))
    return refs

def build_ref_bank(refs):
    """Normaliza todas as refs para PATCH_H×PATCH_W e aplica multiescala."""
    bank = []
    for (_rgb, g01, _u8) in refs:
        for s in SCALES:
            g01s = resize_keep_aspect(g01, s).astype(np.float32)
            g01n = cv2.resize(g01s, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA)
            u8n  = (np.clip(g01n, 0, 1) * 255).astype(np.uint8)
            bank.append((g01n, u8n))
    return bank

def get_roi_mask(H, W):
    """Carrega ROI de roi.json; senão ROI_PX_DEFAULT; senão campo inteiro."""
    cfg_path = ROOT / "roi.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if cfg.get("mode") == "manual_px":
                x1, y1, x2, y2 = map(int, cfg["roi_px"])
                x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
                y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
                m = np.zeros((H, W), np.uint8)
                if x2 > x1 and y2 > y1:
                    m[y1:y2, x1:x2] = 1
                return m
        except Exception as e:
            print(f"[AVISO] falha ao ler roi.json: {e}")

    if ROI_PX_DEFAULT is not None:
        x1, y1, x2, y2 = ROI_PX_DEFAULT
        x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
        m = np.zeros((H, W), np.uint8)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1
        return m

    return np.ones((H, W), np.uint8)

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
    if not boxes: return []
    boxes = np.array(boxes, np.float32)
    scores = np.array(scores, np.float32)
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0 and len(keep) < topk:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1: break
        remain = []
        for j in idxs[1:]:
            if _iou(boxes[i], boxes[j]) < iou_thr:
                remain.append(j)
        idxs = np.array(remain, dtype=int)
    return keep

# --------- Faixa horizontal (linha) ----------
def line_row_scores(img_rgb, roi_mask):
    """
    Retorna vetor por LINHA (H,) com score 0..1 de “faixa horizontal”.
    """
    H, W = img_rgb.shape[:2]
    _, g8 = to_gray_u8(img_rgb)
    dy = cv2.Sobel(g8, cv2.CV_32F, 0, 1, ksize=3)
    dy = cv2.GaussianBlur(np.abs(dy), (0, 0), LINE_SMOOTH_SIG)

    cols = np.where(roi_mask.any(axis=0))[0]
    if cols.size == 0:
        return np.zeros(H, np.float32)
    x1, x2 = int(cols.min()), int(cols.max()) + 1

    line_energy = dy[:, x1:x2].mean(axis=1).astype(np.float32)
    m, s = float(line_energy.mean()), float(line_energy.std() + 1e-6)
    z = (line_energy - m) / s  # z-score

    # mapeia z para 0..1 usando ponto de corte LINE_MIN_PROM (cap em 5σ)
    line_score = (z - LINE_MIN_PROM) / (5.0 - LINE_MIN_PROM)
    line_score = np.clip(line_score, 0.0, 1.0)
    return line_score  # shape (H,)

# ========================
# Pipeline principal
# ========================
def analyze_image(img_rgb, ref_bank, roi_mask):
    H, W = img_rgb.shape[:2]
    img_g01, _ = to_gray_u8(img_rgb)

    ny = max(1, 1 + (H - PATCH_H) // STRIDE_Y) if H >= PATCH_H else 1
    nx = max(1, 1 + (W - PATCH_W) // STRIDE_X) if W >= PATCH_W else 1

    # 1) Dissimilaridade ao padrão por patch
    score_grid = np.zeros((ny, nx), np.float32)
    for iy in range(ny):
        for ix in range(nx):
            cy = iy * STRIDE_Y + PATCH_H // 2
            cx = ix * STRIDE_X + PATCH_W // 2
            if roi_mask[cy, cx] == 0:
                continue
            y0, x0 = iy * STRIDE_Y, ix * STRIDE_X
            tile = img_g01[y0:y0 + PATCH_H, x0:x0 + PATCH_W]
            if tile.shape != (PATCH_H, PATCH_W):
                continue
            diss_best = 1.0
            for ref_g01, ref_u8 in ref_bank:
                d = best_dissimilarity(tile, ref_g01, ref_u8)
                if d < diss_best:
                    diss_best = d
            score_grid[iy, ix] = diss_best

    # 2) Energia de faixa horizontal por linha → expandida para a grade
    line_scores = line_row_scores(img_rgb, roi_mask)  # (H,)
    line_grid = np.zeros_like(score_grid)
    for iy in range(ny):
        y0 = iy * STRIDE_Y
        ys = slice(y0, y0 + PATCH_H)
        line_grid[iy, :] = float(line_scores[ys].mean())

    # 3) Combinação conservadora (pega o pior caso)
    score_grid_final = np.maximum(score_grid, line_grid)

    return score_grid_final, ny, nx, line_grid

def heatmap_overlay(score_grid, img_rgb, alpha=0.45):
    H, W = img_rgb.shape[:2]
    ny, nx = score_grid.shape
    heat = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for iy in range(ny):
        for ix in range(nx):
            s = float(np.clip(score_grid[iy, ix], 0, 1))
            y0, x0 = iy * STRIDE_Y, ix * STRIDE_X
            heat[y0:y0 + PATCH_H, x0:x0 + PATCH_W] += s
            cnt[y0:y0 + PATCH_H, x0:x0 + PATCH_W]  += 1
    cnt[cnt == 0] = 1
    heat /= cnt
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat_color + (1 - alpha) * img_rgb).astype(np.uint8)
    return heat, overlay

def run():
    # Banco de padrões
    refs = load_reference_patches()
    ref_bank = build_ref_bank(refs)

    analis = sorted(list(DIR_ANALIS.glob("*.jpg")) + list(DIR_ANALIS.glob("*.png")))
    if not analis:
        print(f"Nenhuma imagem para analisar em {DIR_ANALIS}")
        return

    with open(OUT_DIR / "resumo.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["imagem", "detectadas", "thr_defect", "patch", "stride"])

        for p in analis:
            bgr = imread_unicode(p)
            if bgr is None:
                print(f"[AVISO] Falha ao ler {p}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]

            roi_mask = get_roi_mask(H, W)

            # Analisar
            score_grid, ny, nx, line_grid = analyze_image(rgb, ref_bank, roi_mask)

            heat, overlay = heatmap_overlay(score_grid, rgb)

            # Desenhar ROI
            overlay_roi = overlay.copy()
            cnts, _ = cv2.findContours((roi_mask * 255).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_roi, cnts, -1, (0, 255, 255), 2)

            # Caixas por limiar
            boxes, scores = [], []
            for iy in range(ny):
                for ix in range(nx):
                    s = float(score_grid[iy, ix])
                    if s >= THRESH_DEFECT:
                        y0, x0 = iy * STRIDE_Y, ix * STRIDE_X
                        boxes.append([x0, y0, x0 + PATCH_W, y0 + PATCH_H])
                        scores.append(s)

            keep = nms(boxes, scores, iou_thr=NMS_IOU, topk=TOPK) if boxes else []
            overlay_boxes = overlay_roi.copy()
            for i in keep:
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(overlay_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(overlay_boxes, f"{scores[i]:.2f}", (x1 + 4, y1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

            # Salvar
            out_overlay       = OUT_DIR / f"{p.stem}_overlay.png"
            out_overlay_roi   = OUT_DIR / f"{p.stem}_overlay_roi.png"
            out_overlay_boxes = OUT_DIR / f"{p.stem}_overlay_boxes.png"
            out_heat_legend   = OUT_DIR / f"{p.stem}_heat_legend.png"

            cv2.imwrite(str(out_overlay),       cv2.cvtColor(overlay,       cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_overlay_roi),   cv2.cvtColor(overlay_roi,   cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_overlay_boxes), cv2.cvtColor(overlay_boxes, cv2.COLOR_RGB2BGR))

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1); plt.title(p.name); plt.imshow(rgb); plt.axis("off")
            plt.subplot(1, 2, 2); plt.title("Score final (0=ok, 1=ruim)")
            im = plt.imshow(score_grid, vmin=0, vmax=1, cmap="jet")
            plt.axis("off"); plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout(); plt.savefig(out_heat_legend, dpi=150); plt.close()

            print(f"Salvo: {out_overlay} | {out_overlay_roi} | {out_overlay_boxes} | {out_heat_legend}")
            w.writerow([p.name, len(keep), THRESH_DEFECT,
                        f"{PATCH_H}x{PATCH_W}", f"{STRIDE_Y}x{STRIDE_X}"])

if __name__ == "__main__":
    run()
