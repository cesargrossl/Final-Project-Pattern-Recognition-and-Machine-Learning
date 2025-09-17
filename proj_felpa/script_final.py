import warnings
warnings.filterwarnings("ignore", message="Applying `local_binary_pattern`")

from pathlib import Path
import numpy as np
import cv2
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.filters import gabor
from skimage import img_as_ubyte, exposure
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy import fft

# ======================== Configuração Otimizada ================
ROOT = Path(__file__).resolve().parent
DIR_PADRAO = ROOT / "dataset" / "padrao"
DIR_FALHAS = ROOT / "dataset" / "falhas"
DIR_ANALIS = ROOT / "dataset" / "analisar"
OUT_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "cache"

# Hiperparâmetros otimizados
SCALES = [(48, 24)]  # Uma escala
NU = 0.005
GAMMA = "scale"

# Features reduzidas
GABOR_THETAS = [0, np.pi/2]  # 2 orientações
GABOR_FREQS = [0.2]  # 1 frequência
GLCM_ANGLES = [0]  # 1 ângulo
GLCM_DIST = [1]  # 1 distância
GLCM_LEVELS = 32  # Reduzido de 64
LBP_P = 4  # Reduzido de 8
LBP_R = 1
RPS_BINS = 8  # Reduzido de 16

# Pós-processamento
THR_PERC = 99
MIN_BOX = 0.01
NMS_IOU = 0.25
VERBOSE = True

# Performance
USE_MULTIPROCESSING = True
N_JOBS = min(4, mp.cpu_count())
BATCH_SIZE = 32  # Reduzido de 50
USE_CACHE = True

# ======================== Cache System ===========================
def get_cache_path(name):
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{name}.pkl"

def save_cache(obj, name):
    if not USE_CACHE:
        return
    try:
        with open(get_cache_path(name), 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        if VERBOSE:
            print(f"[CACHE] Erro ao salvar {name}: {e}")

def load_cache(name):
    if not USE_CACHE:
        return None
    try:
        cache_path = get_cache_path(name)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        if VERBOSE:
            print(f"[CACHE] Erro ao carregar {name}: {e}")
    return None

# ======================== I/O Otimizado ==========================
def imread_unicode_fast(p: Path):
    try:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            return img
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def list_images_fast(d: Path):
    if not d.exists():
        return []
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    return [f for f in d.iterdir() if f.suffix in extensions]

# ======================== Features Otimizadas ====================
def radial_power_spectrum_fast(img_gray, nbins=RPS_BINS):
    h, w = img_gray.shape
    if h * w < 1024:
        f = fft.fftshift(fft.fft2(img_gray))
    else:
        win = np.outer(np.hanning(h), np.hanning(w))
        f = fft.fftshift(fft.fft2(img_gray * win))
    
    psd2D = np.abs(f) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - cx, y - cy)
    r = r / (r.max() + 1e-8) * np.pi
    
    hist, _ = np.histogram(r, bins=nbins, range=(0, np.pi), weights=psd2D)
    return (hist / (hist.sum() + 1e-9)).astype(np.float32)

def compute_gabor_batch(patches, frequencies, thetas):
    # Converter para grayscale e normalizar em lote
    g_floats = np.array([rgb2gray(patch).astype(np.float32) for patch in patches])
    medians = np.median(g_floats, axis=(1, 2), keepdims=True)
    stds = np.std(g_floats, axis=(1, 2), keepdims=True) + 1e-6
    g_norms = (g_floats - medians) / stds
    
    all_features = []
    for g_norm in g_norms:
        gabor_feats = []
        for f in frequencies:
            for t in thetas:
                try:
                    real, imag = gabor(g_norm, frequency=f, theta=t)
                    gabor_feats.extend([real.mean(), real.std(), imag.mean(), imag.std()])
                except:
                    gabor_feats.extend([0.0] * 4)
        all_features.append(np.array(gabor_feats, dtype=np.float32))
    
    return np.vstack(all_features)

def feats_for_patch_fast(img_rgb):
    g_float = rgb2gray(img_rgb).astype(np.float32)
    g_norm = (g_float - np.median(g_float)) / (np.std(g_float) + 1e-6)
    g_u8 = img_as_ubyte(exposure.rescale_intensity(g_float, in_range='image'))
    
    features = []
    
    # 1. Gabor features
    gabor_feats = compute_gabor_batch([img_rgb], GABOR_FREQS, GABOR_THETAS)[0]
    features.extend(gabor_feats)
    
    # 2. LBP
    try:
        lbp = local_binary_pattern(g_u8, P=LBP_P, R=LBP_R, method='uniform')
        hist_lbp, _ = np.histogram(lbp, bins=LBP_P+2, range=(0, LBP_P+1), density=True)
        features.extend(hist_lbp)
    except:
        features.extend([0.0] * (LBP_P+2))
    
    # 3. GLCM
    try:
        glcm = graycomatrix(g_u8, distances=GLCM_DIST, angles=GLCM_ANGLES, 
                           levels=GLCM_LEVELS, symmetric=True, normed=True)
        for prop in ['contrast', 'homogeneity']:
            glcm_feats = graycoprops(glcm, prop).ravel()
            features.extend(glcm_feats)
    except:
        features.extend([0.0] * (len(GLCM_ANGLES) * 2))
    
    # 4. Espectro radial
    try:
        rps = radial_power_spectrum_fast(g_norm, nbins=RPS_BINS)
        features.extend(rps)
    except:
        features.extend([0.0] * RPS_BINS)
    
    # 5. Estatísticas
    try:
        patch_stats = [g_norm.mean(), g_norm.std(), np.median(g_norm)]
        features.extend(patch_stats)
    except:
        features.extend([0.0] * 3)
    
    return np.array(features, dtype=np.float32)

def extract_patches_batch(img_rgb, patch_size, stride):
    H, W, _ = img_rgb.shape
    if H < patch_size or W < patch_size:
        scale = max(patch_size / H, patch_size / W)
        newH, newW = int(H * scale), int(W * scale)
        img_rgb = cv2.resize(img_rgb, (newW, newH), interpolation=cv2.INTER_LINEAR)
        H, W = newH, newW
    
    ny = max(1, 1 + (H - patch_size) // stride)
    nx = max(1, 1 + (W - patch_size) // stride)
    
    patches = np.empty((ny * nx, patch_size, patch_size, 3), dtype=np.uint8)
    coords = []
    
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * stride, ix * stride
            patch = img_rgb[y0:y0+patch_size, x0:x0+patch_size]
            if patch.shape[:2] == (patch_size, patch_size):
                patches[idx] = patch
                coords.append((y0, x0))
                idx += 1
    
    return patches[:idx], (ny, nx), coords[:idx]

# ======================== Processamento em Lote ==================
def process_patches_batch(patches):
    if len(patches) == 0:
        return np.empty((0, 0))
    
    batch_results = []
    for i in range(0, len(patches), BATCH_SIZE):
        batch = patches[i:i+BATCH_SIZE]
        batch_features = compute_gabor_batch(batch, GABOR_FREQS, GABOR_THETAS)
        for j, patch in enumerate(batch):
            features = batch_features[j]
            g_float = rgb2gray(patch).astype(np.float32)
            g_norm = (g_float - np.median(g_float)) / (np.std(g_float) + 1e-6)
            g_u8 = img_as_ubyte(exposure.rescale_intensity(g_float, in_range='image'))
            
            # LBP
            try:
                lbp = local_binary_pattern(g_u8, P=LBP_P, R=LBP_R, method='uniform')
                hist_lbp, _ = np.histogram(lbp, bins=LBP_P+2, range=(0, LBP_P+1), density=True)
                features = np.concatenate([features, hist_lbp])
            except:
                features = np.concatenate([features, [0.0] * (LBP_P+2)])
            
            # GLCM
            try:
                glcm = graycomatrix(g_u8, distances=GLCM_DIST, angles=GLCM_ANGLES, 
                                   levels=GLCM_LEVELS, symmetric=True, normed=True)
                for prop in ['contrast', 'homogeneity']:
                    glcm_feats = graycoprops(glcm, prop).ravel()
                    features = np.concatenate([features, glcm_feats])
            except:
                features = np.concatenate([features, [0.0] * (len(GLCM_ANGLES) * 2)])
            
            # Espectro radial
            try:
                rps = radial_power_spectrum_fast(g_norm, nbins=RPS_BINS)
                features = np.concatenate([features, rps])
            except:
                features = np.concatenate([features, [0.0] * RPS_BINS])
            
            # Estatísticas
            try:
                patch_stats = [g_norm.mean(), g_norm.std(), np.median(g_norm)]
                features = np.concatenate([features, patch_stats])
            except:
                features = np.concatenate([features, [0.0] * 3])
            
            batch_results.append(features)
    
    return np.vstack(batch_results) if batch_results else np.empty((0, 0))

def process_image_parallel(args):
    img_path, scales = args
    bgr = imread_unicode_fast(img_path)
    if bgr is None:
        return None
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    all_features = []
    for ps, st in scales:
        patches, _, _ = extract_patches_batch(rgb, ps, st)
        if len(patches) > 0:
            features = process_patches_batch(patches)
            if features.size > 0:
                all_features.append(features)
    
    if all_features:
        return np.vstack(all_features)
    return None

# ======================== Treino Otimizado ========================
def load_training_data_fast(img_dir: Path, scales):
    files = sorted(list_images_fast(img_dir))
    if not files:
        return None
    
    print(f"[TREINO] {img_dir} -> {len(files)} arquivo(s)")
    
    cache_key = f"training_{img_dir.name}_{hash(str(scales))}"
    cached_data = load_cache(cache_key)
    if cached_data is not None:
        print(f"[CACHE] Dados de treino carregados do cache")
        return cached_data
    
    if USE_MULTIPROCESSING and len(files) > 2:
        args_list = [(f, scales) for f in files]
        with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
            results = list(executor.map(process_image_parallel, args_list))
        
        valid_results = [r for r in results if r is not None]
        if valid_results:
            X = np.vstack(valid_results)
        else:
            X = None
    else:
        X_feats = []
        for p in files:
            result = process_image_parallel((p, scales))
            if result is not None:
                X_feats.append(result)
        
        X = np.vstack(X_feats) if X_feats else None
    
    if X is not None:
        save_cache(X, cache_key)
    
    return X

def train_model_fast(scales):
    model_cache_key = f"model_{hash(str(scales))}"
    cached_model = load_cache(model_cache_key)
    if cached_model is not None:
        print(f"[CACHE] Modelo carregado do cache")
        return cached_model
    
    X = None
    trained_on = None
    
    if DIR_PADRAO.exists():
        X = load_training_data_fast(DIR_PADRAO, scales)
        if X is not None:
            trained_on = "PADRÃO (normais)"
    
    if X is None and DIR_FALHAS.exists():
        X = load_training_data_fast(DIR_FALHAS, scales)
        if X is not None:
            trained_on = "FALHAS (fallback)"
    
    if X is None:
        raise SystemExit("[ERRO] Sem dados de treino")
    
    print(f"[TREINO] patches={X.shape[0]} | dim={X.shape[1]} | origem={trained_on}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    oc = OneClassSVM(kernel="rbf", nu=NU, gamma=GAMMA)
    oc.fit(X_scaled)
    
    model_data = (scaler, oc, trained_on)
    
    save_cache(model_data, model_cache_key)
    
    return model_data

# ======================== Análise Otimizada =======================
def analyze_image_fast(scaler, oc, rgb, trained_on, scales):
    H, W = rgb.shape[:2]
    heat_maps = []
    
    for ps, st in scales:
        patches, grid_shape, coords = extract_patches_batch(rgb, ps, st)
        if len(patches) == 0:
            continue
        
        X = process_patches_batch(patches)
        if X.size == 0:
            continue
        
        scores = oc.decision_function(scaler.transform(X))
        
        if "PADRÃO" in trained_on:
            anomaly_scores = -scores
        else:
            anomaly_scores = scores
        
        if len(anomaly_scores) > 1:
            #p1, p99 = np.percentile(anomaly_scores, [1, 99])
            p1, p99 = np.percentile(anomaly_scores, [10, 99])
            anomaly_scores = np.clip((anomaly_scores - p1) / (p99 - p1 + 1e-6), 0, 1)
        
        ny, nx = grid_shape
        score_grid = anomaly_scores.reshape(ny, nx)
        
        heat = np.zeros((H, W), dtype=np.float32)
        for i, (y0, x0) in enumerate(coords):
            iy, ix = i // nx, i % nx
            if iy < ny and ix < nx:
                score = score_grid[iy, ix]
                heat[y0:y0+ps, x0:x0+ps] = np.maximum(heat[y0:y0+ps, x0:x0+ps], score)
        
        heat_maps.append(heat)
    
    if not heat_maps:
        return None
    
    final_heat = np.maximum.reduce(heat_maps)
    
    if final_heat.max() > final_heat.min():
        final_heat = (final_heat - final_heat.min()) / (final_heat.max() - final_heat.min())
    
    return final_heat

# ======================== Funções Auxiliares ====================
def colorize_heat_fast(heat, rgb, alpha=0.45):
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat_color + (1 - alpha) * rgb).astype(np.uint8)
    return heat_color, overlay

def boxes_from_heat_fast(heat, min_box_ratio=MIN_BOX, thr_perc=THR_PERC, min_score=0.96):
    H, W = heat.shape
    thr = np.percentile(heat, thr_perc)
    mask = (heat >= thr).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes, scores = [], []
    min_area = float(min_box_ratio) * (H * W)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            s = float(heat[y:y+h, x:x+w].mean())
            if s >= min_score:  # filtro de confiança
                boxes.append((x, y, w, h))
                scores.append(s)

    # 🔽 Se houver múltiplos, escolhe só o mais central
    if boxes:
        cx_img, cy_img = W // 2, H // 2
        dists = []
        for (x, y, w, h), s in zip(boxes, scores):
            cx_box, cy_box = x + w // 2, y + h // 2
            dist = np.hypot(cx_box - cx_img, cy_box - cy_img)
            dists.append((dist, s))
        # ordena por proximidade ao centro e score maior
        best_idx = sorted(range(len(boxes)), key=lambda i: (dists[i][0], -dists[i][1]))[0]
        boxes = [boxes[best_idx]]
        scores = [scores[best_idx]]

    return boxes, scores, mask


# ======================== Main Otimizado ===========================
def run_optimized():
    start_time = time.time()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[OTIMIZAÇÃO] Usando {N_JOBS} processos, cache={'ON' if USE_CACHE else 'OFF'}")
    
    print("[TREINO] Iniciando...")
    train_start = time.time()
    scaler, oc, trained_on = train_model_fast(SCALES)
    train_time = time.time() - train_start
    print(f"[TREINO] Concluído em {train_time:.2f}s")
    
    analis = sorted(list_images_fast(DIR_ANALIS))
    print(f"[RUN] {len(analis)} imagens | modelo: {trained_on}")
    
    if not analis:
        print(f"[ERRO] Coloque imagens em {DIR_ANALIS}")
        return
    
    total_detections = 0
    for i, p in enumerate(analis):
        img_start = time.time()
        
        bgr = imread_unicode_fast(p)
        if bgr is None:
            print(f"[AVISO] Não foi possível ler {p.name}")
            continue
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        heat = analyze_image_fast(scaler, oc, rgb, trained_on, SCALES)
        if heat is None:
            print(f"[AVISO] {p.name}: falha na análise")
            continue
        
        boxes, scores, mask = boxes_from_heat_fast(heat)
        
        heat_color, overlay = colorize_heat_fast(heat, rgb)
        
        boxed = rgb.copy()
        for (x, y, w, h), s in zip(boxes, scores):
            cv2.rectangle(boxed, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(boxed, f"{s:.2f}", (x, max(0, y-5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        base_name = p.stem
        cv2.imwrite(str(OUT_DIR / f"{base_name}_original.png"), 
                   cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT_DIR / f"{base_name}_heatmap.png"), 
                   cv2.cvtColor(heat_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT_DIR / f"{base_name}_overlay.png"), 
                   cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT_DIR / f"{base_name}_detections.png"), 
                   cv2.cvtColor(boxed, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT_DIR / f"{base_name}_mask.png"), mask)
        
        img_time = time.time() - img_start
        total_detections += len(boxes)
        
        print(f"[{i+1}/{len(analis)}] {p.name}: {len(boxes)} defeitos em {img_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n[CONCLUÍDO] {len(analis)} imagens, {total_detections} defeitos totais em {total_time:.2f}s")
    print(f"[PERFORMANCE] {len(analis)/total_time:.2f} imagens/s")

if __name__ == "__main__":
    run_optimized()