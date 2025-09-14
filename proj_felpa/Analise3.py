[código omitido acima para brevidade — agora inclui tudo aqui]

# (continuação do detector_falhas_gabor_v2.py)

# ======================== Heatmap & Análise ========================
def score_grid(scaler, oc, rgb, trained_on, patch_size, stride):
    patches, grid, _ = extract_patches(rgb, patch_size, stride)
    if len(patches) == 0: return None
    X = np.vstack([feats_for_patch(px) for px in patches])
    d = oc.decision_function(scaler.transform(X)).ravel()
    a = -d if "PADRAO" in trained_on else d
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
        if heat is None: continue
        any_scale = True
        accum = np.maximum(accum, heat)
    if not any_scale:
        return None
    if accum.max() > accum.min():
        accum = (accum - accum.min()) / (accum.max() - accum.min())
    return accum

def colorize_heat(heat, rgb, alpha=0.45):
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat_color + (1 - alpha) * rgb).astype(np.uint8)
    return heat_color, overlay

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
    H, W = heat.shape[:2]
    thr = np.percentile(heat, thr_perc)
    mask = (heat >= thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, scores = [], []
    min_area = float(min_box_ratio) * (H * W)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area: continue
        s = float(heat[y:y+h, x:x+w].mean())
        boxes.append((x,y,w,h)); scores.append(s)
    keep = nms(boxes, scores, iou_thr=nms_iou)
    return [boxes[i] for i in keep], [scores[i] for i in keep], mask

# ======================== Execução ===============================
def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[INFO] Treinando modelo...")
    scaler, oc, trained_on = train_model(SCALES)
    imagens = sorted(list_images(DIR_ANALIS))
    if not imagens:
        print(f"[ERRO] Coloque JPG/PNG em {DIR_ANALIS}")
        return

    csv_rows = [("imagem", "x", "y", "w", "h", "score")]
    for p in imagens:
        bgr = imread_unicode(p)
        if bgr is None:
            print(f"[AVISO] não li {p.name}"); continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        heat = analyze_multi_scale(scaler, oc, rgb, trained_on, SCALES)
        if heat is None:
            print(f"[ERRO] Falha ao gerar heatmap para {p.name}")
            continue

        heat_color, overlay = colorize_heat(heat, rgb, alpha=0.45)
        boxes, scores, mask = boxes_from_pixel_heat(heat)

        boxed = rgb.copy()
        for (x,y,w,h), s in zip(boxes, scores):
            cv2.rectangle(boxed, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(boxed, f"{s:.2f}", (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            csv_rows.append((p.name, x, y, w, h, round(s,4)))

        def save(name, arr): imwrite_unicode(OUT_DIR / name, arr)
        save(f"{p.stem}_orig.png",    cv2.cvtColor(rgb,        cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_overlay.png", cv2.cvtColor(overlay,    cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_boxes.png",   cv2.cvtColor(boxed,      cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_heat.png",    cv2.cvtColor(heat_color, cv2.COLOR_RGB2BGR))
        save(f"{p.stem}_mask.png",    mask)
        print(f"[OK] {p.name}: {len(boxes)} falhas detectadas")

    with open(OUT_DIR / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
        print(f"[CSV] resultados.csv salvo com {len(csv_rows)-1} registros")

if __name__ == "__main__":
    run()
