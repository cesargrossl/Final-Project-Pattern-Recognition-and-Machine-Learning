# medir_roi.py
import json, cv2, numpy as np
from pathlib import Path

def imread_unicode(p: Path):
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

ROOT = Path(__file__).resolve().parent
IMG  = ROOT / "dataset" / "analisar" / "01.jpg"   # ajuste se for outro nome

print(f"Tentando abrir: {IMG}")
img = imread_unicode(IMG)
if img is None:
    # debug extra
    print("Falha ao abrir via imdecode. O arquivo existe?", IMG.exists())
    raise SystemExit(f"Falha ao abrir {IMG}")

win = "Selecione a ROI (ENTER confirma, C cancela)"
r = cv2.selectROI(win, img, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

x, y, w, h = map(int, r)
if w == 0 or h == 0:
    print("Nenhuma ROI selecionada. Saindo.")
    raise SystemExit(0)

roi_px = (x, y, x+w, y+h)
print(f"ROI_PX = {roi_px}  |  largura={w}px  altura={h}px")

with open("roi.json", "w", encoding="utf-8") as f:
    json.dump({"mode":"manual_px","roi_px":roi_px}, f, indent=2)
print("Salvo: roi.json")
