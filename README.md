# Final-Project-Pattern-Recognition-and-Machine-Learning
Pattern Recognition and Machine Learning


Como usar

Single-scale (padrão 128/64):

python test01.py


Single-scale com áreas menores (64/32):

python test01.py --patch 64 --stride 32 --thr 80 --minbox 0.02


Multi-escala (recomendado):

python test01.py --multiscale --scales "96,48;64,32;48,24" --thr 80 --minbox 0.02
