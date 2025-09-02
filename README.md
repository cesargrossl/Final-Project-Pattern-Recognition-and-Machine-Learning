# Detector de Falhas por Textura

Sistema de detecção automática de falhas/anomalias em imagens através da análise de características de textura usando machine learning.

## 🎯 Objetivo

Detectar anomalias e defeitos visuais em superfícies e materiais através da análise multi-escala de texturas, utilizando um modelo One-Class SVM treinado com características extraídas por diferentes técnicas de processamento de imagem.

## 🔧 Características do Sistema

### Extração de Features
O sistema combina múltiplas técnicas de análise de textura:

- **Filtros de Gabor**: Capturam padrões direcionais e de frequência
  - 6 orientações (0°, 30°, 60°, 90°, 120°, 150°)
  - 4 frequências (0.05, 0.10, 0.20, 0.30)

- **LBP (Local Binary Pattern)**: Detecta padrões de textura local
  - 8 pontos com raio 1
  - Histograma uniforme

- **GLCM (Gray-Level Co-occurrence Matrix)**: Analisa relações espaciais
  - Propriedades: contraste, dissimilaridade, homogeneidade, energia, correlação
  - 4 direções angulares

- **Espectro de Potência Radial (FFT)**: Análise de frequências
  - 32 bins radiais no domínio da frequência

- **Estatísticas Descritivas**: Média, desvio padrão, mediana, assimetria, curtose

### Abordagem Multi-escala
Análise em três escalas fixas para capturar defeitos de diferentes tamanhos:
- **96x48** pixels (patch/stride) - Defeitos grandes
- **64x32** pixels (patch/stride) - Defeitos médios  
- **48x24** pixels (patch/stride) - Defeitos pequenos

### Modelo de Machine Learning
- **One-Class SVM** com kernel RBF
- Aprende padrões "normais" e identifica desvios
- Normalização automática com StandardScaler
- Parâmetro `nu=0.10` (10% de outliers esperados)

## 📁 Estrutura de Diretórios

```
projeto/
├── detector_falhas.py          # Código principal
├── dataset/
│   ├── padrao/                 # ⭐ Imagens normais para treinamento
│   │   ├── normal_001.jpg
│   │   ├── normal_002.png
│   │   └── ...
│   ├── falhas/                 # Imagens com defeitos (fallback)
│   │   ├── defeito_001.jpg
│   │   └── ...
│   └── analisar/              # ⭐ Imagens para análise
│       ├── teste_001.jpg
│       ├── teste_002.png
│       └── ...
└── outputs/                   # Resultados gerados automaticamente
    ├── teste_001_orig.png
    ├── teste_001_overlay.png
    ├── teste_001_boxes.png
    ├── teste_001_heat.png
    ├── teste_001_mask.png
    └── ...
```

## 🚀 Como Usar

### 1. Preparar os Dados
- Coloque imagens **normais/padrão** em `dataset/padrao/`
- Coloque imagens **para análise** em `dataset/analisar/`
- Formatos suportados: JPG, JPEG, PNG, BMP

### 2. Executar o Sistema
```bash
python detector_falhas.py
```

### 3. Analisar Resultados
O sistema gera automaticamente para cada imagem analisada:

- **`*_orig.png`**: Imagem original
- **`*_heat.png`**: Mapa de calor das anomalias
- **`*_overlay.png`**: Sobreposição do mapa de calor na imagem
- **`*_boxes.png`**: Caixas delimitadoras das falhas detectadas
- **`*_mask.png`**: Máscara binária das regiões anômalas

## ⚙️ Parâmetros Configuráveis

### Modelo
- **`NU`**: Sensibilidade do modelo (padrão: 0.10)
  - Valores menores = mais sensível
  - 0.05 para falhas muito sutis

- **`GAMMA`**: Parâmetro do kernel RBF ("scale" automático)

### Detecção
- **`THR_PERC`**: Percentil do mapa de calor para threshold (padrão: 75%)
- **`MIN_BOX`**: Área mínima das caixas (padrão: 0.02 = 2% da imagem)
- **`NMS_IOU`**: Threshold de IoU para supressão de caixas sobrepostas (0.25)

### Multi-escala
- **`SCALES`**: Lista de (patch_size, stride) - atualmente fixo em [(96,48), (64,32), (48,24)]

## 📊 Exemplo de Saída

```
[TREINO] dataset/padrao -> 15 arquivo(s)
[TREINO] patches=1250 | dim=187 | origem=PADRÃO (normais)
[RUN] imagens para analisar: 3 | modelo=PADRÃO (normais)
[INFO] Multi-escala FIXO -> [(96, 48), (64, 32), (48, 24)] | thr=75 | min_box=0.02
[OK] teste_001.jpg: caixas=2
[OK] teste_002.jpg: caixas=0  
[OK] teste_003.jpg: caixas=1
```

## 🔍 Aplicações

- **Controle de Qualidade Industrial**: Detecção de defeitos em produtos manufaturados
- **Inspeção de Materiais**: Análise de superfícies, soldas, acabamentos
- **Controle de Processo**: Monitoramento de qualidade em linhas de produção
- **Análise de Imagens Médicas**: Detecção de anomalias em exames (com adaptações)
- **Manutenção Preditiva**: Identificação de deterioração em equipamentos

## 📋 Dependências

```python
numpy
opencv-python
scikit-image
scikit-learn
scipy
pathlib
warnings
```

## 💡 Dicas de Uso

### Para Melhores Resultados:
1. **Use muitas imagens normais** (>20) para treinamento robusto
2. **Imagens similares** às que serão analisadas (iluminação, ângulo, resolução)
3. **Ajuste o parâmetro `NU`** baseado na taxa esperada de defeitos
4. **Ajuste `THR_PERC`** se muitos/poucos defeitos estão sendo detectados

### Troubleshooting:
- **Muitos falsos positivos**: Aumente `THR_PERC` ou `NU`
- **Poucos defeitos detectados**: Diminua `THR_PERC` ou `NU` 
- **Caixas muito pequenas**: Diminua `MIN_BOX`
- **Imagens muito pequenas**: O sistema redimensiona automaticamente

## 📈 Algoritmo

1. **Treinamento**:
   - Extrai patches multi-escala das imagens padrão
   - Calcula features de textura para cada patch
   - Treina One-Class SVM nos features normalizados

2. **Detecção**:
   - Extrai patches da imagem de teste
   - Calcula features e pontua com SVM
   - Gera mapa de calor das anomalias
   - Aplica threshold e morphologia
   - Detecta contornos e gera caixas delimitadoras

## 📄 Licença

Este código é fornecido como exemplo educacional e pode ser adaptado conforme necessário para projetos específicos.