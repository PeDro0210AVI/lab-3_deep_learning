# Laboratorio 3 — Deep Learning: Reconocimiento de Lenguaje de Señas (ASL)

CC3084 – Data Science | Semestre II 2026 | Universidad del Valle de Guatemala

## Contexto

SignBridge es una startup guatemalteca de tecnología inclusiva que está desarrollando un
traductor de Lenguaje de Señas Americano (ASL) en tiempo real. Este repositorio contiene el
primer prototipo del motor de reconocimiento: un clasificador de letras del alfabeto ASL a
partir de imágenes de manos.

Dataset: [ASL Alphabet (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Estructura del repositorio

```
data/
  raw/            # dataset descargado de Kaggle (no versionado)
  processed/      # imágenes submuestreadas y redimensionadas (no versionado)
  own_photos/     # fotos de señas hechas por los integrantes del grupo (ejercicio 8)
notebooks/
  asl_lab3.ipynb  # notebook principal: EDA, preprocesamiento, modelos, informe
models/           # pesos de modelos entrenados (no versionado)
reports/          # figuras / informe exportado
src/              # scripts auxiliares reutilizables (descarga, dataset loader, etc.)
```

## Entorno

Este proyecto usa un flake de Nix (`flake.nix`) con Python 3 + PyTorch, scikit-learn,
pandas, matplotlib, seaborn. En Windows, se usó un entorno Python equivalente instalado vía pip.

Para reproducir en Windows:

```
pip install torch pandas matplotlib seaborn scikit-learn pillow kaggle jupyterlab ipykernel opencv-python-headless
```

## Cómo descargar el dataset

1. Crear un token de API en https://www.kaggle.com/settings (sección API → "Create New Token").
2. Colocar el archivo `kaggle.json` descargado en `~/.kaggle/kaggle.json`.
3. Ejecutar:

```
kaggle datasets download -d grassknoted/asl-alphabet -p data/raw --unzip
```

## Submuestra utilizada

Se documenta y justifica en el notebook principal (`notebooks/asl_lab3.ipynb`), sección de
preprocesamiento. En resumen: 600 imágenes por clase (17,400 en total), redimensionadas a
64x64, con partición estratificada 70/15/15 (train/val/test).

## Resultados

| Modelo | Accuracy en test |
| --- | --- |
| **CNN — DeepCNN (lr 3e-4)** — mejor modelo | **98.85%** |
| Random Forest (PCA 100 comp. + 400 árboles, profundidad 30) | 91.00% |
| CNN — DeepCNN con image augmentation | 90.96% |
| MLP simple (fully-connected) | 39.46% |

La CNN profunda fue el mejor modelo. La augmentation (rotación, traslación, brillo/contraste —
sin flips, ver discusión en el notebook sobre por qué el flip horizontal no tiene sentido para
lenguaje de señas) no mejoró el resultado con el presupuesto de épocas usado. El detalle completo
del EDA, la justificación de cada modelo y la reflexión de accesibilidad están en
`notebooks/asl_lab3.ipynb`.

## Integrantes

- Pedro Avila
- Brandon Rivera
- Javier Lopez
