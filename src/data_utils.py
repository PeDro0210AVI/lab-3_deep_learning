"""Utilidades de carga, submuestreo y preprocesamiento del dataset ASL Alphabet."""
from pathlib import Path
import numpy as np
from PIL import Image

RAW_TRAIN_DIR = Path("data/raw/asl_alphabet_train/asl_alphabet_train")
RAW_TEST_DIR = Path("data/raw/asl_alphabet_test/asl_alphabet_test")
PROCESSED_DIR = Path("data/processed")

# Orden fijo y estable de clases: A-Z, luego las 3 clases especiales.
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
SPECIAL = ["del", "nothing", "space"]
CLASSES = LETTERS + SPECIAL


def list_class_files(raw_dir: Path = RAW_TRAIN_DIR) -> dict[str, list[Path]]:
    """Regresa un diccionario clase -> lista de rutas de imágenes."""
    out = {}
    for cls in CLASSES:
        cls_dir = raw_dir / cls
        out[cls] = sorted(cls_dir.glob("*.jpg"))
    return out


def load_image(path: Path, size: int = 64) -> np.ndarray:
    """Carga una imagen, la redimensiona a size x size y la regresa como uint8 RGB."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def build_subsample(
    n_per_class: int = 600,
    img_size: int = 64,
    seed: int = 42,
    raw_dir: Path = RAW_TRAIN_DIR,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Toma n_per_class imágenes aleatorias por clase, las redimensiona y regresa
    (X, y, filenames_relativos). X: uint8 [N, size, size, 3]. y: int labels."""
    rng = np.random.default_rng(seed)
    files_by_class = list_class_files(raw_dir)

    xs, ys, names = [], [], []
    for label_idx, cls in enumerate(CLASSES):
        files = files_by_class[cls]
        n = min(n_per_class, len(files))
        chosen = rng.choice(len(files), size=n, replace=False)
        for i in chosen:
            f = files[i]
            xs.append(load_image(f, img_size))
            ys.append(label_idx)
            names.append(f"{cls}/{f.name}")

    X = np.stack(xs).astype(np.uint8)
    y = np.array(ys, dtype=np.int64)
    return X, y, names


def stratified_split(y: np.ndarray, train_frac=0.7, val_frac=0.15, seed=42):
    """Regresa índices (train_idx, val_idx, test_idx) estratificados por clase."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train : n_train + n_val])
        test_idx.append(idx[n_train + n_val :])
    return (
        np.concatenate(train_idx),
        np.concatenate(val_idx),
        np.concatenate(test_idx),
    )


def save_processed(X, y, names, split_name: str, out_dir: Path = PROCESSED_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{split_name}.npz",
        X=X,
        y=y,
        names=np.array(names),
    )


def load_processed(split_name: str, out_dir: Path = PROCESSED_DIR):
    data = np.load(out_dir / f"{split_name}.npz", allow_pickle=True)
    return data["X"], data["y"], data["names"]


if __name__ == "__main__":
    import time

    t0 = time.time()
    X, y, names = build_subsample(n_per_class=600, img_size=64)
    print(f"Cargadas {X.shape[0]} imagenes en {time.time()-t0:.1f}s, shape={X.shape}, dtype={X.dtype}")

    train_idx, val_idx, test_idx = stratified_split(y)
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    save_processed(X[train_idx], y[train_idx], [names[i] for i in train_idx], "train")
    save_processed(X[val_idx], y[val_idx], [names[i] for i in val_idx], "val")
    save_processed(X[test_idx], y[test_idx], [names[i] for i in test_idx], "test")
    print(f"listo en {time.time()-t0:.1f}s total")
