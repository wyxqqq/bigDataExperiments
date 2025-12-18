import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# -----------------------------
# Utils: load UCI HAR files
# -----------------------------
def read_whitespace_table(path: Path) -> np.ndarray:
    # UCI HAR uses whitespace-separated txt files
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python").to_numpy()


def load_metadata(data_dir: Path):
    # activity_labels: id -> name
    act = pd.read_csv(data_dir / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "name"])
    act_id2name = dict(zip(act["id"], act["name"]))

    # features: 561 feature names
    feat = pd.read_csv(data_dir / "features.txt", sep=r"\s+", header=None, names=["idx", "name"])
    feature_names = feat["name"].tolist()
    return act_id2name, feature_names


def load_split(data_dir: Path, split: str):
    # split in {"train","test"}
    split_dir = data_dir / split
    X = read_whitespace_table(split_dir / f"X_{split}.txt")
    y = read_whitespace_table(split_dir / f"y_{split}.txt").reshape(-1)
    subject = read_whitespace_table(split_dir / f"subject_{split}.txt").reshape(-1)
    return X, y, subject


def load_inertial_signal(data_dir: Path, split: str, signal_name: str) -> np.ndarray:
    """
    signal_name examples:
      total_acc_x, total_acc_y, total_acc_z
      body_acc_x, body_acc_y, body_acc_z
      body_gyro_x, body_gyro_y, body_gyro_z
    returns: (n_samples, 128)
    """
    p = data_dir / split / "Inertial Signals" / f"{signal_name}_{split}.txt"
    return read_whitespace_table(p)


# -----------------------------
# Plots
# -----------------------------
def plot_class_distribution(y, act_id2name, outpath: Path):
    names = [act_id2name[i] for i in sorted(np.unique(y))]
    counts = [int((y == i).sum()) for i in sorted(np.unique(y))]

    plt.figure()
    plt.bar(names, counts)
    plt.xticks(rotation=25, ha="right")
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pca_scatter(X, y, act_id2name, outpath: Path, max_points=4000, seed=42):
    # subsample for speed/clarity
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    if n > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)

    Xs = X[idx]
    ys = y[idx]

    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(StandardScaler().fit_transform(Xs))

    plt.figure()
    for cls in sorted(np.unique(ys)):
        m = ys == cls
        plt.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.6, label=act_id2name[int(cls)])
    plt.title("PCA (2D) on handcrafted features")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_raw_magnitude_examples(data_dir: Path, act_id2name, outpath: Path, split="train", seed=0):
    """
    Plot magnitude of total acceleration for one random sample per activity.
    magnitude = sqrt(x^2 + y^2 + z^2)
    """
    rng = np.random.default_rng(seed)
    Xax = load_inertial_signal(data_dir, split, "total_acc_x")
    Xay = load_inertial_signal(data_dir, split, "total_acc_y")
    Xaz = load_inertial_signal(data_dir, split, "total_acc_z")
    y = read_whitespace_table(data_dir / split / f"y_{split}.txt").reshape(-1)

    mag = np.sqrt(Xax**2 + Xay**2 + Xaz**2)  # (n,128)

    classes = sorted(np.unique(y))
    plt.figure(figsize=(10, 6))
    for cls in classes:
        idxs = np.where(y == cls)[0]
        pick = int(rng.choice(idxs))
        plt.plot(mag[pick], label=f"{act_id2name[int(cls)]} (sample {pick})")

    plt.title("Total Acceleration Magnitude (one sample per activity)")
    plt.xlabel("Time index (128 points/window)")
    plt.ylabel("Magnitude")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# Simple model (optional-but-nice)
# -----------------------------
def train_and_evaluate(X_train, y_train, X_test, y_test, act_id2name, outdir: Path):
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            max_iter=4000,
            n_jobs=-1,
            multi_class="auto"
        ))
    ])

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print(f"[OK] Test accuracy = {acc:.4f}")

    # confusion matrix
    labels = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, pred, labels=labels)

    # save CM figure
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Logistic Regression)")
    plt.colorbar()
    tick_names = [act_id2name[int(i)] for i in labels]
    plt.xticks(range(len(labels)), tick_names, rotation=25, ha="right")
    plt.yticks(range(len(labels)), tick_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=200)
    plt.close()

    # text report
    name_map = {k: act_id2name[int(k)] for k in labels}
    target_names = [name_map[k] for k in labels]
    report = classification_report(y_test, pred, labels=labels, target_names=target_names, digits=4)
    (outdir / "classification_report.txt").write_text(report, encoding="utf-8")
    print("[OK] Saved classification_report.txt and confusion_matrix.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help='Path to "UCI HAR Dataset" folder')
    ap.add_argument("--out_dir", type=str, default="exp5_outputs", help="Output dir for figures/results")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    act_id2name, feature_names = load_metadata(data_dir)
    X_train, y_train, subject_train = load_split(data_dir, "train")
    X_test, y_test, subject_test = load_split(data_dir, "test")

    print("[INFO] Train:", X_train.shape, y_train.shape, " Test:", X_test.shape, y_test.shape)

    # 1) Class distribution (train)
    plot_class_distribution(y_train, act_id2name, out_dir / "class_distribution_train.png")

    # 2) PCA scatter (train)
    plot_pca_scatter(X_train, y_train, act_id2name, out_dir / "pca_scatter_train.png")

    # 3) Raw signal magnitude examples
    plot_raw_magnitude_examples(data_dir, act_id2name, out_dir / "raw_total_acc_magnitude_examples.png", split="train")

    # 4) Simple classifier
    train_and_evaluate(X_train, y_train, X_test, y_test, act_id2name, out_dir)

    print(f"[DONE] All outputs saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
