"""PDF report generation for SubspaceAD anomaly detection results."""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

# --------------- regex patterns for run.log parsing ---------------
CATEGORY_RE = re.compile(r"--- Processing Category: (.+?) ---")
PCA_RE = re.compile(r"PCA: selected k=(\d+) components to explain ([\d.]+)%")
FEATURE_RE = re.compile(r"Feature dim: (\d+), Tokens per image: (\d+)")
TRAIN_TEST_RE = re.compile(r"Train: (\d+) \| Test: (\d+) \((\d+) good, (\d+) anomalous\)")
TOTAL_TIME_RE = re.compile(r"Total inference time: ([\d.]+) s")
AVG_TIME_RE = re.compile(r"Avg\. time per image: ([\d.]+) s")
FPS_RE = re.compile(r"Images per second \(FPS\): ([\d.]+)")
LAYER_WARN_RE = re.compile(r"Using valid layers: \[(.+?)\]")

# key config params to show (key, display_label)
CONFIG_KEYS = [
    ("model_ckpt", "Model"),
    ("image_res", "Resolution"),
    ("layers", "Layers"),
    ("agg_method", "Aggregation"),
    ("k_shot", "K-Shot"),
    ("aug_count", "Augmentations"),
    ("pca_ev", "PCA Variance"),
    ("score_method", "Score Method"),
    ("img_score_agg", "Image Score Agg."),
    ("batch_size", "Batch Size"),
    ("bg_mask_method", "Background Mask"),
    ("use_specular_filter", "Specular Filter"),
    ("drop_k", "Drop K"),
    ("whiten", "PCA Whitening"),
]


def _fmt(val, fmt=".4f"):
    """Format a metric value; return 'N/A' for NaN/None."""
    if val is None:
        return "N/A"
    try:
        if np.isnan(float(val)):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return f"{float(val):{fmt}}"


# ----------------------------------------------------------------
#  Log parsing
# ----------------------------------------------------------------
def _parse_log(outdir: str) -> dict:
    log_path = os.path.join(outdir, "run.log")
    if not os.path.exists(log_path):
        return {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    sections = CATEGORY_RE.split(text)
    cats = {}
    for i in range(1, len(sections), 2):
        name = sections[i]
        body = sections[i + 1] if i + 1 < len(sections) else ""
        info = {}
        m = PCA_RE.search(body)
        if m:
            info["pca_k"] = int(m.group(1))
            info["pca_var"] = float(m.group(2))
        m = FEATURE_RE.search(body)
        if m:
            info["feat_dim"] = int(m.group(1))
            info["tokens"] = int(m.group(2))
        m = TRAIN_TEST_RE.search(body)
        if m:
            info["n_train"] = int(m.group(1))
            info["n_test"] = int(m.group(2))
            info["n_good"] = int(m.group(3))
            info["n_anom"] = int(m.group(4))
        m = TOTAL_TIME_RE.search(body)
        if m:
            info["time_total"] = float(m.group(1))
        m = AVG_TIME_RE.search(body)
        if m:
            info["time_avg"] = float(m.group(1))
        m = FPS_RE.search(body)
        if m:
            info["fps"] = float(m.group(1))
        cats[name] = info
    return cats


# ----------------------------------------------------------------
#  Diagnostic plots (matplotlib → PNG)
# ----------------------------------------------------------------
def _plot_roc_curve(labels, scores, save_path, category):
    """ROC curve with AUC."""
    if len(np.unique(labels)) < 2:
        return False
    fpr, tpr, _ = roc_curve(labels, scores)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {category}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


def _plot_score_histogram(labels, scores, save_path, category):
    """Score distribution histogram for good vs. anomalous."""
    good_scores = scores[labels == 0]
    anom_scores = scores[labels == 1]
    if len(good_scores) == 0 and len(anom_scores) == 0:
        return False

    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.linspace(min(scores), max(scores), 30)
    if len(good_scores) > 0:
        ax.hist(good_scores, bins=bins, alpha=0.6, label=f"Good ({len(good_scores)})", color="green")
    if len(anom_scores) > 0:
        ax.hist(anom_scores, bins=bins, alpha=0.6, label=f"Anomalous ({len(anom_scores)})", color="red")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution — {category}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


def _plot_confusion_matrix(labels, scores, save_path, category):
    """Confusion matrix at optimal F1 threshold + threshold analysis."""
    if len(np.unique(labels)) < 2:
        return False, {}

    # Find optimal F1 threshold
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1s = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0,
    )
    # thresholds has len = len(precision) - 1
    best_idx = np.argmax(f1s[:-1])
    best_thr = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    preds = (scores >= best_thr).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Threshold bias analysis
    total_errors = fp + fn
    if total_errors > 0:
        fp_ratio = fp / total_errors
        fn_ratio = fn / total_errors
    else:
        fp_ratio = fn_ratio = 0.0

    if fp_ratio > 0.6:
        bias = "FP-lastig (zu viele Fehlalarme)"
    elif fn_ratio > 0.6:
        bias = "FN-lastig (zu viele verpasste Defekte)"
    else:
        bias = "Balanced"

    info = {
        "threshold": best_thr,
        "f1": best_f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "bias": bias,
        "fp_ratio": fp_ratio,
        "fn_ratio": fn_ratio,
    }

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Good", "Anomalous"])
    ax.set_yticklabels(["Good", "Anomalous"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — F1={best_f1:.3f}\nThreshold={best_thr:.4f} ({bias})")

    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True, info


def _plot_f1_vs_threshold(labels, scores, save_path, category):
    """F1, Precision, Recall vs. threshold."""
    if len(np.unique(labels)) < 2:
        return False
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1s = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0,
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(thresholds, precision[:-1], label="Precision", alpha=0.8)
    ax.plot(thresholds, recall[:-1], label="Recall", alpha=0.8)
    ax.plot(thresholds, f1s[:-1], label="F1", linewidth=2)
    best_idx = np.argmax(f1s[:-1])
    ax.axvline(thresholds[best_idx], color="gray", linestyle="--", alpha=0.5,
               label=f"Best thr={thresholds[best_idx]:.4f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Analysis — {category}")
    ax.legend(fontsize=8)
    ax.set_xlim(thresholds.min(), thresholds.max())
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


# ----------------------------------------------------------------
#  Image collection
# ----------------------------------------------------------------
def _collect_images(outdir, category, subfolder, max_count=4):
    """Collect sample images from intro_overlays or visualizations."""
    # Try intro_overlays first
    overlay_dir = Path(outdir) / "intro_overlays" / category
    if overlay_dir.is_dir():
        imgs = sorted(overlay_dir.glob("*.png"))[:max_count]
        if imgs:
            return [str(p) for p in imgs]
    # Fallback to visualizations
    vis_dir = Path(outdir) / "visualizations"
    if vis_dir.is_dir():
        pattern = f"{category}_example_*.png"
        imgs = sorted(vis_dir.glob(pattern))[:max_count]
        return [str(p) for p in imgs]
    return []


def _collect_good_images(scores_csv_path, max_count=4):
    """Collect paths to good (OK) test images from the scores CSV."""
    if not os.path.exists(scores_csv_path):
        return []
    df = pd.read_csv(scores_csv_path)
    good_df = df[df["label"] == 0].sort_values("score")  # lowest score = most normal
    return good_df["path"].head(max_count).tolist()


# ----------------------------------------------------------------
#  PDF report
# ----------------------------------------------------------------
class _Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128)
            self.cell(0, 5, "SubspaceAD Report", align="L")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Seite {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)
        self.set_text_color(0)

    def kv_row(self, key, value, bold_val=False):
        self.set_font("Helvetica", "", 9)
        self.cell(55, 6, key, border=0)
        self.set_font("Helvetica", "B" if bold_val else "", 9)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")


def generate_report(outdir: str) -> str:
    """Generate a comprehensive PDF report for a SubspaceAD run.

    Args:
        outdir: Path to the run output directory.

    Returns:
        Path to the generated report.pdf, or empty string on failure.
    """
    outdir = str(outdir)
    plots_dir = os.path.join(outdir, "_report_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load data sources
    config = {}
    config_path = os.path.join(outdir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    results_path = os.path.join(outdir, "benchmark_results.csv")
    results_df = pd.read_csv(results_path) if os.path.exists(results_path) else pd.DataFrame()

    log_data = _parse_log(outdir)

    # Load per-category scores
    scores_dir = os.path.join(outdir, "scores")
    category_scores = {}
    if os.path.isdir(scores_dir):
        for csv_file in Path(scores_dir).glob("*.csv"):
            cat = csv_file.stem
            df = pd.read_csv(csv_file)
            category_scores[cat] = df

    # Get categories
    categories = config.get("categories", list(log_data.keys()))
    if not categories and not results_df.empty:
        categories = [c for c in results_df["Category"].tolist() if c != "Average"]

    # ---------- Build PDF ----------
    pdf = _Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ===== PAGE 1: Title + Config =====
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 15, "SubspaceAD", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80)
    pdf.cell(0, 8, "Anomaly Detection Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100)
    pdf.cell(0, 6, datetime.now().strftime("%d.%m.%Y %H:%M"), new_x="LMARGIN", new_y="NEXT", align="C")
    project = config.get("project_name", "")
    if project:
        pdf.cell(0, 6, f"Projekt: {project}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Config table
    pdf.section_title("Konfiguration")
    for key, label in CONFIG_KEYS:
        val = config.get(key, "N/A")
        if val is None:
            val = "-"
        elif isinstance(val, bool):
            val = "Ja" if val else "Nein"
        elif isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        pdf.kv_row(label, val)

    # ===== PAGE 2: Results =====
    pdf.add_page()
    pdf.section_title("Ergebnisse")

    if not results_df.empty:
        metric_cols = ["Image AUROC", "Image AUPR", "Pixel AUROC", "AU-PRO", "Image F1", "Pixel F1"]
        headers = ["Kategorie"] + [c.replace("Image ", "I-").replace("Pixel ", "P-") for c in metric_cols]
        col_w = [65] + [20] * len(metric_cols)

        # Header
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(30, 60, 120)
        pdf.set_text_color(255)
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 7, h, border=1, fill=True, align="C")
        pdf.ln()

        # Rows
        pdf.set_text_color(0)
        for _, row in results_df.iterrows():
            cat = str(row.get("Category", ""))
            is_avg = cat == "Average"
            pdf.set_font("Helvetica", "B" if is_avg else "", 8)
            pdf.cell(col_w[0], 6, cat[:35], border=1)
            for j, mc in enumerate(metric_cols):
                val = row.get(mc)
                txt = _fmt(val)
                # Color code
                try:
                    v = float(val)
                    if not np.isnan(v):
                        if v >= 0.95:
                            pdf.set_text_color(0, 128, 0)
                        elif v >= 0.80:
                            pdf.set_text_color(200, 140, 0)
                        else:
                            pdf.set_text_color(200, 0, 0)
                except (TypeError, ValueError):
                    pass
                pdf.cell(col_w[j + 1], 6, txt, border=1, align="C")
                pdf.set_text_color(0)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, "Keine Benchmark-Ergebnisse vorhanden.")
        pdf.ln()

    # PCA + Training info per category
    pdf.ln(6)
    pdf.section_title("Training & PCA Details")
    for cat in categories:
        info = log_data.get(cat, {})
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, cat, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        if info:
            pdf.kv_row("Train-Bilder", info.get("n_train", "N/A"))
            pdf.kv_row("Test-Bilder", f"{info.get('n_test', '?')} ({info.get('n_good', '?')} good, {info.get('n_anom', '?')} anomalous)")
            pdf.kv_row("Feature Dim", info.get("feat_dim", "N/A"))
            pdf.kv_row("Tokens/Bild", info.get("tokens", "N/A"))
            pdf.kv_row("PCA Komponenten", f"k={info.get('pca_k', '?')} ({info.get('pca_var', '?')}% Varianz)")
            pdf.kv_row("Inferenzzeit", f"{info.get('time_total', '?')} s ({info.get('fps', '?')} FPS)")
        else:
            pdf.cell(0, 6, "Keine Log-Daten verfuegbar.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # ===== DIAGNOSTIC PAGES per category =====
    for cat in categories:
        if cat not in category_scores:
            continue
        sdf = category_scores[cat]
        labels = sdf["label"].values
        scores = sdf["score"].values

        if len(np.unique(labels)) < 2:
            continue

        # Generate plots
        roc_path = os.path.join(plots_dir, f"{cat}_roc.png")
        hist_path = os.path.join(plots_dir, f"{cat}_hist.png")
        cm_path = os.path.join(plots_dir, f"{cat}_cm.png")
        f1thr_path = os.path.join(plots_dir, f"{cat}_f1thr.png")

        has_roc = _plot_roc_curve(labels, scores, roc_path, cat)
        has_hist = _plot_score_histogram(labels, scores, hist_path, cat)
        has_cm, cm_info = _plot_confusion_matrix(labels, scores, cm_path, cat)
        has_f1thr = _plot_f1_vs_threshold(labels, scores, f1thr_path, cat)

        # === Diagnostics page: ROC + Histogram ===
        pdf.add_page()
        pdf.section_title(f"Diagnostik - {cat}")

        if has_roc:
            pdf.image(roc_path, x=10, w=90)
        if has_hist:
            pdf.image(hist_path, x=105, w=90)

        # === Confusion Matrix + Threshold Analysis ===
        y_after_plots = pdf.get_y() + 5
        if has_cm:
            pdf.set_y(y_after_plots)
            pdf.image(cm_path, x=10, w=75)
        if has_f1thr:
            pdf.image(f1thr_path, x=105, w=90, y=y_after_plots)

        # Threshold details text
        if cm_info:
            pdf.ln(5)
            pdf.section_title("Schwellwert-Analyse")
            pdf.kv_row("Optimaler Schwellwert (F1)", f"{cm_info['threshold']:.4f}", bold_val=True)
            pdf.kv_row("F1-Score", f"{cm_info['f1']:.4f}")
            pdf.kv_row("Precision", f"{cm_info['precision']:.4f}")
            pdf.kv_row("Recall", f"{cm_info['recall']:.4f}")
            pdf.kv_row("True Positives (TP)", str(cm_info['tp']))
            pdf.kv_row("False Positives (FP)", str(cm_info['fp']))
            pdf.kv_row("True Negatives (TN)", str(cm_info['tn']))
            pdf.kv_row("False Negatives (FN)", str(cm_info['fn']))
            pdf.kv_row("Fehler-Bias", cm_info['bias'], bold_val=True)
            pdf.kv_row("FP-Anteil an Fehlern", f"{cm_info['fp_ratio']:.0%}")
            pdf.kv_row("FN-Anteil an Fehlern", f"{cm_info['fn_ratio']:.0%}")

    # ===== VISUALIZATION PAGES =====
    for cat in categories:
        # Anomalous images (overlays)
        anom_imgs = _collect_images(outdir, cat, "intro_overlays", max_count=6)
        if anom_imgs:
            pdf.add_page()
            pdf.section_title(f"Anomalie-Bilder - {cat}")
            _place_image_grid(pdf, anom_imgs, cols=2, img_w=90)

        # Good (OK) images
        scores_csv = os.path.join(outdir, "scores", f"{cat}.csv")
        good_paths = _collect_good_images(scores_csv, max_count=6)
        valid_good = [p for p in good_paths if os.path.exists(p)]
        if valid_good:
            pdf.add_page()
            pdf.section_title(f"OK-Bilder (Good) - {cat}")
            _place_image_grid(pdf, valid_good, cols=3, img_w=58)

    # Save
    report_path = os.path.join(outdir, "report.pdf")
    pdf.output(report_path)
    logging.info(f"PDF report saved to {report_path}")
    return report_path


def _place_image_grid(pdf, image_paths, cols=2, img_w=90):
    """Place images in a grid on the current page."""
    gap = 5
    x_start = pdf.l_margin
    y_start = pdf.get_y()

    for idx, img_path in enumerate(image_paths):
        col = idx % cols
        row = idx // cols
        x = x_start + col * (img_w + gap)

        # Estimate height proportionally (assume ~3:4 aspect)
        img_h = img_w * 0.75
        y = y_start + row * (img_h + 12)

        # Check page overflow
        if y + img_h + 12 > pdf.h - pdf.b_margin:
            pdf.add_page()
            y_start = pdf.get_y()
            y = y_start

        try:
            pdf.image(img_path, x=x, y=y, w=img_w)
        except Exception:
            pdf.set_xy(x, y)
            pdf.set_font("Helvetica", "I", 8)
            pdf.cell(img_w, 6, f"[Bild nicht ladbar]")

        # Caption
        pdf.set_xy(x, y + img_h)
        pdf.set_font("Helvetica", "", 7)
        caption = Path(img_path).stem[:40]
        pdf.cell(img_w, 5, caption, align="C")

    pdf.ln(10)
