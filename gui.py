"""SubspaceAD GUI — tkinter + ttkbootstrap interface for all pipeline parameters."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, END, WORD, DISABLED, NORMAL

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
try:
    from ttkbootstrap.widgets.scrolled import ScrolledText
except ImportError:
    from ttkbootstrap.scrolled import ScrolledText

CONFIG_FILE = Path(__file__).parent / "config_last.json"

# DINOv2 model presets
MODEL_PRESETS = [
    "facebook/dinov2-with-registers-large",
    "facebook/dinov2-with-registers-base",
    "facebook/dinov2-with-registers-small",
    "facebook/dinov2-with-registers-giant",
    "facebook/dinov2-large",
    "facebook/dinov2-base",
    "facebook/dinov2-small",
    "facebook/dinov2-giant",
]

DATASET_TYPES = ["custom", "mvtec_ad", "mvtec_ad2", "visa"]
AGG_METHODS = ["mean", "concat", "group"]
SCORE_METHODS = ["reconstruction", "mahalanobis", "cosine", "euclidean"]
IMG_SCORE_AGGS = ["mtop1p", "max", "mean", "p99", "mtop5"]
BG_MASK_METHODS = ["None", "dino_saliency", "pca_normality"]
MASK_THRESHOLD_METHODS = ["percentile", "otsu"]
KERNEL_PCA_KERNELS = ["rbf", "linear", "poly", "sigmoid", "cosine"]
AUGMENTATIONS = ["rotate", "hflip", "vflip", "color_jitter", "affine"]

# Default layers per model variant (55%-70% deep, from SubspaceAD ablation)
MODEL_DEFAULT_LAYERS = {
    "small":  "-4,-5",                       # 12 layers → layers 7-8
    "base":   "-4,-5",                       # 12 layers → layers 7-8
    "large":  "-7,-8,-9,-10,-11",            # 24 layers → layers 13-17
    "giant":  "-12,-13,-14,-15,-16,-17,-18", # 40 layers → layers 22-28
}


class SubspaceADGui(ttk.Window):
    def __init__(self):
        super().__init__(title="SubspaceAD", themename="darkly", size=(920, 780))
        self.process = None
        self._vars = {}
        self._aug_vars = {}
        self._category_listbox = None

        self._build_ui()
        self._load_config()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Main paned: top=tabs, bottom=log
        pane = ttk.Panedwindow(self, orient="vertical")
        pane.pack(fill=BOTH, expand=True, padx=6, pady=6)

        # Tabs
        nb = ttk.Notebook(pane)
        pane.add(nb, weight=3)

        self._build_projekt_tab(nb)
        self._build_model_tab(nb)
        self._build_augmentation_tab(nb)
        self._build_pca_tab(nb)
        self._build_scoring_tab(nb)
        self._build_masking_tab(nb)
        self._build_specular_tab(nb)
        self._build_options_tab(nb)

        # Bottom: log + controls
        bottom = ttk.Frame(pane)
        pane.add(bottom, weight=2)

        ctrl = ttk.Frame(bottom)
        ctrl.pack(fill=X, pady=(0, 4))

        ttk.Button(ctrl, text="Config speichern", command=self._save_config_dialog, bootstyle="info-outline").pack(side=LEFT, padx=2)
        ttk.Button(ctrl, text="Config laden", command=self._load_config_dialog, bootstyle="info-outline").pack(side=LEFT, padx=2)
        ttk.Separator(ctrl, orient="vertical").pack(side=LEFT, fill=Y, padx=8)
        self._btn_run = ttk.Button(ctrl, text="Train", command=self._run, bootstyle="success")
        self._btn_run.pack(side=LEFT, padx=2)
        self._btn_cancel = ttk.Button(ctrl, text="Abbrechen", command=self._cancel, bootstyle="danger", state=DISABLED)
        self._btn_cancel.pack(side=LEFT, padx=2)
        ttk.Separator(ctrl, orient="vertical").pack(side=LEFT, fill=Y, padx=8)
        ttk.Button(ctrl, text="Ergebnisse oeffnen", command=self._open_results, bootstyle="secondary-outline").pack(side=LEFT, padx=2)

        self._log = ScrolledText(bottom, height=12, wrap=WORD, autohide=True)
        self._log.pack(fill=BOTH, expand=True)

    # -- helpers
    def _make_var(self, key, default=""):
        var = ttk.StringVar(value=str(default))
        self._vars[key] = var
        return var

    def _make_bool(self, key, default=False):
        var = ttk.BooleanVar(value=default)
        self._vars[key] = var
        return var

    def _make_int(self, key, default=0):
        var = ttk.IntVar(value=default)
        self._vars[key] = var
        return var

    def _labeled_entry(self, parent, label, key, default="", width=30, row=None, col=0):
        var = self._make_var(key, default)
        if row is not None:
            ttk.Label(parent, text=label).grid(row=row, column=col, sticky=W, padx=4, pady=2)
            e = ttk.Entry(parent, textvariable=var, width=width)
            e.grid(row=row, column=col + 1, sticky=W, padx=4, pady=2)
        return var

    def _labeled_combo(self, parent, label, key, values, default, row, col=0, width=28):
        var = self._make_var(key, default)
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky=W, padx=4, pady=2)
        cb = ttk.Combobox(parent, textvariable=var, values=values, width=width, state="readonly")
        cb.grid(row=row, column=col + 1, sticky=W, padx=4, pady=2)
        return var

    def _labeled_spin(self, parent, label, key, from_, to, default, row, col=0, width=10, increment=1):
        var = self._make_var(key, default)
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky=W, padx=4, pady=2)
        sb = ttk.Spinbox(parent, textvariable=var, from_=from_, to=to, width=width, increment=increment)
        sb.grid(row=row, column=col + 1, sticky=W, padx=4, pady=2)
        return var

    def _labeled_check(self, parent, label, key, default=False, row=None, col=0):
        var = self._make_bool(key, default)
        cb = ttk.Checkbutton(parent, text=label, variable=var, bootstyle="round-toggle")
        if row is not None:
            cb.grid(row=row, column=col, columnspan=2, sticky=W, padx=4, pady=2)
        return var

    # ---------------------------------------------------------------- TABS
    def _build_projekt_tab(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Projekt")

        self._labeled_entry(f, "Projektname:", "project_name", "my_project", row=0)

        # Dataset path
        ttk.Label(f, text="Dataset-Pfad:").grid(row=1, column=0, sticky=W, padx=4, pady=2)
        path_frame = ttk.Frame(f)
        path_frame.grid(row=1, column=1, sticky=W, padx=4, pady=2)
        self._make_var("dataset_path", "")
        ttk.Entry(path_frame, textvariable=self._vars["dataset_path"], width=55).pack(side=LEFT)
        ttk.Button(path_frame, text="...", command=self._browse_dataset, width=3).pack(side=LEFT, padx=4)

        self._labeled_combo(f, "Dataset-Typ:", "dataset_name", DATASET_TYPES, "custom", row=2)

        # Categories (datasets found under the selected path)
        ttk.Label(f, text="Datensaetze:").grid(row=3, column=0, sticky=NW, padx=4, pady=2)
        cat_outer = ttk.Frame(f)
        cat_outer.grid(row=3, column=1, sticky=(W, E), padx=4, pady=2)
        f.columnconfigure(1, weight=1)

        tree_frame = ttk.Frame(cat_outer)
        tree_frame.pack(side=LEFT, fill=BOTH, expand=True)
        self._category_listbox = ttk.Treeview(
            tree_frame, height=6, show="headings", selectmode="extended",
            columns=("name", "train", "test"),
        )
        self._category_listbox.heading("name", text="Name", anchor=W)
        self._category_listbox.heading("train", text="Train", anchor=CENTER)
        self._category_listbox.heading("test", text="Test", anchor=CENTER)
        self._category_listbox.column("name", width=340, minwidth=200, stretch=True)
        self._category_listbox.column("train", width=80, minwidth=60, stretch=False)
        self._category_listbox.column("test", width=120, minwidth=80, stretch=False)
        cat_scroll_x = ttk.Scrollbar(tree_frame, orient=HORIZONTAL, command=self._category_listbox.xview)
        self._category_listbox.configure(xscrollcommand=cat_scroll_x.set)
        self._category_listbox.pack(side=TOP, fill=BOTH, expand=True)
        cat_scroll_x.pack(side=BOTTOM, fill=X)

        cat_btn_frame = ttk.Frame(cat_outer)
        cat_btn_frame.pack(side=LEFT, padx=4)
        ttk.Button(cat_btn_frame, text="Scan", command=self._scan_categories, width=6).pack(pady=2)
        ttk.Button(cat_btn_frame, text="Alle", command=self._select_all_categories, width=6).pack(pady=2)

        # Output dir
        ttk.Label(f, text="Output-Ordner:").grid(row=4, column=0, sticky=W, padx=4, pady=2)
        out_frame = ttk.Frame(f)
        out_frame.grid(row=4, column=1, sticky=W, padx=4, pady=2)
        self._make_var("outdir", "results")
        ttk.Entry(out_frame, textvariable=self._vars["outdir"], width=55).pack(side=LEFT)
        ttk.Button(out_frame, text="...", command=self._browse_outdir, width=3).pack(side=LEFT, padx=4)

        # Memory bank + model cache
        self._labeled_check(f, "Memory Bank verwenden", "memory_bank", False, row=5)

        ttk.Label(f, text="Model-Cache:").grid(row=6, column=0, sticky=W, padx=4, pady=2)
        cache_frame = ttk.Frame(f)
        cache_frame.grid(row=6, column=1, sticky=W, padx=4, pady=2)
        self._make_var("model_cache_dir", "./models")
        ttk.Entry(cache_frame, textvariable=self._vars["model_cache_dir"], width=55).pack(side=LEFT)
        ttk.Button(cache_frame, text="...", command=self._browse_model_cache, width=3).pack(side=LEFT, padx=4)

    def _build_model_tab(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Modell")

        model_var = self._labeled_combo(f, "Model Checkpoint:", "model_ckpt", MODEL_PRESETS,
                            "facebook/dinov2-with-registers-large", row=0, width=45)
        # allow typing custom values and auto-update layers on model change
        for child in f.winfo_children():
            if isinstance(child, ttk.Combobox):
                child.configure(state="normal")
                child.bind("<<ComboboxSelected>>", lambda e: self._on_model_changed())
                break

        self._labeled_spin(f, "Image Resolution:", "image_res", 56, 1344, 672, row=1, increment=14)
        self._labeled_entry(f, "Patch Size (leer=kein Patching):", "patch_size", "", row=2, width=10)
        self._labeled_spin(f, "Patch Overlap:", "patch_overlap", 0.0, 0.9, 0.0, row=3, increment=0.1)
        self._labeled_spin(f, "Batch Size:", "batch_size", 1, 64, 1, row=4)
        self._labeled_entry(f, "K-Shot (leer=alle):", "k_shot", "", row=5, width=10)
        self._labeled_combo(f, "Aggregation:", "agg_method", AGG_METHODS, "mean", row=6)
        self._labeled_entry(f, "Layers:", "layers", "-7,-8,-9,-10,-11", row=7, width=30)
        self._labeled_entry(f, "Grouped Layers:", "grouped_layers", "", row=8, width=30)
        self._labeled_check(f, "Center Crop (docrop)", "docrop", False, row=9)
        self._labeled_check(f, "CLAHE verwenden", "use_clahe", False, row=10)

    def _build_augmentation_tab(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Augmentation")

        self._labeled_spin(f, "Augmentation-Anzahl pro Bild:", "aug_count", 0, 200, 0, row=0)

        ttk.Label(f, text="Augmentationen:").grid(row=1, column=0, sticky=NW, padx=4, pady=8)
        aug_frame = ttk.Frame(f)
        aug_frame.grid(row=1, column=1, sticky=W, padx=4, pady=8)
        for i, aug in enumerate(AUGMENTATIONS):
            var = self._make_bool(f"aug_{aug}", aug == "rotate")
            self._aug_vars[aug] = var
            ttk.Checkbutton(aug_frame, text=aug, variable=var, bootstyle="round-toggle").grid(
                row=i, column=0, sticky=W, pady=2
            )

        self._labeled_entry(f, "Keine Aug. fuer Kategorien:", "no_aug_categories", "transistor", row=2, width=30)

    def _build_pca_tab(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="PCA")

        self._labeled_entry(f, "PCA Dim (leer=auto):", "pca_dim", "", row=0, width=10)
        self._labeled_spin(f, "Explained Variance:", "pca_ev", 0.5, 1.0, 0.99, row=1, increment=0.01)
        self._labeled_check(f, "Whitening", "whiten", False, row=2)
        self._labeled_check(f, "Kernel PCA verwenden", "use_kernel_pca", False, row=3)
        self._labeled_combo(f, "Kernel:", "kernel_pca_kernel", KERNEL_PCA_KERNELS, "rbf", row=4)
        self._labeled_entry(f, "Gamma (leer=auto):", "kernel_pca_gamma", "", row=5, width=10)

    def _build_scoring_tab(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Scoring")

        self._labeled_combo(f, "Score-Methode:", "score_method", SCORE_METHODS, "reconstruction", row=0)
        self._labeled_spin(f, "Drop K:", "drop_k", 0, 100, 0, row=1)
        self._labeled_combo(f, "Image Score Agg.:", "img_score_agg", IMG_SCORE_AGGS, "mtop1p", row=2)
        self._labeled_spin(f, "PRO Integration Limit:", "pro_integration_limit", 0.0, 1.0, 0.3, row=3, increment=0.1)

    def _build_masking_tab(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Masking")

        self._labeled_combo(f, "BG Mask Methode:", "bg_mask_method", BG_MASK_METHODS, "None", row=0)
        self._labeled_combo(f, "Threshold-Methode:", "mask_threshold_method", MASK_THRESHOLD_METHODS, "percentile", row=1)
        self._labeled_spin(f, "Percentile Threshold:", "percentile_threshold", 0.0, 1.0, 0.15, row=2, increment=0.05)
        self._labeled_spin(f, "DINO Saliency Layer:", "dino_saliency_layer", 0, 30, 6, row=3)

    def _build_specular_tab(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Specular")

        self._labeled_check(f, "Specular Filter aktivieren", "use_specular_filter", False, row=0)
        self._labeled_spin(f, "Specular Tau:", "specular_tau", 0.0, 1.0, 0.6, row=1, increment=0.1)
        self._labeled_spin(f, "Size Threshold Factor:", "specular_size_threshold_factor", 0.0, 10.0, 1.5, row=2, increment=0.1)

    def _build_options_tab(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="Optionen")

        self._labeled_spin(f, "Visualisierungen pro Kat.:", "vis_count", 0, 100, 10, row=0)
        self._labeled_check(f, "Intro Overlays speichern", "save_intro_overlays", False, row=1)
        self._labeled_entry(f, "Debug Limit (leer=aus):", "debug_limit", "", row=2, width=10)
        self._labeled_check(f, "Batched Zero-Shot", "batched_zero_shot", False, row=3)

    # ------------------------------------------------------------ ACTIONS
    def _on_model_changed(self):
        """Auto-set default layers when model changes."""
        ckpt = self._vars["model_ckpt"].get().lower()
        for variant, layers in MODEL_DEFAULT_LAYERS.items():
            if variant in ckpt:
                self._vars["layers"].set(layers)
                break

    def _browse_dataset(self):
        p = filedialog.askdirectory(title="Dataset-Ordner waehlen")
        if p:
            self._vars["dataset_path"].set(p)
            self._scan_categories()

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Output-Ordner waehlen")
        if p:
            self._vars["outdir"].set(p)

    def _browse_model_cache(self):
        p = filedialog.askdirectory(title="Model-Cache Ordner waehlen")
        if p:
            self._vars["model_cache_dir"].set(p)

    def _count_images(self, folder):
        """Count image files in a folder."""
        IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        if not folder.is_dir():
            return 0
        return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS)

    def _describe_dataset(self, cat_path):
        """Return (train_info, test_info) strings for a category folder."""
        train_good = cat_path / "train" / "good"
        n_train = self._count_images(train_good)
        train_str = f"{n_train} good"

        test_dir = cat_path / "test"
        test_parts = []
        if test_dir.is_dir():
            for sub in sorted(test_dir.iterdir()):
                if sub.is_dir():
                    n = self._count_images(sub)
                    test_parts.append(f"{n} {sub.name}")
        test_str = ", ".join(test_parts) if test_parts else "leer"
        return train_str, test_str

    def _add_category(self, cat_name, cat_path):
        """Add a category row to the treeview with train/test info."""
        train_str, test_str = self._describe_dataset(cat_path)
        self._category_listbox.insert(
            "", END, text=cat_name,
            values=(cat_name, train_str, test_str),
        )

    def _scan_categories(self):
        dataset_path = self._vars["dataset_path"].get()
        if not dataset_path or not Path(dataset_path).is_dir():
            return
        self._category_listbox.delete(*self._category_listbox.get_children())
        root = Path(dataset_path)

        # Case 1: selected folder itself is a dataset (has train/ and test/)
        if (root / "train").is_dir() and (root / "test").is_dir():
            self._vars["dataset_path"].set(str(root.parent))
            self._add_category(root.name, root)
            self._category_listbox.selection_set(self._category_listbox.get_children())
            return

        # Case 2: scan direct children for train/ + test/
        try:
            entries = sorted(root.iterdir())
        except PermissionError:
            return
        for d in entries:
            if d.is_dir() and (d / "train").is_dir() and (d / "test").is_dir():
                self._add_category(d.name, d)
        # Auto-select all found categories
        children = self._category_listbox.get_children()
        if children:
            self._category_listbox.selection_set(children)

    def _select_all_categories(self):
        items = self._category_listbox.get_children()
        self._category_listbox.selection_set(items)

    def _get_selected_categories(self):
        return [self._category_listbox.item(i, "values")[0] for i in self._category_listbox.selection()]

    def _build_command(self):
        args = [sys.executable, "main.py"]

        args += ["--dataset_name", self._vars["dataset_name"].get()]
        args += ["--dataset_path", self._vars["dataset_path"].get()]

        cats = self._get_selected_categories()
        if cats:
            args += ["--categories"] + cats

        args += ["--model_ckpt", self._vars["model_ckpt"].get()]
        args += ["--image_res", self._vars["image_res"].get()]

        patch_size = self._vars["patch_size"].get().strip()
        if patch_size:
            args += ["--patch_size", patch_size]

        args += ["--patch_overlap", self._vars["patch_overlap"].get()]
        args += ["--batch_size", self._vars["batch_size"].get()]

        k_shot = self._vars["k_shot"].get().strip()
        if k_shot:
            args += ["--k_shot", k_shot]

        args += ["--agg_method", self._vars["agg_method"].get()]
        args.append(f"--layers={self._vars['layers'].get()}")

        grouped = self._vars["grouped_layers"].get().strip()
        if grouped:
            args += ["--grouped_layers", grouped]

        if self._vars["docrop"].get():
            args.append("--docrop")
        if self._vars["use_clahe"].get():
            args.append("--use_clahe")

        # Augmentation
        args += ["--aug_count", self._vars["aug_count"].get()]
        selected_augs = [a for a in AUGMENTATIONS if self._aug_vars[a].get()]
        if selected_augs:
            args += ["--aug_list"] + selected_augs

        no_aug = self._vars["no_aug_categories"].get().strip()
        if no_aug:
            args += ["--no_aug_categories"] + no_aug.split()

        # PCA
        pca_dim = self._vars["pca_dim"].get().strip()
        if pca_dim:
            args += ["--pca_dim", pca_dim]
        args += ["--pca_ev", self._vars["pca_ev"].get()]
        if self._vars["whiten"].get():
            args.append("--whiten")
        if self._vars["use_kernel_pca"].get():
            args.append("--use_kernel_pca")
            args += ["--kernel_pca_kernel", self._vars["kernel_pca_kernel"].get()]
            gamma = self._vars["kernel_pca_gamma"].get().strip()
            if gamma:
                args += ["--kernel_pca_gamma", gamma]

        # Scoring
        args += ["--score_method", self._vars["score_method"].get()]
        args += ["--drop_k", self._vars["drop_k"].get()]
        args += ["--img_score_agg", self._vars["img_score_agg"].get()]
        args += ["--pro_integration_limit", self._vars["pro_integration_limit"].get()]

        # Masking
        bg = self._vars["bg_mask_method"].get()
        if bg != "None":
            args += ["--bg_mask_method", bg]
        args += ["--mask_threshold_method", self._vars["mask_threshold_method"].get()]
        args += ["--percentile_threshold", self._vars["percentile_threshold"].get()]
        args += ["--dino_saliency_layer", self._vars["dino_saliency_layer"].get()]

        # Specular
        if self._vars["use_specular_filter"].get():
            args.append("--use_specular_filter")
        args += ["--specular_tau", self._vars["specular_tau"].get()]
        args += ["--specular_size_threshold_factor", self._vars["specular_size_threshold_factor"].get()]

        # Output
        args += ["--outdir", self._vars["outdir"].get()]
        args += ["--vis_count", self._vars["vis_count"].get()]
        if self._vars["save_intro_overlays"].get():
            args.append("--save_intro_overlays")
        debug = self._vars["debug_limit"].get().strip()
        if debug:
            args += ["--debug_limit", debug]
        if self._vars["batched_zero_shot"].get():
            args.append("--batched_zero_shot")

        # Memory bank + project name + model cache
        if self._vars["memory_bank"].get():
            args.append("--memory_bank")
        pname = self._vars["project_name"].get().strip()
        if pname:
            args += ["--project_name", pname]
        cache = self._vars["model_cache_dir"].get().strip()
        if cache:
            args += ["--model_cache_dir", cache]

        return args

    def _run(self):
        if not self._vars["dataset_path"].get().strip():
            messagebox.showwarning("Fehler", "Bitte Dataset-Pfad angeben.")
            return

        self._log.text.delete("1.0", END)
        cmd = self._build_command()
        self._log_line(f">>> {subprocess.list2cmdline(cmd)}\n")
        self._save_config()

        self._btn_run.configure(state=DISABLED)
        self._btn_cancel.configure(state=NORMAL)

        def worker():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(Path(__file__).parent),
                )
                for line in self.process.stdout:
                    self._log_line(line)
                self.process.wait()
                rc = self.process.returncode
                self._log_line(f"\n--- Prozess beendet (Exit Code: {rc}) ---\n")
            except Exception as e:
                self._log_line(f"\nFEHLER: {e}\n")
            finally:
                self.process = None
                self.after(0, lambda: self._btn_run.configure(state=NORMAL))
                self.after(0, lambda: self._btn_cancel.configure(state=DISABLED))

        threading.Thread(target=worker, daemon=True).start()

    def _cancel(self):
        if self.process:
            self.process.terminate()
            self._log_line("\n--- Abgebrochen ---\n")

    def _log_line(self, text):
        self.after(0, lambda: self._log_append(text))

    def _log_append(self, text):
        self._log.text.insert(END, text)
        self._log.text.see(END)

    def _open_results(self):
        outdir = self._vars["outdir"].get()
        p = Path(outdir)
        if p.is_dir():
            os.startfile(str(p))
        else:
            messagebox.showinfo("Info", f"Ordner existiert noch nicht:\n{p}")

    # -------------------------------------------------------- CONFIG I/O
    def _get_config_dict(self):
        cfg = {}
        for key, var in self._vars.items():
            cfg[key] = var.get()
        cfg["_selected_categories"] = self._get_selected_categories()
        return cfg

    def _set_config_dict(self, cfg):
        for key, val in cfg.items():
            if key == "_selected_categories":
                continue
            if key in self._vars:
                self._vars[key].set(val)
        # Sync layers to the loaded model
        self._on_model_changed()
        # Restore category selection after scan
        if cfg.get("dataset_path"):
            self._scan_categories()
            sel_cats = cfg.get("_selected_categories", [])
            if sel_cats:
                for item in self._category_listbox.get_children():
                    if self._category_listbox.item(item, "values")[0] in sel_cats:
                        self._category_listbox.selection_add(item)

    def _save_config(self, path=None):
        path = path or CONFIG_FILE
        cfg = self._get_config_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def _load_config(self, path=None):
        path = path or CONFIG_FILE
        if not Path(path).exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._set_config_dict(cfg)
        except Exception:
            pass

    def _save_config_dialog(self):
        p = filedialog.asksaveasfilename(
            title="Config speichern",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialdir=str(Path(__file__).parent),
        )
        if p:
            self._save_config(p)

    def _load_config_dialog(self):
        p = filedialog.askopenfilename(
            title="Config laden",
            filetypes=[("JSON", "*.json")],
            initialdir=str(Path(__file__).parent),
        )
        if p:
            self._load_config(p)


if __name__ == "__main__":
    app = SubspaceADGui()
    app.mainloop()
