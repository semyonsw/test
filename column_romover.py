#!/usr/bin/env python3
"""
Column Picker
=============

A small desktop tool to load a CSV or Excel file, pick which columns to keep
using checkboxes, and export a new file containing only the selected columns.

Workflow:
    1. Click "Browse..." and select a .csv / .xlsx / .xls file.
    2. (Excel only) Choose the sheet to read.
    3. Tick the columns you want to KEEP (everything is kept by default).
       Use the search box and Select all / Clear all to work quickly.
    4. Click "Export selected columns..." and choose where to save the result.

Dependencies:
    pip install pandas openpyxl

Run:
    python column_romover.py
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import pandas as pd
except ImportError:  # pandas is required to read/write the files.
    pd = None


PREVIEW_ROWS = 100          # how many rows to show in the preview table
MAX_CELL_CHARS = 80         # truncate long cell values in the preview


class ColumnPickerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Column Picker — CSV / Excel column selector")
        self.geometry("960x680")
        self.minsize(760, 540)

        # ----- data state --------------------------------------------------
        self.df = None              # currently loaded DataFrame
        self.excel_file = None      # pd.ExcelFile when an Excel file is open
        self.col_vars = []          # [(column_name, BooleanVar)] aligned to df.columns
        self.checkbuttons = []      # [(checkbutton_widget, column_name)] for searching

        # ----- tk variables ------------------------------------------------
        self.file_var = tk.StringVar()
        self.sheet_var = tk.StringVar()
        self.search_var = tk.StringVar()
        self.count_var = tk.StringVar(value="No file loaded.")
        self.status_var = tk.StringVar(value="Ready.")

        self.search_var.trace_add("write", self._apply_filter)

        self._build_ui()

    # ======================================================================
    #  UI construction
    # ======================================================================
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # ----- 1. file picker row -----------------------------------------
        top = ttk.LabelFrame(self, text="1. Choose a file")
        top.pack(fill="x", **pad)

        entry = ttk.Entry(top, textvariable=self.file_var)
        entry.pack(side="left", fill="x", expand=True, padx=8, pady=8)
        ttk.Button(top, text="Browse...", command=self.browse_file).pack(
            side="left", padx=(0, 4), pady=8)
        ttk.Button(top, text="Load", command=self.load_file).pack(
            side="left", padx=(0, 8), pady=8)

        # ----- Excel sheet selector (hidden until needed) ------------------
        self.sheet_frame = ttk.Frame(self)
        ttk.Label(self.sheet_frame, text="Sheet:").pack(side="left", padx=(8, 4))
        self.sheet_combo = ttk.Combobox(
            self.sheet_frame, textvariable=self.sheet_var, state="readonly", width=40)
        self.sheet_combo.pack(side="left", padx=(0, 8), pady=4)
        self.sheet_combo.bind("<<ComboboxSelected>>", self._on_sheet_change)
        # self.sheet_frame is packed/forgotten dynamically in load_file().

        # ----- 2. columns area --------------------------------------------
        cols_frame = ttk.LabelFrame(self, text="2. Select columns to keep")
        cols_frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        toolbar = ttk.Frame(cols_frame)
        toolbar.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(toolbar, text="Search:").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.search_var, width=28).pack(
            side="left", padx=(4, 12))
        ttk.Button(toolbar, text="Select all", command=lambda: self._set_all(True)).pack(
            side="left", padx=2)
        ttk.Button(toolbar, text="Clear all", command=lambda: self._set_all(False)).pack(
            side="left", padx=2)
        ttk.Button(toolbar, text="Invert", command=self._invert).pack(side="left", padx=2)
        ttk.Label(toolbar, textvariable=self.count_var).pack(side="right")

        # scrollable list of checkbuttons (Canvas + inner Frame + Scrollbar)
        list_holder = ttk.Frame(cols_frame)
        list_holder.pack(fill="both", expand=True, padx=6, pady=(2, 6))

        self.canvas = tk.Canvas(list_holder, borderwidth=0, highlightthickness=0)
        vbar = ttk.Scrollbar(list_holder, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.cols_inner = ttk.Frame(self.canvas)
        self._inner_id = self.canvas.create_window(
            (0, 0), window=self.cols_inner, anchor="nw")
        self.cols_inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # Mouse-wheel scrolling only while the pointer is over the list.
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        # ----- 3. preview --------------------------------------------------
        prev_frame = ttk.LabelFrame(
            self, text="Preview (first %d rows)" % PREVIEW_ROWS)
        prev_frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        prev_bar = ttk.Frame(prev_frame)
        prev_bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(prev_bar, text="Preview all columns",
                   command=lambda: self._populate_preview(selected_only=False)).pack(
            side="left", padx=2)
        ttk.Button(prev_bar, text="Preview selected columns",
                   command=lambda: self._populate_preview(selected_only=True)).pack(
            side="left", padx=2)

        tree_holder = ttk.Frame(prev_frame)
        tree_holder.pack(fill="both", expand=True, padx=6, pady=6)
        self.tree = ttk.Treeview(tree_holder, show="headings", height=6)
        ysb = ttk.Scrollbar(tree_holder, orient="vertical", command=self.tree.yview)
        xsb = ttk.Scrollbar(tree_holder, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        tree_holder.rowconfigure(0, weight=1)
        tree_holder.columnconfigure(0, weight=1)

        # ----- 4. export + status -----------------------------------------
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bottom, text="Export selected columns...",
                   command=self.export).pack(side="left")
        ttk.Label(bottom, textvariable=self.status_var, relief="sunken",
                  anchor="w").pack(side="left", fill="x", expand=True, padx=(8, 0))

    # ======================================================================
    #  Scrolling helpers
    # ======================================================================
    def _on_canvas_configure(self, event):
        # Make the inner frame match the canvas width so checkbuttons fill it.
        self.canvas.itemconfigure(self._inner_id, width=event.width)

    def _bind_mousewheel(self, _event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)      # Win / macOS
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)        # Linux up
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)        # Linux down

    def _unbind_mousewheel(self, _event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

    # ======================================================================
    #  Loading files
    # ======================================================================
    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select a CSV or Excel file",
            filetypes=[
                ("Spreadsheet files", "*.csv *.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.file_var.set(path)
            self.load_file()

    def load_file(self):
        if pd is None:
            messagebox.showerror(
                "Missing dependency",
                "pandas is not installed.\n\nInstall it with:\n    pip install pandas openpyxl")
            return

        path = self.file_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please choose a file first.")
            return
        if not os.path.isfile(path):
            messagebox.showerror("Not found", "File does not exist:\n%s" % path)
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".xlsx", ".xls", ".xlsm", ".xlsb"):
                self.excel_file = pd.ExcelFile(path)
                sheets = list(self.excel_file.sheet_names)
                self.sheet_combo.configure(values=sheets)
                self.sheet_var.set(sheets[0])
                # Show the sheet selector under the file row.
                self.sheet_frame.pack(fill="x", padx=8, pady=(0, 6),
                                      after=self.children_top_ref())
                self._load_sheet(sheets[0])
            else:
                # Treat everything else as delimited text (CSV/TSV/etc.).
                self.excel_file = None
                self.sheet_frame.pack_forget()
                self.df = self._read_csv(path)
                self._populate_columns()
        except Exception as exc:  # noqa: BLE001 - surface any read error to the user
            self.df = None
            self._populate_columns()
            messagebox.showerror("Could not read file", "%s\n\n%s" % (path, exc))

    def children_top_ref(self):
        # The file-picker LabelFrame is the first packed child; place the sheet
        # selector right after it. Falls back gracefully if not found.
        for child in self.pack_slaves():
            if isinstance(child, ttk.LabelFrame):
                return child
        return None

    def _read_csv(self, path):
        """Read a delimited text file, trying a few encodings and delimiters."""
        attempts = []
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                with open(path, "r", encoding=enc, newline="") as fh:
                    header = fh.readline()
            except UnicodeDecodeError as exc:
                attempts.append("read header encoding=%s -> %s" % (enc, exc))
                continue

            # Only let pandas sniff the delimiter if the header actually
            # contains a common delimiter. Otherwise csv.Sniffer can pick a
            # stray letter and shred a legitimate single-column file.
            kwargs_list = []
            if any(d in header for d in (",", ";", "\t", "|")):
                kwargs_list.append({"sep": None, "engine": "python"})
            kwargs_list.append({})  # plain comma read (also the 1-column case)

            for kwargs in kwargs_list:
                try:
                    return pd.read_csv(path, encoding=enc, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    attempts.append("encoding=%s %s -> %s" % (enc, kwargs, exc))
        raise RuntimeError(
            "Unable to parse the file as CSV. Attempts:\n  " + "\n  ".join(attempts))

    def _on_sheet_change(self, _event=None):
        if self.excel_file is not None:
            self._load_sheet(self.sheet_var.get())

    def _load_sheet(self, sheet_name):
        try:
            self.df = self.excel_file.parse(sheet_name)
            self._populate_columns()
        except Exception as exc:  # noqa: BLE001
            self.df = None
            self._populate_columns()
            messagebox.showerror("Could not read sheet", str(exc))

    # ======================================================================
    #  Column checkboxes
    # ======================================================================
    def _populate_columns(self):
        # Clear any existing checkbuttons.
        for child in self.cols_inner.winfo_children():
            child.destroy()
        self.col_vars = []
        self.checkbuttons = []

        if self.df is None:
            self.count_var.set("No file loaded.")
            self.status_var.set("Ready.")
            self._clear_preview()
            return

        for col in self.df.columns:
            name = str(col)
            var = tk.BooleanVar(value=True)  # keep everything by default
            cb = ttk.Checkbutton(self.cols_inner, text=name, variable=var,
                                 command=self._update_count)
            cb.pack(fill="x", anchor="w", padx=4, pady=1)
            self.col_vars.append((name, var))
            self.checkbuttons.append((cb, name))

        self.search_var.set("")           # reset any previous filter
        self.canvas.yview_moveto(0)
        self._update_count()
        self._populate_preview(selected_only=False)
        rows = len(self.df.index)
        self.status_var.set(
            "Loaded %d row(s) and %d column(s)." % (rows, len(self.df.columns)))

    def _set_all(self, value):
        # Affect only the columns currently visible under the search filter.
        visible = self._visible_indices()
        for i, (_, var) in enumerate(self.col_vars):
            if i in visible:
                var.set(value)
        self._update_count()

    def _invert(self):
        visible = self._visible_indices()
        for i, (_, var) in enumerate(self.col_vars):
            if i in visible:
                var.set(not var.get())
        self._update_count()

    def _visible_indices(self):
        # Work with positions (always unique) rather than names, which can
        # repeat. col_vars holds (name, var) tuples.
        q = self.search_var.get().strip().lower()
        if not q:
            return set(range(len(self.col_vars)))
        return {i for i, (name, _) in enumerate(self.col_vars)
                if q in name.lower()}

    def _apply_filter(self, *_):
        q = self.search_var.get().strip().lower()
        # Re-pack in original order so the list stays stable.
        for cb, _ in self.checkbuttons:
            cb.pack_forget()
        for cb, name in self.checkbuttons:
            if not q or q in name.lower():
                cb.pack(fill="x", anchor="w", padx=4, pady=1)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _selected_positions(self):
        return [i for i, (_, var) in enumerate(self.col_vars) if var.get()]

    def _update_count(self):
        total = len(self.col_vars)
        chosen = len(self._selected_positions())
        self.count_var.set("%d of %d columns selected" % (chosen, total))

    # ======================================================================
    #  Preview table
    # ======================================================================
    def _clear_preview(self):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = ()

    def _populate_preview(self, selected_only):
        self._clear_preview()
        if self.df is None:
            return

        if selected_only:
            positions = self._selected_positions()
            if not positions:
                self.status_var.set("Nothing to preview — no columns selected.")
                return
            view = self.df.iloc[:PREVIEW_ROWS, positions]
        else:
            view = self.df.iloc[:PREVIEW_ROWS, :]

        columns = [str(c) for c in view.columns]
        self.tree["columns"] = columns
        for c in columns:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=140, minwidth=60, stretch=False, anchor="w")

        for _, row in view.iterrows():
            values = []
            for v in row.tolist():
                if pd.isna(v):
                    text = ""
                elif isinstance(v, float) and v.is_integer():
                    # A whole number that pandas upcast to float (because the
                    # column had blanks) should still read as an integer.
                    text = "%d" % v
                else:
                    text = str(v)
                if len(text) > MAX_CELL_CHARS:
                    text = text[:MAX_CELL_CHARS - 1] + "…"
                values.append(text)
            self.tree.insert("", "end", values=values)

    # ======================================================================
    #  Export
    # ======================================================================
    def export(self):
        if self.df is None:
            messagebox.showwarning("Nothing loaded", "Load a file first.")
            return

        positions = self._selected_positions()
        if not positions:
            messagebox.showwarning(
                "No columns selected", "Tick at least one column to keep.")
            return

        source = self.file_var.get().strip()
        base = os.path.splitext(os.path.basename(source))[0] or "export"
        suggested = base + "_filtered.csv"

        out_path = filedialog.asksaveasfilename(
            title="Export selected columns",
            defaultextension=".csv",
            initialfile=suggested,
            filetypes=[("CSV file", "*.csv"), ("Excel file", "*.xlsx")],
        )
        if not out_path:
            return

        subset = self.df.iloc[:, positions]
        lower = out_path.lower()
        try:
            if lower.endswith(".xls"):
                # openpyxl writes .xlsx only; legacy .xls needs the unmaintained
                # xlwt package, so transparently upgrade the target to .xlsx.
                out_path = out_path[:-4] + ".xlsx"
                subset.to_excel(out_path, index=False)
            elif lower.endswith(".xlsx"):
                subset.to_excel(out_path, index=False)
            else:
                # utf-8-sig keeps Excel happy with non-ASCII characters.
                subset.to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc))
            return

        self.status_var.set(
            "Exported %d row(s) x %d column(s) to %s"
            % (len(subset.index), len(subset.columns), out_path))
        messagebox.showinfo(
            "Export complete",
            "Saved %d column(s) and %d row(s) to:\n%s"
            % (len(subset.columns), len(subset.index), out_path))


def main():
    if pd is None:
        # Try to show a GUI error; fall back to console.
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Missing dependency",
                "pandas is required.\n\nInstall it with:\n    pip install pandas openpyxl")
            root.destroy()
        except Exception:
            print("pandas is required. Install with: pip install pandas openpyxl")
        return

    app = ColumnPickerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
