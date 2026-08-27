#!/usr/bin/env python3
"""Read a source CSV (semicolon-delimited, every field wrapped in quotes) and
write a clean version with proper separate columns and no " symbols.

Desktop UI (browse for the file, look through it, then save with a dialog):
    python csv_parser.py

Original one-shot command line (unchanged):
    python csv_parser.py INPUT.csv OUTPUT.csv
"""

import csv
import os
import sys

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:  # headless box without tk: the command line still works.
    tk = ttk = filedialog = messagebox = None

PREVIEW_ROWS = 500          # how many rows the preview table holds at once
MAX_CELL_CHARS = 120        # truncate long cell values in the preview
CHAR_PX = 7                 # rough pixels per character, for column sizing

# Label shown in the combobox -> actual delimiter character.
DELIMITERS = {
    "Semicolon   ;": ";",
    "Comma   ,": ",",
    "Tab   \\t": "\t",
    "Pipe   |": "|",
}
AUTO_DETECT = "Auto-detect"
ENCODINGS = ["utf-8-sig", "utf-8", "cp1251", "latin-1"]


# ==========================================================================
#  Parsing / writing (shared by the UI and the command line)
# ==========================================================================
def parse_rows(path, delimiter=";", quotechar='"', encoding="utf-8-sig"):
    """Read a delimited file into a list of rows. csv handles the unquoting;
    the extra strip('"') catches quotes left inside a field."""
    with open(path, newline="", encoding=encoding) as f_in:
        reader = csv.reader(f_in, delimiter=delimiter, quotechar=quotechar)
        return [[cell.strip().strip('"') for cell in row] for row in reader]


def sniff_delimiter(path, encoding="utf-8-sig", default=";"):
    """Guess the separator from the first chunk of the file."""
    try:
        with open(path, newline="", encoding=encoding) as fh:
            sample = fh.read(64 * 1024)
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except (OSError, UnicodeDecodeError, csv.Error):
        return default


def flatten_newlines(rows):
    """QUOTE_NONE cannot protect a line break inside a field, so a field
    containing one would split into two records. Replace those with spaces and
    report how many fields were touched."""
    fixed = 0
    out = []
    for row in rows:
        new_row = []
        for cell in row:
            if "\n" in cell or "\r" in cell:
                cell = " ".join(cell.split())
                fixed += 1
            new_row.append(cell)
        out.append(new_row)
    return out, fixed


def write_rows(rows, path, delimiter=",", strip_quotes=True, encoding="utf-8"):
    """Write rows out. strip_quotes=True never adds a " back (QUOTE_NONE);
    False falls back to normal quoting when a field needs it.

    Returns a report of the compromises QUOTE_NONE forced, so the caller can
    tell the user about them."""
    report = {"newlines": 0, "escaped": 0}
    if strip_quotes:
        rows, report["newlines"] = flatten_newlines(rows)
        # Without quotes a field holding the separator can only be escaped
        # (\,) and most readers will still split it into two columns.
        report["escaped"] = sum(
            1 for row in rows for cell in row if delimiter in cell)

    with open(path, "w", newline="", encoding=encoding) as f_out:
        if strip_quotes:
            writer = csv.writer(f_out, delimiter=delimiter,
                                quoting=csv.QUOTE_NONE, escapechar="\\")
        else:
            writer = csv.writer(f_out, delimiter=delimiter,
                                quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)
    return report


# ==========================================================================
#  Desktop UI
# ==========================================================================
# Inherit from tk.Tk only when tkinter imported, so the CLI survives without it.
_BASE = tk.Tk if tk is not None else object


class CsvBrowserApp(_BASE):
    def __init__(self):
        super().__init__()
        self.title("CSV Cleaner — browse a CSV, save a clean copy")
        self.geometry("1040x700")
        self.minsize(820, 560)

        # ----- data state --------------------------------------------------
        self.rows = []              # every parsed row, exactly as it will be saved
        self.visible = []           # (row_index, row) currently in the preview
        self.columns = 0            # widest row, so ragged rows still display

        # ----- tk variables ------------------------------------------------
        self.file_var = tk.StringVar()
        self.in_delim_var = tk.StringVar(value=AUTO_DETECT)
        self.out_delim_var = tk.StringVar(value="Comma   ,")
        self.encoding_var = tk.StringVar(value=ENCODINGS[0])
        self.header_var = tk.BooleanVar(value=True)
        self.strip_var = tk.BooleanVar(value=True)
        self.search_var = tk.StringVar()
        self.info_var = tk.StringVar(value="No file loaded.")
        self.status_var = tk.StringVar(value="Ready — pick a CSV to get started.")

        self.search_var.trace_add("write", lambda *_: self._populate_preview())

        self._apply_theme()
        self._build_ui()
        self._bind_keys()

    # ------------------------------------------------------------------ look
    def _apply_theme(self):
        style = ttk.Style(self)
        for name in ("vista", "clam", "default"):
            if name in style.theme_names():
                style.theme_use(name)
                break
        style.configure("Treeview", rowheight=22)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("Info.TLabel", foreground="#41556b")
        style.configure("Save.TButton", font=("Segoe UI", 9, "bold"))

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # ----- 1. source file ---------------------------------------------
        top = ttk.LabelFrame(self, text="1. Source file")
        top.pack(fill="x", **pad)
        row = ttk.Frame(top)
        row.pack(fill="x", padx=8, pady=8)
        ttk.Entry(row, textvariable=self.file_var).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse...", command=self.browse_file).pack(
            side="left", padx=(6, 0))
        ttk.Button(row, text="Reload", command=self.load_file).pack(
            side="left", padx=(4, 0))

        # ----- 2. how to read / write -------------------------------------
        opts = ttk.LabelFrame(self, text="2. How to read and write it")
        opts.pack(fill="x", padx=10, pady=(0, 6))
        grid = ttk.Frame(opts)
        grid.pack(fill="x", padx=8, pady=8)

        ttk.Label(grid, text="Input separator:").grid(row=0, column=0, sticky="w")
        in_combo = ttk.Combobox(
            grid, textvariable=self.in_delim_var, state="readonly", width=16,
            values=[AUTO_DETECT] + list(DELIMITERS))
        in_combo.grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(grid, text="Encoding:").grid(row=0, column=2, sticky="w")
        enc_combo = ttk.Combobox(
            grid, textvariable=self.encoding_var, state="readonly", width=12,
            values=ENCODINGS)
        enc_combo.grid(row=0, column=3, sticky="w", padx=(6, 18))

        ttk.Checkbutton(grid, text="First row is a header",
                        variable=self.header_var,
                        command=self._rebuild_columns).grid(
            row=0, column=4, sticky="w")

        ttk.Label(grid, text="Output separator:").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        out_combo = ttk.Combobox(
            grid, textvariable=self.out_delim_var, state="readonly", width=16,
            values=list(DELIMITERS))
        out_combo.grid(row=1, column=1, sticky="w", padx=(6, 18), pady=(8, 0))

        ttk.Checkbutton(grid, text='Never write " characters',
                        variable=self.strip_var).grid(
            row=1, column=2, columnspan=2, sticky="w", pady=(8, 0))

        # Re-read the file when a reading option changes.
        for combo in (in_combo, enc_combo):
            combo.bind("<<ComboboxSelected>>", lambda _e: self.load_file())

        # ----- 3. preview / browse ----------------------------------------
        prev = ttk.LabelFrame(self, text="3. Browse the parsed rows")
        prev.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        bar = ttk.Frame(prev)
        bar.pack(fill="x", padx=8, pady=(8, 2))
        ttk.Label(bar, text="Filter:").pack(side="left")
        ttk.Entry(bar, textvariable=self.search_var, width=30).pack(
            side="left", padx=(4, 6))
        ttk.Button(bar, text="Clear",
                   command=lambda: self.search_var.set("")).pack(side="left")
        ttk.Label(bar, textvariable=self.info_var, style="Info.TLabel").pack(
            side="right")

        holder = ttk.Frame(prev)
        holder.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        self.tree = ttk.Treeview(holder, show="headings", selectmode="browse")
        ysb = ttk.Scrollbar(holder, orient="vertical", command=self.tree.yview)
        xsb = ttk.Scrollbar(holder, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)

        self.tree.tag_configure("odd", background="#f4f7fa")
        self.tree.bind("<Double-1>", self._show_row_detail)
        self.tree.bind("<Return>", self._show_row_detail)

        # ----- 4. save + status -------------------------------------------
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(bottom, text="Save clean CSV as...", style="Save.TButton",
                   command=self.save_as).pack(side="left")
        ttk.Label(bottom, textvariable=self.status_var, relief="sunken",
                  anchor="w").pack(side="left", fill="x", expand=True,
                                   padx=(8, 0))

    def _bind_keys(self):
        self.bind("<Control-o>", lambda _e: self.browse_file())
        self.bind("<Control-s>", lambda _e: self.save_as())
        self.bind("<F5>", lambda _e: self.load_file())

    # ------------------------------------------------------------- loading
    def browse_file(self):
        current = self.file_var.get().strip()
        path = filedialog.askopenfilename(
            title="Select a CSV file",
            initialdir=os.path.dirname(current) if current else os.getcwd(),
            filetypes=[("CSV / text files", "*.csv *.txt *.tsv"),
                       ("All files", "*.*")],
        )
        if path:
            self.file_var.set(path)
            self.load_file()

    def load_file(self):
        path = self.file_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Choose a CSV file first.")
            return
        if not os.path.isfile(path):
            messagebox.showerror("Not found", "File does not exist:\n%s" % path)
            return

        encoding = self.encoding_var.get()
        choice = self.in_delim_var.get()
        if choice == AUTO_DETECT:
            delimiter = sniff_delimiter(path, encoding)
        else:
            delimiter = DELIMITERS[choice]

        self.status_var.set("Reading %s ..." % os.path.basename(path))
        self.update_idletasks()
        try:
            self.rows = parse_rows(path, delimiter=delimiter, encoding=encoding)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            self.rows = []
            self._rebuild_columns()
            messagebox.showerror(
                "Could not read file",
                "%s\n\n%s\n\nTry a different encoding or separator."
                % (path, exc))
            self.status_var.set("Read failed.")
            return

        self._rebuild_columns()
        shown = "%s (%s)" % (
            "auto-detected" if choice == AUTO_DETECT else "chosen",
            {";": "semicolon", ",": "comma", "\t": "tab",
             "|": "pipe"}.get(delimiter, repr(delimiter)))
        self.status_var.set("Loaded %d row(s) — separator %s." % (len(self.rows), shown))

    # ------------------------------------------------------------- preview
    def _rebuild_columns(self):
        """Set up the table columns from the current rows, then fill it."""
        self.columns = max((len(r) for r in self.rows), default=0)
        col_ids = ["c%d" % i for i in range(self.columns)]
        self.tree.configure(columns=col_ids)

        header = self.rows[0] if (self.rows and self.header_var.get()) else []
        for i, cid in enumerate(col_ids):
            title = header[i].strip() if i < len(header) and header[i].strip() \
                else "Column %d" % (i + 1)
            self.tree.heading(cid, text=title)
            self.tree.column(cid, width=self._column_width(i, title),
                             minwidth=60, stretch=False, anchor="w")
        self._populate_preview()

    def _column_width(self, index, title):
        """Size a column from its title and the first rows of data."""
        widest = len(title)
        for row in self._data_rows()[:60]:
            if index < len(row):
                widest = max(widest, len(row[index]))
        return max(70, min(320, (widest + 2) * CHAR_PX))

    def _data_rows(self):
        """Rows to display — the header row is a heading, not data."""
        if self.rows and self.header_var.get():
            return self.rows[1:]
        return self.rows

    def _populate_preview(self):
        self.tree.delete(*self.tree.get_children())
        needle = self.search_var.get().strip().lower()
        data = self._data_rows()

        matches = 0
        self.visible = []
        for index, row in enumerate(data):
            if needle and not any(needle in cell.lower() for cell in row):
                continue
            matches += 1
            if len(self.visible) >= PREVIEW_ROWS:
                continue
            values = [
                (cell[:MAX_CELL_CHARS] + "..." if len(cell) > MAX_CELL_CHARS
                 else cell)
                for cell in row
            ]
            values += [""] * (self.columns - len(values))
            self.tree.insert("", "end", iid=str(index), values=values,
                             tags=("odd",) if len(self.visible) % 2 else ())
            self.visible.append((index, row))

        if not data:
            self.info_var.set(
                "No file loaded." if not self.rows else "No data rows.")
            return
        shown = len(self.visible)
        parts = ["%d row(s)" % len(data), "%d column(s)" % self.columns]
        if needle:
            parts.append("%d match(es)" % matches)
        if shown < matches:
            parts.append("showing first %d" % shown)
        self.info_var.set("   ·   ".join(parts))

    def _show_row_detail(self, _event=None):
        """Full, untruncated values for the selected row — long cells get cut
        off in the table, so this is how you read them."""
        selection = self.tree.selection()
        if not selection:
            return
        row_index = int(selection[0])
        row = self._data_rows()[row_index]
        header = self.rows[0] if (self.rows and self.header_var.get()) else []

        win = tk.Toplevel(self)
        win.title("Row %d" % (row_index + 1))
        win.geometry("620x420")
        win.transient(self)

        holder = ttk.Frame(win)
        holder.pack(fill="both", expand=True, padx=10, pady=10)
        text = tk.Text(holder, wrap="word", height=20, borderwidth=1,
                       relief="solid")
        sb = ttk.Scrollbar(holder, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)

        for i, cell in enumerate(row):
            name = header[i] if i < len(header) and header[i].strip() \
                else "Column %d" % (i + 1)
            text.insert("end", "%s\n" % name, "name")
            text.insert("end", "%s\n\n" % (cell if cell else "(empty)"))
        text.tag_configure("name", font=("Segoe UI", 9, "bold"),
                           foreground="#1f4e79")
        text.configure(state="disabled")

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 10))
        win.bind("<Escape>", lambda _e: win.destroy())

    # ---------------------------------------------------------------- save
    def save_as(self):
        if not self.rows:
            messagebox.showwarning("Nothing loaded", "Load a CSV file first.")
            return

        source = self.file_var.get().strip()
        base = os.path.splitext(os.path.basename(source))[0] or "export"
        out_path = filedialog.asksaveasfilename(
            title="Save clean CSV as",
            defaultextension=".csv",
            initialdir=os.path.dirname(source) if source else os.getcwd(),
            initialfile=base + "_clean.csv",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
        )
        if not out_path:
            return
        if os.path.abspath(out_path) == os.path.abspath(source):
            messagebox.showerror(
                "Same file",
                "That is the source file. Pick a different name so the "
                "original is not overwritten.")
            return

        try:
            report = write_rows(
                self.rows, out_path,
                delimiter=DELIMITERS[self.out_delim_var.get()],
                strip_quotes=self.strip_var.get())
        except (OSError, csv.Error) as exc:
            messagebox.showerror("Save failed", str(exc))
            self.status_var.set("Save failed.")
            return

        note = ""
        if report["newlines"]:
            note += ("\n\n%d field(s) contained line breaks; those were "
                     "replaced with spaces to keep one record per line."
                     % report["newlines"])
        if report["escaped"]:
            note += ("\n\n%d field(s) contain the output separator. With "
                     'quoting off they were written as \\%s, which many '
                     "readers will still split — untick "
                     "'Never write \" characters' if that matters."
                     % (report["escaped"],
                        DELIMITERS[self.out_delim_var.get()]))
        self.status_var.set("Wrote %d row(s) to %s" % (len(self.rows), out_path))
        messagebox.showinfo(
            "Saved",
            "Wrote %d row(s) x %d column(s) to:\n%s%s"
            % (len(self.rows), self.columns, out_path, note))


# ==========================================================================
#  Entry points
# ==========================================================================
def run_cli(input_path, output_path):
    rows = parse_rows(input_path)
    report = write_rows(rows, output_path)
    print("Wrote %d rows to %s" % (len(rows), output_path))
    if report["newlines"]:
        print("Note: %d field(s) had line breaks replaced with spaces."
              % report["newlines"])
    if report["escaped"]:
        print("Note: %d field(s) contain the output separator and were escaped."
              % report["escaped"])


def main():
    if len(sys.argv) > 1:
        # Original behaviour: csv_parser.py INPUT [OUTPUT]
        run_cli(sys.argv[1],
                sys.argv[2] if len(sys.argv) > 2 else "100_clean.csv")
        return

    if tk is None:
        print("tkinter is not available, so the UI cannot start.\n"
              "Use the command line instead: python csv_parser.py INPUT.csv OUTPUT.csv")
        return

    CsvBrowserApp().mainloop()


if __name__ == "__main__":
    main()
