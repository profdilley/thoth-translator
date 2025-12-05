"""
Tkinter GUI for THOTH translation tool.

This module provides a professional graphical interface for translating
CSV/Excel files with column selection, preview, and progress tracking.
"""

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from .config import Config
from .detector import LanguageDetector
from .engine_base import TranslationEngine, TranslationEngineFactory
from .languages import LanguageMapper
from .processor import CSVProcessor, ColumnInfo
from .progress import ProgressState, ProgressTracker

logger = logging.getLogger(__name__)


class THOTHApp:
    """
    Main THOTH GUI application.

    Provides a two-panel interface with:
    - Column list with checkboxes and language dropdowns
    - Preview panel showing live translation samples
    - Progress bar with cancellation support
    - Engine selection and configuration
    """

    # Window dimensions
    WINDOW_WIDTH = 900
    WINDOW_HEIGHT = 700
    MIN_WIDTH = 800
    MIN_HEIGHT = 600

    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize the GUI application.

        Args:
            config: Configuration settings
        """
        self._config = config or Config.load()
        self._root: Optional[tk.Tk] = None
        self._processor: Optional[CSVProcessor] = None
        self._engine: Optional[TranslationEngine] = None
        self._detector: Optional[LanguageDetector] = None
        self._language_mapper = LanguageMapper()

        # Thread management
        self._worker_thread: Optional[threading.Thread] = None
        self._progress_tracker: Optional[ProgressTracker] = None
        self._update_queue: queue.Queue = queue.Queue()

        # UI state
        self._column_vars: dict[str, tk.BooleanVar] = {}
        self._column_lang_vars: dict[str, tk.StringVar] = {}
        self._file_loaded = False
        self._is_translating = False

    def run(self) -> None:
        """Start the GUI application."""
        self._create_window()
        self._create_widgets()
        self._setup_bindings()
        self._root.mainloop()

    def _create_window(self) -> None:
        """Create the main window."""
        self._root = tk.Tk()
        self._root.title("THOTH - Offline Translation Tool")
        self._root.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self._root.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)

        # Configure style
        self._style = ttk.Style()
        self._style.theme_use("clam")

        # Configure colors
        self._style.configure("TFrame", background="#f0f0f0")
        self._style.configure("TLabel", background="#f0f0f0")
        self._style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))
        self._style.configure("Title.TLabel", font=("Helvetica", 16, "bold"))

        # Center window
        self._root.update_idletasks()
        x = (self._root.winfo_screenwidth() - self.WINDOW_WIDTH) // 2
        y = (self._root.winfo_screenheight() - self.WINDOW_HEIGHT) // 2
        self._root.geometry(f"+{x}+{y}")

    def _create_widgets(self) -> None:
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self._root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            title_frame,
            text="THOTH - Offline Translation Tool",
            style="Title.TLabel",
        ).pack(side=tk.LEFT)

        # File selection frame
        self._create_file_frame(main_frame)

        # Engine selection frame
        self._create_engine_frame(main_frame)

        # Main content: columns and preview
        self._create_content_frame(main_frame)

        # Progress frame
        self._create_progress_frame(main_frame)

        # Action buttons
        self._create_action_frame(main_frame)

    def _create_file_frame(self, parent: ttk.Frame) -> None:
        """Create file selection widgets."""
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        # Input file
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill=tk.X, pady=2)

        ttk.Label(input_frame, text="Input:", width=8).pack(side=tk.LEFT)
        self._input_var = tk.StringVar()
        self._input_entry = ttk.Entry(input_frame, textvariable=self._input_var)
        self._input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(
            input_frame,
            text="Browse...",
            command=self._browse_input,
        ).pack(side=tk.RIGHT)

        # Output file
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=2)

        ttk.Label(output_frame, text="Output:", width=8).pack(side=tk.LEFT)
        self._output_var = tk.StringVar()
        self._output_entry = ttk.Entry(output_frame, textvariable=self._output_var)
        self._output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(
            output_frame,
            text="Browse...",
            command=self._browse_output,
        ).pack(side=tk.RIGHT)

    def _create_engine_frame(self, parent: ttk.Frame) -> None:
        """Create engine selection widgets."""
        engine_frame = ttk.Frame(parent)
        engine_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(engine_frame, text="Engine:").pack(side=tk.LEFT)

        self._engine_var = tk.StringVar(value=self._config.default_engine)
        engine_combo = ttk.Combobox(
            engine_frame,
            textvariable=self._engine_var,
            values=["nllb", "argos"],
            state="readonly",
            width=25,
        )
        engine_combo.pack(side=tk.LEFT, padx=10)

        # Engine description
        self._engine_desc = ttk.Label(
            engine_frame,
            text="NLLB-200: 200 languages, recommended",
            foreground="gray",
        )
        self._engine_desc.pack(side=tk.LEFT, padx=10)

        engine_combo.bind("<<ComboboxSelected>>", self._on_engine_change)

    def _create_content_frame(self, parent: ttk.Frame) -> None:
        """Create main content area with columns and preview."""
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Configure grid
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)

        # Column panel
        self._create_column_panel(content_frame)

        # Preview panel
        self._create_preview_panel(content_frame)

    def _create_column_panel(self, parent: ttk.Frame) -> None:
        """Create column selection panel."""
        column_frame = ttk.LabelFrame(parent, text="Columns", padding=10)
        column_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Filter entry
        filter_frame = ttk.Frame(column_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
        self._filter_var = tk.StringVar()
        self._filter_entry = ttk.Entry(filter_frame, textvariable=self._filter_var)
        self._filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self._filter_var.trace_add("write", self._on_filter_change)

        # Column list with scrollbar
        list_frame = ttk.Frame(column_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self._column_canvas = tk.Canvas(list_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self._column_canvas.yview,
        )
        self._column_list_frame = ttk.Frame(self._column_canvas)

        self._column_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._column_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._column_canvas_window = self._column_canvas.create_window(
            (0, 0),
            window=self._column_list_frame,
            anchor="nw",
        )

        self._column_list_frame.bind("<Configure>", self._on_column_frame_configure)
        self._column_canvas.bind("<Configure>", self._on_canvas_configure)

        # Select/Deselect buttons
        button_frame = ttk.Frame(column_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(
            button_frame,
            text="Select All",
            command=self._select_all_columns,
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            button_frame,
            text="Deselect All",
            command=self._deselect_all_columns,
        ).pack(side=tk.LEFT)

    def _create_preview_panel(self, parent: ttk.Frame) -> None:
        """Create translation preview panel."""
        preview_frame = ttk.LabelFrame(parent, text="Preview", padding=10)
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Preview tree
        columns = ("original", "translation")
        self._preview_tree = ttk.Treeview(
            preview_frame,
            columns=columns,
            show="headings",
            height=10,
        )

        self._preview_tree.heading("original", text="Original")
        self._preview_tree.heading("translation", text="Translation")

        self._preview_tree.column("original", width=180)
        self._preview_tree.column("translation", width=180)

        preview_scroll = ttk.Scrollbar(
            preview_frame,
            orient="vertical",
            command=self._preview_tree.yview,
        )
        self._preview_tree.configure(yscrollcommand=preview_scroll.set)

        self._preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Refresh button
        ttk.Button(
            preview_frame,
            text="Refresh Preview",
            command=self._refresh_preview,
        ).pack(pady=(5, 0))

        # Selected column for preview
        self._preview_column: Optional[str] = None

    def _create_progress_frame(self, parent: ttk.Frame) -> None:
        """Create progress bar and status display."""
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        # Progress bar
        self._progress_var = tk.DoubleVar()
        self._progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self._progress_var,
            maximum=100,
        )
        self._progress_bar.pack(fill=tk.X, pady=(0, 5))

        # Status labels
        status_frame = ttk.Frame(progress_frame)
        status_frame.pack(fill=tk.X)

        self._status_label = ttk.Label(status_frame, text="Ready")
        self._status_label.pack(side=tk.LEFT)

        self._eta_label = ttk.Label(status_frame, text="")
        self._eta_label.pack(side=tk.RIGHT)

        self._progress_label = ttk.Label(status_frame, text="")
        self._progress_label.pack(side=tk.RIGHT, padx=20)

    def _create_action_frame(self, parent: ttk.Frame) -> None:
        """Create action buttons."""
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X)

        # Translate button
        self._translate_btn = ttk.Button(
            action_frame,
            text="Translate",
            command=self._start_translation,
            state="disabled",
        )
        self._translate_btn.pack(side=tk.LEFT, padx=5)

        # Cancel button
        self._cancel_btn = ttk.Button(
            action_frame,
            text="Cancel",
            command=self._cancel_translation,
            state="disabled",
        )
        self._cancel_btn.pack(side=tk.LEFT, padx=5)

        # Status
        self._action_status = ttk.Label(action_frame, text="")
        self._action_status.pack(side=tk.RIGHT, padx=5)

    def _setup_bindings(self) -> None:
        """Setup event bindings."""
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.bind("<Configure>", self._on_window_resize)

        # Periodic update check
        self._root.after(100, self._process_updates)

    def _browse_input(self) -> None:
        """Open file browser for input file."""
        filetypes = [
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)

        if path:
            self._input_var.set(path)
            # Auto-generate output path
            input_path = Path(path)
            output_path = input_path.parent / f"{input_path.stem}_translated{input_path.suffix}"
            self._output_var.set(str(output_path))
            # Load and analyze file
            self._load_file(path)

    def _browse_output(self) -> None:
        """Open file browser for output file."""
        filetypes = [
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx"),
        ]
        path = filedialog.asksaveasfilename(
            filetypes=filetypes,
            defaultextension=".csv",
        )
        if path:
            self._output_var.set(path)

    def _load_file(self, path: str) -> None:
        """Load and analyze a file."""
        self._status_label.config(text="Loading file...")
        self._root.update()

        try:
            # Create processor
            self._processor = CSVProcessor(self._config)
            self._processor.load_file(path)

            # Load detector if needed
            if self._detector is None:
                self._status_label.config(text="Loading language detector...")
                self._root.update()
                self._detector = LanguageDetector(
                    model_path=str(self._config.get_lid_path()),
                )
                self._detector.load_model()

            # Analyze columns
            self._status_label.config(text="Analyzing columns...")
            self._root.update()

            self._processor._detector = self._detector
            columns = self._processor.analyze_columns()

            # Update column list
            self._update_column_list(columns)

            self._file_loaded = True
            self._translate_btn.config(state="normal")
            self._status_label.config(
                text=f"Loaded: {self._processor.row_count:,} rows, "
                     f"{self._processor.column_count} columns"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            self._status_label.config(text="Error loading file")
            logger.error(f"Load error: {e}")

    def _update_column_list(self, columns: list[ColumnInfo]) -> None:
        """Update the column list display."""
        # Clear existing
        for widget in self._column_list_frame.winfo_children():
            widget.destroy()

        self._column_vars.clear()
        self._column_lang_vars.clear()

        # Get language options
        lang_options = ["Auto"] + [
            lang.name for lang in self._language_mapper.get_all_languages()
        ]

        # Create column rows
        for col in columns:
            row = ttk.Frame(self._column_list_frame)
            row.pack(fill=tk.X, pady=2)

            # Checkbox
            var = tk.BooleanVar(value=col.selected)
            self._column_vars[col.name] = var

            cb = ttk.Checkbutton(row, variable=var)
            cb.pack(side=tk.LEFT)

            # Column name
            name_label = ttk.Label(row, text=col.name, width=20, anchor="w")
            name_label.pack(side=tk.LEFT, padx=5)

            # Type indicator
            if col.column_type in ("numeric", "date", "empty"):
                type_label = ttk.Label(
                    row,
                    text=f"({col.column_type})",
                    foreground="gray",
                )
                type_label.pack(side=tk.LEFT)
            else:
                # Language dropdown
                lang_var = tk.StringVar(value="Auto")
                if col.language_name:
                    lang_var.set(col.language_name)
                self._column_lang_vars[col.name] = lang_var

                lang_combo = ttk.Combobox(
                    row,
                    textvariable=lang_var,
                    values=lang_options,
                    state="readonly",
                    width=15,
                )
                lang_combo.pack(side=tk.LEFT)

            # Make row clickable for preview
            # Make row clickable for preview
            name_label.bind("<Button-1>", lambda e, c=col.name: self._select_for_preview(c))
            name_label.bind("<Enter>", lambda e: e.widget.config(cursor="hand2"))
            name_label.bind("<Leave>", lambda e: e.widget.config(cursor=""))

        # Auto-select first translatable column for preview
        translatable = [c for c in columns if c.column_type not in ("numeric", "date", "empty")]
        if translatable:
            self._preview_column = translatable[0].name
            self._root.after(500, self._refresh_preview)  # Delay to let engine load

    def _select_for_preview(self, column_name: str) -> None:
        """Select a column for preview."""
        self._preview_column = column_name
        
        # Update visual highlighting
        for widget in self._column_list_frame.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, ttk.Label) and child.cget("width") == 20:
                    if child.cget("text") == column_name:
                        child.config(foreground="blue", font=("TkDefaultFont", 10, "bold"))
                    else:
                        child.config(foreground="black", font=("TkDefaultFont", 10, "normal"))
        
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        """Refresh the preview panel."""
        if not self._processor or not self._preview_column:
            return

        # Clear preview
        for item in self._preview_tree.get_children():
            self._preview_tree.delete(item)

        self._status_label.config(text="Loading preview...")
        self._root.update()

        try:
            # Ensure engine is loaded
            if self._engine is None:
                engine_name = self._engine_var.get()
                self._engine = TranslationEngineFactory.create(engine_name)
                self._engine.load_model()

            # Get preview
            preview = self._processor.get_preview(
                self._preview_column,
                self._engine,
                num_rows=5,
            )

            # Display
            for orig, trans in preview:
                # Truncate for display
                orig_display = orig[:50] + "..." if len(orig) > 50 else orig
                trans_display = trans[:50] + "..." if len(trans) > 50 else trans
                self._preview_tree.insert("", "end", values=(orig_display, trans_display))

            self._status_label.config(text="Preview loaded")

        except Exception as e:
            logger.error(f"Preview error: {e}")
            self._status_label.config(text="Preview failed")

    def _select_all_columns(self) -> None:
        """Select all translatable columns."""
        for var in self._column_vars.values():
            var.set(True)

    def _deselect_all_columns(self) -> None:
        """Deselect all columns."""
        for var in self._column_vars.values():
            var.set(False)

    def _on_filter_change(self, *args) -> None:
        """Handle column filter change."""
        filter_text = self._filter_var.get().lower()

        for widget in self._column_list_frame.winfo_children():
            # Get column name from the widget
            col_name = None
            for child in widget.winfo_children():
                if isinstance(child, ttk.Label):
                    col_name = child.cget("text")
                    break

            if col_name:
                if filter_text in col_name.lower():
                    widget.pack(fill=tk.X, pady=2)
                else:
                    widget.pack_forget()

    def _on_engine_change(self, event=None) -> None:
        """Handle engine selection change."""
        engine = self._engine_var.get()
        if engine == "nllb":
            self._engine_desc.config(text="NLLB-200: 200 languages, recommended")
        else:
            self._engine_desc.config(text="Argos: Good for Western European languages")

        # Unload current engine if different
        if self._engine and self._engine.get_engine_id() != engine:
            self._engine.unload_model()
            self._engine = None

    def _start_translation(self) -> None:
        """Start the translation process in a background thread."""
        if not self._processor or self._is_translating:
            return

        # Get selected columns
        selected = [
            name for name, var in self._column_vars.items()
            if var.get()
        ]

        if not selected:
            messagebox.showwarning("Warning", "No columns selected for translation")
            return

        # Update processor selection
        self._processor.set_column_selection(selected)

        # Update language overrides
        for col_name, lang_var in self._column_lang_vars.items():
            lang_name = lang_var.get()
            if lang_name != "Auto":
                lang_info = self._language_mapper.get_language(lang_name)
                if lang_info:
                    self._processor.set_column_language(col_name, lang_info.nllb_code)

        # Update UI
        self._is_translating = True
        self._translate_btn.config(state="disabled")
        self._cancel_btn.config(state="normal")
        self._progress_var.set(0)

        # Create progress tracker
        self._progress_tracker = ProgressTracker()
        self._progress_tracker.on_progress = self._on_progress_update

        # Start worker thread
        self._worker_thread = threading.Thread(target=self._translation_worker)
        self._worker_thread.daemon = True
        self._worker_thread.start()

    def _translation_worker(self) -> None:
        """Background worker for translation."""
        try:
            # Load engine if needed
            if self._engine is None:
                self._queue_update("status", "Loading translation engine...")
                engine_name = self._engine_var.get()
                self._engine = TranslationEngineFactory.create(engine_name)
                self._engine.load_model()

            # Translate
            self._queue_update("status", "Translating...")
            result = self._processor.translate(
                self._engine,
                self._progress_tracker,
            )

            if result.success and not self._progress_tracker.is_cancelled():
                # Save
                self._queue_update("status", "Saving...")
                output_path = self._output_var.get() or None
                saved_path = self._processor.save(output_path)

                self._queue_update(
                    "complete",
                    f"Saved to: {saved_path}\n\n"
                    f"Rows: {result.rows_processed:,}\n"
                    f"Columns: {result.columns_translated}\n"
                    f"Cells: {result.cells_translated:,}\n"
                    f"Time: {result.processing_time:.1f}s"
                )
            elif self._progress_tracker.is_cancelled():
                self._queue_update("cancelled", "Translation cancelled")
            else:
                self._queue_update("error", result.error or "Unknown error")

        except Exception as e:
            logger.error(f"Translation worker error: {e}")
            self._queue_update("error", str(e))

    def _queue_update(self, update_type: str, message: str) -> None:
        """Queue a UI update from worker thread."""
        self._update_queue.put((update_type, message))

    def _on_progress_update(self, state: ProgressState) -> None:
        """Handle progress update from tracker."""
        self._update_queue.put(("progress", state))

    def _process_updates(self) -> None:
        """Process queued updates from worker thread."""
        try:
            while True:
                update = self._update_queue.get_nowait()
                update_type, data = update

                if update_type == "progress":
                    state: ProgressState = data
                    self._progress_var.set(state.percentage)
                    self._status_label.config(text=state.message)
                    self._progress_label.config(
                        text=f"{state.current:,}/{state.total:,}"
                    )
                    self._eta_label.config(text=f"ETA: {state.eta_formatted}")

                elif update_type == "status":
                    self._status_label.config(text=data)

                elif update_type == "complete":
                    self._is_translating = False
                    self._translate_btn.config(state="normal")
                    self._cancel_btn.config(state="disabled")
                    self._progress_var.set(100)
                    self._status_label.config(text="Complete")
                    messagebox.showinfo("Translation Complete", data)

                elif update_type == "error":
                    self._is_translating = False
                    self._translate_btn.config(state="normal")
                    self._cancel_btn.config(state="disabled")
                    self._status_label.config(text="Error")
                    messagebox.showerror("Error", data)

                elif update_type == "cancelled":
                    self._is_translating = False
                    self._translate_btn.config(state="normal")
                    self._cancel_btn.config(state="disabled")
                    self._status_label.config(text="Cancelled")

        except queue.Empty:
            pass

        # Schedule next check
        if self._root:
            self._root.after(100, self._process_updates)

    def _cancel_translation(self) -> None:
        """Cancel the translation process."""
        if self._progress_tracker:
            self._progress_tracker.cancel()
            self._status_label.config(text="Cancelling...")

    def _on_column_frame_configure(self, event) -> None:
        """Update canvas scroll region."""
        self._column_canvas.configure(
            scrollregion=self._column_canvas.bbox("all")
        )

    def _on_canvas_configure(self, event) -> None:
        """Adjust column list width."""
        self._column_canvas.itemconfig(
            self._column_canvas_window,
            width=event.width,
        )

    def _on_window_resize(self, event) -> None:
        """Handle window resize."""
        pass

    def _on_close(self) -> None:
        """Handle window close."""
        if self._is_translating:
            if messagebox.askyesno(
                "Confirm Exit",
                "Translation in progress. Cancel and exit?"
            ):
                self._cancel_translation()
            else:
                return

        # Cleanup
        if self._engine:
            self._engine.unload_model()
        if self._detector:
            self._detector.unload_model()

        self._root.destroy()


def run_gui(config: Optional[Config] = None) -> None:
    """
    Launch the THOTH GUI.

    Args:
        config: Configuration settings
    """
    app = THOTHApp(config)
    app.run()


if __name__ == "__main__":
    run_gui()
