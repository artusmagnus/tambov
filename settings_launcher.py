#!/usr/bin/env python3
"""Simple Tk UI launcher for mask_annotator.py and mask_mapper.py settings."""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Wet/Dry Mapper Launcher")
        self.geometry("760x740")

        self.script_var = tk.StringVar(value="mask_mapper.py")
        self.source_var = tk.StringVar()
        self.out_dir_var = tk.StringVar(value="mask_output")
        self.load_dir_var = tk.StringVar()
        self.analysis_source_var = tk.StringVar()
        self.output_video_var = tk.StringVar()

        self.mode_var = tk.StringVar(value="live")
        self.rtsp_transport_var = tk.StringVar(value="auto")

        self.scale_var = tk.DoubleVar(value=1.0)
        self.alpha_var = tk.DoubleVar(value=0.45)
        self.analysis_max_distance_var = tk.DoubleVar(value=0.08)
        self.analysis_time_window_var = tk.DoubleVar(value=5.0)
        self.analysis_sample_interval_var = tk.DoubleVar(value=1.0)
        self.hex_size_var = tk.IntVar(value=40)
        self.brush_size_var = tk.IntVar(value=20)
        self.max_display_width_var = tk.IntVar(value=1280)
        self.live_analysis_interval_var = tk.DoubleVar(value=1.0)
        self.fps_var = tk.StringVar(value="")
        self.stream_reconnect_delay_var = tk.DoubleVar(value=2.0)

        self.show_hex_values_var = tk.BooleanVar(value=False)
        self.average_wetness_only_var = tk.BooleanVar(value=False)
        self.use_opencl_var = tk.BooleanVar(value=False)
        self.use_obstruction_colors_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._refresh_script_mode()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="Script").pack(side="left")
        script_combo = ttk.Combobox(top, textvariable=self.script_var, values=["mask_mapper.py", "mask_annotator.py"], state="readonly", width=20)
        script_combo.pack(side="left", padx=8)
        script_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_script_mode())

        paths = ttk.LabelFrame(root, text="Paths", padding=8)
        paths.pack(fill="x", pady=6)
        self._path_row(paths, "Source", self.source_var, is_file=True)
        self._path_row(paths, "Out dir", self.out_dir_var, is_dir=True)
        self._path_row(paths, "Load dir", self.load_dir_var, is_dir=True)
        self._path_row(paths, "Analysis source", self.analysis_source_var, is_file=True)
        self._path_row(paths, "Output video", self.output_video_var, is_save_file=True)

        self.mode_frame = ttk.LabelFrame(root, text="mask_mapper mode", padding=8)
        self.mode_frame.pack(fill="x", pady=6)
        ttk.Radiobutton(self.mode_frame, text="Live stream", variable=self.mode_var, value="live").pack(side="left")
        ttk.Radiobutton(self.mode_frame, text="Process video", variable=self.mode_var, value="process").pack(side="left", padx=8)

        nums = ttk.LabelFrame(root, text="Numeric options", padding=8)
        nums.pack(fill="x", pady=6)
        self._num_row(nums, "Scale", self.scale_var)
        self._num_row(nums, "Alpha", self.alpha_var)
        self._num_row(nums, "Analysis max distance", self.analysis_max_distance_var)
        self._num_row(nums, "Analysis time window (s)", self.analysis_time_window_var)
        self._num_row(nums, "Analysis sample interval (s)", self.analysis_sample_interval_var)
        self._num_row(nums, "Hex size", self.hex_size_var)
        self._num_row(nums, "Brush size", self.brush_size_var)
        self._num_row(nums, "Max display width", self.max_display_width_var)
        self._num_row(nums, "Live analysis interval (s)", self.live_analysis_interval_var)
        self._text_row(nums, "Target FPS (optional)", self.fps_var)
        self._num_row(nums, "Reconnect delay (s)", self.stream_reconnect_delay_var)

        checks = ttk.LabelFrame(root, text="Flags", padding=8)
        checks.pack(fill="x", pady=6)
        for text, var in [
            ("Show hex values", self.show_hex_values_var),
            ("Average wetness only", self.average_wetness_only_var),
            ("Use OpenCL", self.use_opencl_var),
            ("Use obstruction colors", self.use_obstruction_colors_var),
        ]:
            ttk.Checkbutton(checks, text=text, variable=var).pack(anchor="w")

        transport = ttk.Frame(root)
        transport.pack(fill="x", pady=4)
        ttk.Label(transport, text="RTSP transport").pack(side="left")
        ttk.Combobox(
            transport,
            textvariable=self.rtsp_transport_var,
            values=["auto", "tcp", "udp", "udp_multicast", "http"],
            state="readonly",
            width=16,
        ).pack(side="left", padx=8)

        btns = ttk.Frame(root)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="Show command", command=self.show_command).pack(side="left")
        ttk.Button(btns, text="Run", command=self.run_command).pack(side="left", padx=8)

        self.command_preview = tk.Text(root, height=8, wrap="word")
        self.command_preview.pack(fill="both", expand=True, pady=(8, 0))

    def _path_row(self, parent: ttk.Widget, label: str, var: tk.StringVar, *, is_file: bool = False, is_dir: bool = False, is_save_file: bool = False) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=16).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

        def browse() -> None:
            if is_dir:
                chosen = filedialog.askdirectory()
            elif is_save_file:
                chosen = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4"), ("All", "*.*")])
            elif is_file:
                chosen = filedialog.askopenfilename()
            else:
                chosen = ""
            if chosen:
                var.set(chosen)

        ttk.Button(row, text="Browse", command=browse).pack(side="left", padx=6)

    def _num_row(self, parent: ttk.Widget, label: str, var: tk.Variable) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=28).pack(side="left")
        ttk.Entry(row, textvariable=var, width=16).pack(side="left")

    def _text_row(self, parent: ttk.Widget, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=28).pack(side="left")
        ttk.Entry(row, textvariable=var, width=16).pack(side="left")

    def _refresh_script_mode(self) -> None:
        is_mapper = self.script_var.get() == "mask_mapper.py"
        state = "normal" if is_mapper else "disabled"
        for child in self.mode_frame.winfo_children():
            child.configure(state=state)

    def build_command(self) -> list[str]:
        script = self.script_var.get()
        source = self.source_var.get().strip()
        if not source:
            raise ValueError("Source is required.")

        cmd: list[str] = [sys.executable, script, source]
        cmd += ["--out-dir", self.out_dir_var.get().strip() or "mask_output"]

        if self.load_dir_var.get().strip():
            cmd += ["--load-dir", self.load_dir_var.get().strip()]

        cmd += ["--scale", str(self.scale_var.get())]
        cmd += ["--alpha", str(self.alpha_var.get())]
        cmd += ["--analysis-max-distance", str(self.analysis_max_distance_var.get())]
        cmd += ["--analysis-time-window", str(self.analysis_time_window_var.get())]
        cmd += ["--analysis-sample-interval", str(self.analysis_sample_interval_var.get())]
        cmd += ["--hex-size", str(self.hex_size_var.get())]
        cmd += ["--brush-size", str(self.brush_size_var.get())]
        cmd += ["--max-display-width", str(self.max_display_width_var.get())]

        if self.show_hex_values_var.get():
            cmd.append("--show-hex-values")
        if self.use_opencl_var.get():
            cmd.append("--use-opencl")

        if script == "mask_mapper.py":
            cmd.append("--live-stream" if self.mode_var.get() == "live" else "--process-video")
            cmd += ["--live-analysis-interval", str(self.live_analysis_interval_var.get())]
            cmd += ["--stream-reconnect-delay", str(self.stream_reconnect_delay_var.get())]
            cmd += ["--rtsp-transport", self.rtsp_transport_var.get()]
            if self.average_wetness_only_var.get():
                cmd.append("--average-wetness-only")
            if self.use_obstruction_colors_var.get():
                cmd.append("--use-obstruction-colors")
            if self.analysis_source_var.get().strip():
                cmd += ["--analysis-source", self.analysis_source_var.get().strip()]
            if self.output_video_var.get().strip():
                cmd += ["--output-video", self.output_video_var.get().strip()]
            if self.fps_var.get().strip():
                cmd += ["--fps", self.fps_var.get().strip()]

        return cmd

    def show_command(self) -> None:
        try:
            cmd = self.build_command()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        preview = subprocess.list2cmdline(cmd)
        self.command_preview.delete("1.0", tk.END)
        self.command_preview.insert("1.0", preview)

    def run_command(self) -> None:
        try:
            cmd = self.build_command()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        self.show_command()
        try:
            subprocess.Popen(cmd, cwd=Path(__file__).resolve().parent)
        except Exception as exc:
            messagebox.showerror("Run failed", str(exc))


def main() -> int:
    app = Launcher()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
