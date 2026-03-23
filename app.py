"""
app.py — Control Panel GUI for the Android Sudoku Solver.

A dark-themed tkinter dashboard with:
  • "Scan & Solve" button to start a run
  • Live scrolling log showing every step
  • Board display (initial → solved)
  • Stats panel (givens, solve time, etc.)
  • Runs in a background thread so the UI stays responsive

Usage
-----
    python app.py
    python app.py --serial <device_serial>
"""

import argparse
import json
import os
import threading
import time
import tkinter as tk
from tkinter import scrolledtext
import cv2
import numpy as np

# ── Project imports ──────────────────────────────────────────────────
from device_connector import DeviceConnector
from vision import detect_grid_and_coords
from ocr import recognise_board
from solver import solve, is_valid_board
from executor import execute_solution

# ── Colour palette (dark theme) ─────────────────────────────────────
BG         = "#1e1e2e"
BG_CARD    = "#2a2a3c"
FG         = "#cdd6f4"
FG_DIM     = "#6c7086"
ACCENT     = "#89b4fa"
GREEN      = "#a6e3a1"
RED        = "#f38ba8"
YELLOW     = "#f9e2af"
BORDER     = "#45475a"
FONT_MONO  = ("Consolas", 10)
FONT_BOARD = ("Consolas", 13, "bold")
FONT_TITLE = ("Segoe UI", 14, "bold")
FONT_BTN   = ("Segoe UI", 12, "bold")
FONT_STAT  = ("Consolas", 11)


class SudokuApp:
    """Main application window."""

    def __init__(self, root: tk.Tk, device_serial: str | None = None):
        self.root = root
        self.device_serial = device_serial
        self.device: DeviceConnector | None = None
        self._running = False

        # ── Window setup ─────────────────────────────────────────────
        root.title("Sudoku Solver  —  Control Panel")
        root.configure(bg=BG)
        root.resizable(False, False)

        # ── Title bar ────────────────────────────────────────────────
        title_frame = tk.Frame(root, bg=BG)
        title_frame.pack(fill="x", padx=16, pady=(14, 4))
        tk.Label(title_frame, text="⬡  Sudoku Solver",
                 font=FONT_TITLE, bg=BG, fg=ACCENT).pack(side="left")
        self.status_dot = tk.Label(title_frame, text="●  Idle",
                                   font=("Segoe UI", 10), bg=BG, fg=FG_DIM)
        self.status_dot.pack(side="right")

        # ── Control bar ──────────────────────────────────────────────
        ctrl_frame = tk.Frame(root, bg=BG)
        ctrl_frame.pack(fill="x", padx=16, pady=6)

        self.btn_start = tk.Button(
            ctrl_frame, text="▶  Scan & Solve", font=FONT_BTN,
            bg=ACCENT, fg="#1e1e2e", activebackground="#74c7ec",
            relief="flat", padx=18, pady=6, cursor="hand2",
            command=self._on_start,
        )
        self.btn_start.pack(side="left")

        self.btn_connect = tk.Button(
            ctrl_frame, text="🔌  Connect", font=("Segoe UI", 10),
            bg=BG_CARD, fg=FG, activebackground=BORDER,
            relief="flat", padx=12, pady=4, cursor="hand2",
            command=self._on_connect,
        )
        self.btn_connect.pack(side="left", padx=(10, 0))

        # ── Middle row: boards + stats ───────────────────────────────
        mid_frame = tk.Frame(root, bg=BG)
        mid_frame.pack(fill="x", padx=16, pady=6)

        # Initial board
        board_left = tk.LabelFrame(mid_frame, text=" Initial Board ",
                                   font=("Segoe UI", 9), bg=BG_CARD,
                                   fg=FG_DIM, labelanchor="n",
                                   relief="groove", bd=1)
        board_left.pack(side="left", padx=(0, 6), fill="both", expand=True)
        self.initial_board_text = tk.Text(
            board_left, width=25, height=13, font=FONT_BOARD,
            bg=BG_CARD, fg=FG, relief="flat", state="disabled",
            highlightthickness=0, wrap="none",
        )
        self.initial_board_text.pack(padx=6, pady=6)

        # Solved board
        board_right = tk.LabelFrame(mid_frame, text=" Solved Board ",
                                    font=("Segoe UI", 9), bg=BG_CARD,
                                    fg=FG_DIM, labelanchor="n",
                                    relief="groove", bd=1)
        board_right.pack(side="left", padx=(6, 6), fill="both", expand=True)
        self.solved_board_text = tk.Text(
            board_right, width=25, height=13, font=FONT_BOARD,
            bg=BG_CARD, fg=GREEN, relief="flat", state="disabled",
            highlightthickness=0, wrap="none",
        )
        self.solved_board_text.pack(padx=6, pady=6)

        # Stats panel
        stats_frame = tk.LabelFrame(mid_frame, text=" Stats ",
                                    font=("Segoe UI", 9), bg=BG_CARD,
                                    fg=FG_DIM, labelanchor="n",
                                    relief="groove", bd=1)
        stats_frame.pack(side="left", padx=(6, 0), fill="both")
        self.stats_text = tk.Text(
            stats_frame, width=20, height=13, font=FONT_STAT,
            bg=BG_CARD, fg=YELLOW, relief="flat", state="disabled",
            highlightthickness=0, wrap="none",
        )
        self.stats_text.pack(padx=6, pady=6)

        # ── Log area ────────────────────────────────────────────────
        log_frame = tk.LabelFrame(root, text=" Live Log ",
                                  font=("Segoe UI", 9), bg=BG_CARD,
                                  fg=FG_DIM, labelanchor="n",
                                  relief="groove", bd=1)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(6, 14))

        self.log = scrolledtext.ScrolledText(
            log_frame, height=12, font=FONT_MONO,
            bg="#181825", fg=FG, relief="flat",
            insertbackground=FG, state="disabled",
            highlightthickness=0, wrap="word",
        )
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

        # Tag colours for log
        self.log.tag_config("info", foreground=FG)
        self.log.tag_config("step", foreground=ACCENT)
        self.log.tag_config("ok", foreground=GREEN)
        self.log.tag_config("warn", foreground=YELLOW)
        self.log.tag_config("err", foreground=RED)

        self._log("Ready.  Click  Connect  then  Scan & Solve.", "info")

    # ── Logging helpers ──────────────────────────────────────────────

    def _log(self, msg: str, tag: str = "info") -> None:
        """Append a line to the log (thread-safe)."""
        def _append():
            self.log.configure(state="normal")
            self.log.insert("end", msg + "\n", tag)
            self.log.see("end")
            self.log.configure(state="disabled")
        self.root.after(0, _append)

    def _set_status(self, text: str, colour: str = FG_DIM) -> None:
        """Update the status indicator."""
        def _update():
            self.status_dot.config(text=f"●  {text}", fg=colour)
        self.root.after(0, _update)

    def _set_board(self, widget: tk.Text, board: np.ndarray | None) -> None:
        """Render a 9×9 board into a Text widget."""
        def _update():
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            if board is None:
                widget.insert("end", "\n   (waiting…)")
            else:
                for r in range(9):
                    if r > 0 and r % 3 == 0:
                        widget.insert("end", " ------+-------+------\n")
                    row_str = ""
                    for c in range(9):
                        if c > 0 and c % 3 == 0:
                            row_str += " │ "
                        v = int(board[r, c])
                        row_str += f" {v}" if v != 0 else " ·"
                    widget.insert("end", f" {row_str}\n")
            widget.configure(state="disabled")
        self.root.after(0, _update)

    def _set_stats(self, lines: list[str]) -> None:
        """Update the stats panel."""
        def _update():
            self.stats_text.configure(state="normal")
            self.stats_text.delete("1.0", "end")
            for ln in lines:
                self.stats_text.insert("end", ln + "\n")
            self.stats_text.configure(state="disabled")
        self.root.after(0, _update)

    # ── Button handlers ──────────────────────────────────────────────

    def _on_connect(self) -> None:
        """Connect to the device in a background thread."""
        self.btn_connect.config(state="disabled")
        threading.Thread(target=self._connect_thread, daemon=True).start()

    def _connect_thread(self) -> None:
        try:
            self._set_status("Connecting…", YELLOW)
            self._log("Connecting to device…", "step")
            self.device = DeviceConnector(device_serial=self.device_serial)
            w, h = self.device.get_screen_size()
            self._log(f"Connected  ({w}×{h})", "ok")
            self._set_status("Connected", GREEN)
        except Exception as e:
            self._log(f"Connection failed: {e}", "err")
            self._set_status("Error", RED)
        finally:
            self.root.after(0, lambda: self.btn_connect.config(state="normal"))



    def _on_start(self) -> None:
        """Kick off a scan-and-solve run."""
        if self.device is None:
            self._log("Not connected — click Connect first.", "warn")
            return
        if self._running:
            self._log("Already running…", "warn")
            return
        self._running = True
        self.btn_start.config(state="disabled")
        threading.Thread(target=self._solve_thread, daemon=True).start()

    # ── Solve pipeline (background thread) ───────────────────────────

    def _solve_thread(self) -> None:
        try:
            t_total = time.perf_counter()
            self._set_board(self.initial_board_text, None)
            self._set_board(self.solved_board_text, None)
            self._set_stats(["  Scanning…"])

            # 0 — Pre-tap (e.g. "New Game" button)
            pretap_file = os.path.join(os.path.dirname(__file__), "pretap.json")
            if os.path.exists(pretap_file):
                with open(pretap_file, "r") as f:
                    pt_data = json.load(f)
                px, py = pt_data.get("x"), pt_data.get("y")
                if px is not None and py is not None:
                    self._set_status("Pre-tap…", YELLOW)
                    self._log(f"▸ Pre-tap → ({px}, {py}) from pretap.json", "step")
                    self.device.tap(px, py)
                    time.sleep(0.5)  # wait for screen to settle

            self._set_status("Capturing…", YELLOW)

            # 1 — Screenshot
            self._log("━" * 40, "info")
            self._log("▸ Capturing screen…", "step")
            t0 = time.perf_counter()
            screenshot = self.device.capture_screen()
            cap_time = time.perf_counter() - t0
            self._log(f"  Screenshot in {cap_time:.2f}s", "info")

            # 2 — Vision / coords
            self._set_status("Detecting grid…", YELLOW)
            self._log("▸ Detecting grid & coordinates…", "step")
            t0 = time.perf_counter()
            screen_w, screen_h = self.device.get_screen_size()

            coords_file = os.path.join(os.path.dirname(__file__), "coords.json")
            if os.path.exists(coords_file):
                self._log("  Using manual coords (coords.json)", "info")
                with open(coords_file, "r") as f:
                    cal = json.load(f)
                cell_coords = [[(c[0], c[1]) for c in row] for row in cal["cell_coords"]]
                button_coords = {int(k): (v[0], v[1]) for k, v in cal["button_coords"].items()}
                cells, _, _, _ = detect_grid_and_coords(screenshot, screen_w, screen_h)
            else:
                cells, cell_coords, button_coords, _ = detect_grid_and_coords(
                    screenshot, screen_w, screen_h)

            vis_time = time.perf_counter() - t0
            self._log(f"  Grid detected in {vis_time:.2f}s", "ok")

            # 3 — OCR
            self._set_status("Reading digits…", YELLOW)
            self._log("▸ Recognising digits (OCR)…", "step")
            t0 = time.perf_counter()
            initial_board = recognise_board(cells)
            ocr_time = time.perf_counter() - t0
            givens = int(np.count_nonzero(initial_board))
            self._log(f"  OCR done: {givens} givens in {ocr_time:.2f}s", "ok")
            self._set_board(self.initial_board_text, initial_board)

            if not is_valid_board(initial_board):
                self._log("  ⚠ Board has conflicts — OCR may have errors", "warn")

            # 4 — Solve
            self._set_status("Solving…", YELLOW)
            self._log("▸ Solving puzzle…", "step")
            solved_board = initial_board.copy()
            t0 = time.perf_counter()
            success = solve(solved_board)
            solve_time = time.perf_counter() - t0

            if not success:
                self._log("✗ Solver failed — board is unsolvable (OCR error?)", "err")
                self._set_status("Failed", RED)
                return

            self._log(f"  Solved in {solve_time * 1000:.1f} ms", "ok")
            self._set_board(self.solved_board_text, solved_board)

            # 5 — Execute
            self._set_status("Filling board…", ACCENT)
            self._log("▸ Executing quick-fill…", "step")
            t0 = time.perf_counter()
            empties = 81 - givens

            for digit in range(1, 10):
                targets = []
                for r in range(9):
                    for c in range(9):
                        if initial_board[r, c] == 0 and solved_board[r, c] == digit:
                            targets.append(cell_coords[r][c])
                if not targets:
                    continue
                bx, by = button_coords[digit]
                self._log(f"  [{digit}]  {len(targets)} cells", "info")
                sequence = [(bx, by)]
                for tx, ty in targets:
                    sequence.append((tx, ty))

                self.device.tap_sequence(sequence)

            exec_time = time.perf_counter() - t0
            total_time = time.perf_counter() - t_total

            # Stats
            self._set_stats([
                f"  Givens:  {givens}",
                f"  Filled:  {empties}",
                "",
                f"  OCR:     {ocr_time:.2f}s",
                f"  Solve:   {solve_time*1000:.1f}ms",
                f"  Tapping: {exec_time:.1f}s",
                "",
                f"  Total:   {total_time:.1f}s",
            ])

            self._log(f"✓ Done — {empties} cells filled in {total_time:.1f}s", "ok")
            self._set_status("Done  ✓", GREEN)

        except Exception as e:
            self._log(f"Error: {e}", "err")
            self._set_status("Error", RED)
        finally:
            self._running = False
            self.root.after(0, lambda: self.btn_start.config(state="normal"))


def main():
    parser = argparse.ArgumentParser(description="Sudoku Solver Control Panel")
    parser.add_argument("--serial", type=str, default=None)
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("760x620")
    app = SudokuApp(root, device_serial=args.serial)
    root.mainloop()


if __name__ == "__main__":
    main()
