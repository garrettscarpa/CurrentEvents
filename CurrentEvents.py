#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voltage Clamp Analysis Suite (PyQt5)
A unified GUI for event detection, curation, and results plotting.
"""

import os
import sys

# ── try to import heavy deps; show a friendly error if missing ──────────────
_missing = []
try:
    import numpy as np
except ImportError:
    _missing.append("numpy")
try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Polygon
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:
    _missing.append("matplotlib")
try:
    from scipy.signal import butter, filtfilt
    from scipy.stats import ttest_ind
except ImportError:
    _missing.append("scipy")
try:
    import pyabf
except ImportError:
    _missing.append("pyabf")

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
except ImportError:
    _missing.append("PyQt5")

if _missing:
    # Try to show a Qt dialog; fall back to console if even Qt is missing.
    try:
        from PyQt5 import QtWidgets as _QtW
        _app = _QtW.QApplication(sys.argv)
        _QtW.QMessageBox.critical(
            None, "Missing dependencies",
            "Please install the following packages before running:\n\n"
            + "\n".join(f"  pip install {p}" for p in _missing)
        )
    except Exception:
        print("Missing dependencies:\n" + "\n".join(f"  pip install {p}" for p in _missing))
    sys.exit(1)

from collections import defaultdict
from itertools import combinations

# NumPy 2.0 renamed np.trapz -> np.trapezoid; provide a uniform helper.
_trapz = getattr(np, "trapz", None) or getattr(np, "trapezoid")


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED SIGNAL-PROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _highpass(data, cutoff, fs):
    b, a = butter(4, cutoff / (0.5 * fs), btype="high")
    return filtfilt(b, a, data)

def _lowpass(data, cutoff, fs):
    b, a = butter(4, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, data)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisSession:
    """Holds all mutable state for one analysis run."""

    def __init__(self, params):
        self.p = params          # dict of user-configured parameters
        self.events = []
        self.trace_cache = {}
        self.current_event = 0
        self.recordings = []

    # ------------------------------------------------------------------
    def discover_recordings(self):
        root = self.p["root"]
        self.recordings = []
        for subdir, _, files in os.walk(root):
            for f in files:
                if f.endswith(".abf"):
                    self.recordings.append((subdir, f))
        return len(self.recordings)

    # ------------------------------------------------------------------
    def load_trace(self, event):
        key = (event["recording_path"], event["file"])
        if key in self.trace_cache:
            return self.trace_cache[key]
        p = self.p
        abf = pyabf.ABF(os.path.join(event["recording_path"], event["file"]))
        abf.setSweep(0)
        fs = abf.dataRate
        time = abf.sweepX
        signal = abf.sweepY
        mask = (time >= p["analysis_start_s"]) & (time <= p["analysis_end_s"])
        time = time[mask]; signal = signal[mask]
        filtered = _highpass(signal, 1, fs)
        filtered = _lowpass(filtered, 1000, fs)
        self.trace_cache[key] = (time, filtered, fs)
        return time, filtered, fs

    # ------------------------------------------------------------------
    def process_all(self, progress_cb=None):
        """Detect events in every recording. Calls progress_cb(i, total, filename)."""
        self.events = []
        for idx, (rec_dir, rec_file) in enumerate(self.recordings):
            if progress_cb:
                progress_cb(idx, len(self.recordings), rec_file)
            self._process_recording(rec_dir, rec_file)
        if progress_cb:
            progress_cb(len(self.recordings), len(self.recordings), "done")

    # ------------------------------------------------------------------
    def _process_recording(self, rec_dir, rec_file):
        p = self.p
        abf = pyabf.ABF(os.path.join(rec_dir, rec_file))
        abf.setSweep(0)
        fs = abf.dataRate
        time = abf.sweepX; signal = abf.sweepY
        mask = (time >= p["analysis_start_s"]) & (time <= p["analysis_end_s"])
        time_f = time[mask]; signal_f = signal[mask]

        filtered = _highpass(signal_f, 1, fs)
        filtered = _lowpass(filtered, 1000, fs)

        mad = np.median(np.abs(filtered - np.median(filtered)))
        noise = mad / 0.6745
        threshold = -p["threshold_sigma"] * noise

        save_file = os.path.join(rec_dir, "event_status.npy")
        saved_events = None
        if os.path.exists(save_file):
            try:
                saved = np.load(save_file, allow_pickle=True).item()
                saved_events = saved.get("events", None)
            except Exception:
                saved_events = None

        if saved_events is not None:
            restored = 0
            for ev in saved_events:
                if ev.get("file") != rec_file:
                    continue
                ev = dict(ev)
                ev["recording_path"] = rec_dir
                ev["file"] = rec_file
                ev.setdefault("manually_added", False)
                ev.setdefault("auc", None)
                self.events.append(ev)
                restored += 1
            if restored > 0:
                return

        # Fresh detection
        crossings = []
        for i in range(1, len(filtered)):
            if filtered[i-1] > threshold and filtered[i] <= threshold:
                crossings.append(time_f[i])

        min_event_samples = 0
        filtered_crossings = []
        for ct in crossings:
            idx = np.argmin(np.abs(time_f - ct))
            sw = int(2e-3 * fs)
            seg = filtered[max(0, idx-sw): min(len(filtered), idx+sw)]
            below = seg <= threshold
            max_run = run = 0
            for b in below:
                if b: run += 1; max_run = max(max_run, run)
                else: run = 0
            if max_run >= min_event_samples:
                filtered_crossings.append(ct)

        window_samples = int(15 / 1000 * fs)
        for ct in filtered_crossings:
            idx = np.argmin(np.abs(time_f - ct))
            start = max(0, idx - int(2e-3 * fs))
            end = start + window_samples
            if end >= len(filtered): continue

            segment = filtered[start:end]
            baseline = np.median(segment[:5])
            peak = np.min(segment)
            amp = baseline - peak

            if not (p["min_amp"] < amp < p["max_amp"]): continue

            peak_idx_local = np.argmin(segment)
            peak_abs_idx = start + peak_idx_local
            art_win = int(p["noise_filter_window_ms"] * fs)
            art_start = max(0, peak_abs_idx - art_win // 2)
            art_end = min(len(filtered), peak_abs_idx + art_win)
            if np.any(filtered[art_start:art_end] >= p["noise_filter_pos_voltage"]):
                continue

            self.events.append({
                "recording_path": rec_dir, "file": rec_file,
                "ct": ct, "status": True,
                "base": None, "peak_t": None,
                "amp": amp, "threshold": threshold,
                "auc": None, "manually_added": False,
            })

    # ------------------------------------------------------------------
    def init_bases(self):
        for event in self.events:
            if event["base"] is None:
                ct = event["ct"]
                time, filtered, fs = self.load_trace(event)
                event["base"] = [max(time[0], ct - 0.001), min(time[-1], ct + 0.001)]

    # ------------------------------------------------------------------
    def save_all(self):
        grouped = defaultdict(list)
        for e in self.events:
            grouped[e["recording_path"]].append(e)
        for rec_path, rec_events in grouped.items():
            out_file = os.path.join(rec_path, "event_status.npy")
            np.save(out_file, {"events": rec_events}, allow_pickle=True)
        return len(grouped)

    # ------------------------------------------------------------------
    def compute_auc(self, event):
        time, filtered, fs = self.load_trace(event)
        left_x, right_x = event["base"]
        mask = (time >= left_x) & (time <= right_x)
        if not np.any(mask): event["auc"] = None; return None
        seg_t = time[mask]; seg_y = filtered[mask]
        y_left = np.interp(left_x, time, filtered)
        y_right = np.interp(right_x, time, filtered)
        baseline = np.interp(seg_t, [left_x, right_x], [y_left, y_right])
        auc = float(np.abs(_trapz(seg_y - baseline, seg_t)))
        event["auc"] = auc
        return auc

    def compute_peak_between_bases(self, event):
        time, filtered, fs = self.load_trace(event)
        if event.get("base") is None:
            return self._search_peak(event)

        left_x, right_x = event["base"]
        mask = (time >= left_x) & (time <= right_x)
        if not np.any(mask): return self._search_peak(event)
        seg_t = time[mask]; seg_y = filtered[mask]
        peak_rel = np.argmin(seg_y)
        event["peak_t"] = float(seg_t[peak_rel])
        return seg_t[peak_rel], seg_y[peak_rel]

    def _search_peak(self, event):
        time, filtered, fs = self.load_trace(event)
        ct = event["ct"]
        idx = np.argmin(np.abs(time - ct))
        half = int(5e-3 * fs)
        s = max(0, idx - half); e = min(len(filtered), idx + half)
        peak_rel = np.argmin(filtered[s:e])
        peak_idx = s + peak_rel
        t_peak = time[peak_idx]
        event["peak_t"] = float(t_peak)
        return t_peak, filtered[peak_idx]

    def get_peak(self, event):
        time, filtered, _ = self.load_trace(event)
        if event["peak_t"] is not None:
            t_peak = event["peak_t"]
            y_peak = np.interp(t_peak, time, filtered)
            return t_peak, y_peak
        return self._search_peak(event)

    def get_recording_key(self, event):
        return (event["recording_path"], event["file"])

    def insert_manual_peak(self, click_time):
        ref_event = self.events[self.current_event]
        rec_key = self.get_recording_key(ref_event)
        rec_dir = ref_event["recording_path"]
        rec_file = ref_event["file"]
        threshold_val = ref_event["threshold"]
        time, filtered, fs = self.load_trace(ref_event)

        search_half = int(5e-3 * fs)
        click_idx = np.argmin(np.abs(time - click_time))
        s = max(0, click_idx - search_half)
        e = min(len(filtered), click_idx + search_half)
        local_min_rel = np.argmin(filtered[s:e])
        peak_abs_idx = s + local_min_rel
        t_peak = float(time[peak_abs_idx])

        base_half = int(3e-3 * fs)
        left_x = float(time[max(0, peak_abs_idx - base_half)])
        right_x = float(time[min(len(time)-1, peak_abs_idx + base_half)])

        new_event = {
            "recording_path": rec_dir, "file": rec_file,
            "ct": t_peak, "status": True,
            "base": [left_x, right_x], "peak_t": t_peak,
            "amp": 0.0, "threshold": threshold_val,
            "auc": None, "manually_added": True,
        }
        y_peak = filtered[peak_abs_idx]
        baseline_est = np.median(filtered[max(0, peak_abs_idx - int(2e-3*fs)): peak_abs_idx+1])
        new_event["amp"] = float(abs(baseline_est - y_peak))
        self.compute_auc(new_event)

        same_rec_indices = [i for i, ev in enumerate(self.events) if self.get_recording_key(ev) == rec_key]
        insert_pos = None
        for gi in same_rec_indices:
            tp, _ = self.get_peak(self.events[gi])
            if tp > t_peak:
                insert_pos = gi; break
        if insert_pos is None:
            insert_pos = same_rec_indices[-1] + 1 if same_rec_indices else len(self.events)

        self.events.insert(insert_pos, new_event)
        self.current_event = insert_pos


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS WINDOW  (matplotlib canvas embedded in a QMainWindow)
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisWindow(QtWidgets.QMainWindow):

    closed = pyqtSignal(object)   # emits n_saved (int) or None

    def __init__(self, session: AnalysisSession, parent=None):
        super().__init__(parent)
        self.s = session
        self._add_peak_active = False
        self._dragging = None
        self._drag_threshold = 0.002
        self._top_panel_key = None
        self._view_xlim = [None, None]
        self._closed_emitted = False
        self.current_xmax = self.s.p["window_samples_view_ms"]
        self.setWindowTitle("Analysis Window")
        self.resize(1200, 800)

        # debounce timer for arrow-key navigation
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(20)
        self._debounce.timeout.connect(self._update_plot)

        self._build_ui()
        self.s.init_bases()
        self._update_top_panel(force=True)
        self._update_plot()

        # The matplotlib canvas (a child widget) takes keyboard focus, so
        # QMainWindow.keyPressEvent never sees arrow/Q keys. Install an event
        # filter on the QApplication to catch key presses for this window
        # regardless of which child widget is focused.
        QtWidgets.QApplication.instance().installEventFilter(self)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # ── matplotlib figure / canvas ──
        # Use Figure() directly (not plt.figure()) so pyplot does not try to
        # manage / create a second native window for this embedded canvas —
        # mixing the two crashes hard on macOS.
        self.fig = Figure(figsize=(13, 8))
        self.fig.patch.set_facecolor("#1a1a2e")

        gs = gridspec.GridSpec(2, 1, height_ratios=[1.6, 1], hspace=0.45,
                               left=0.08, right=0.97, top=0.93, bottom=0.06)
        self.ax_top = self.fig.add_subplot(gs[0])
        self.ax_bot = self.fig.add_subplot(gs[1])

        for ax in (self.ax_top, self.ax_bot):
            ax.set_facecolor("#0d0d1a")
            for sp in ax.spines.values(): sp.set_edgecolor("#444466")
            ax.tick_params(colors="#aaaacc", labelsize=8)
            ax.xaxis.label.set_color("#aaaacc")
            ax.yaxis.label.set_color("#aaaacc")
            ax.title.set_color("#ddddff")

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        vbox.addWidget(self.canvas, stretch=1)

        # matplotlib canvas event hooks (mouse only — keys handled by Qt)
        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event",  self._on_motion)

        # ── control bar (native Qt buttons replace matplotlib widgets) ──
        bar = QtWidgets.QWidget()
        bar.setObjectName("ctrlBar")
        bar.setStyleSheet(
            "#ctrlBar { background:#2c303a; border-top:1px solid #3a3f4b; }")
        hbox = QtWidgets.QHBoxLayout(bar)
        hbox.setContentsMargins(12, 8, 12, 8)
        hbox.setSpacing(10)

        self.x_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(1)      # ms
        self.x_slider.setMaximum(max(500, int(self.current_xmax) * 5))  # ms
        self.x_slider.setValue(int(self.current_xmax))
        self.x_slider.valueChanged.connect(self._change_xrange)

        hbox.addWidget(QtWidgets.QLabel("X Range (ms)"))
        hbox.addWidget(self.x_slider)

        self.btn_add_peak = QtWidgets.QPushButton("Add Peak")
        self.btn_add_peak.setCursor(Qt.PointingHandCursor)
        self.btn_add_peak.clicked.connect(self._toggle_add_peak)
        self._style_button(self.btn_add_peak, "#3c4250", "#e3e3ea")
        hbox.addWidget(self.btn_add_peak)

        hbox.addWidget(self._make_label("Go to #"))
        self.txt_jump = QtWidgets.QLineEdit()
        self.txt_jump.setFixedWidth(70)
        self.txt_jump.setValidator(QtGui.QIntValidator(1, 10_000_000, self))
        self.txt_jump.setStyleSheet(
            "QLineEdit { background:#1e2128; color:#e3e3ea; border:1px solid #3a3f4b;"
            " border-radius:5px; padding:4px; }"
            " QLineEdit:focus { border:1px solid #7c6cf0; }")
        self.txt_jump.returnPressed.connect(self._jump_to_peak)
        hbox.addWidget(self.txt_jump)

        hbox.addStretch(1)

        nav_hint = self._make_label(
            "← →: navigate   |   ↑: accept   |   ↓: reject   |   Q: save & quit")
        nav_hint.setStyleSheet("color:#9aa0ad; background:transparent;")
        hbox.addWidget(nav_hint)

        hbox.addStretch(1)

        self.btn_save = QtWidgets.QPushButton("Save && Close")
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.clicked.connect(self._save_and_close)
        self._style_button(self.btn_save, "#3b8268", "#eafff5")
        hbox.addWidget(self.btn_save)

        vbox.addWidget(bar, stretch=0)

        self.canvas.setFocus()

    @staticmethod
    def _make_label(text):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color:#cfd3dc; font-size:12px; background:transparent;")
        return lbl

    @staticmethod
    def _style_button(btn, bg, fg, padding="7px 14px"):
        btn.setStyleSheet(f"""
            QPushButton {{
                background:{bg}; color:{fg}; border:1px solid #3a3f4b;
                border-radius:6px; padding:{padding}; font-size:12px;
            }}
            QPushButton:hover {{ background:#4a515f; color:white; }}
        """)

    # ── key handling (Qt-native) ──────────────────────────────────────────────

    def eventFilter(self, obj, ev):
        # Catch key presses for this window even when a child widget (the
        # matplotlib canvas) holds focus. Only act when this window is active.
        if ev.type() == QtCore.QEvent.KeyPress and self.isActiveWindow():
            if self.txt_jump.hasFocus():
                return False  # let the jump box handle typing/Enter
            if self._handle_key(ev.key()):
                return True   # consumed
        return super().eventFilter(obj, ev)

    def keyPressEvent(self, ev):
        # Fallback if the window itself is focused.
        if self.txt_jump.hasFocus():
            super().keyPressEvent(ev)
            return
        if not self._handle_key(ev.key()):
            super().keyPressEvent(ev)

    def _handle_key(self, key):
        """Returns True if the key was handled."""
        s = self.s
        if not s.events:
            return False
        if key == Qt.Key_Right:
            s.current_event = (s.current_event + 1) % len(s.events)
            self._schedule_update()
        elif key == Qt.Key_Left:
            s.current_event = (s.current_event - 1) % len(s.events)
            self._schedule_update()
        elif key == Qt.Key_Up:
            s.events[s.current_event]["status"] = True
            self._schedule_update()
        elif key == Qt.Key_Down:
            s.events[s.current_event]["status"] = False
            self._schedule_update()
        elif key == Qt.Key_Q:
            self._save_and_close()
        else:
            return False
        return True

    def _schedule_update(self):
        """Debounce redraws so holding an arrow key doesn't flood the canvas."""
        self._debounce.start()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _toggle_add_peak(self, _=None):
        self._add_peak_active = not self._add_peak_active
        if self._add_peak_active:
            self.btn_add_peak.setText("Click on Signal")
            self._style_button(self.btn_add_peak, "#6b4fd6", "white")
        else:
            self.btn_add_peak.setText("Add Peak")
            self._style_button(self.btn_add_peak, "#3c4250", "#e3e3ea")
        self._update_plot()

    def _jump_to_peak(self):
        text = self.txt_jump.text().strip()
        try:
            n = int(text)
        except ValueError:
            return
        s = self.s
        if not (1 <= n <= len(s.events)):
            return
        s.current_event = n - 1
        self._top_panel_key = None
        self.txt_jump.clearFocus()
        self.canvas.setFocus()
        self._update_plot()

    def _save_and_close(self, _=None):
        n = self.s.save_all()
        self._closed_emitted = True
        self.closed.emit(n)
        self.close()

    def closeEvent(self, ev):
        # Remove the application-level event filter we installed in __init__.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        # If the window is closed via the X (not Save & Close), report "not saved".
        if not self._closed_emitted:
            self._closed_emitted = True
            self.closed.emit(None)
        super().closeEvent(ev)

    # ── mouse callbacks (matplotlib canvas) ────────────────────────────────────

    def _on_press(self, ev):
        s = self.s
        if self._add_peak_active:
            if ev.inaxes == self.ax_bot and ev.xdata is not None:
                self._toggle_add_peak()
                s.insert_manual_peak(ev.xdata)
                self._top_panel_key = None
                self._update_plot()
            return
        if ev.inaxes != self.ax_bot: return
        cur = s.events[s.current_event]
        lx, rx = cur["base"]
        if ev.xdata is None: return
        if abs(ev.xdata - lx) < self._drag_threshold:
            self._dragging = "left"
        elif abs(ev.xdata - rx) < self._drag_threshold:
            self._dragging = "right"

    def _on_release(self, _=None):
        self._dragging = None

    def _on_motion(self, ev):
        if self._dragging is None or ev.inaxes != self.ax_bot: return
        if ev.xdata is None: return
        s = self.s
        cur = s.events[s.current_event]
        lx, rx = cur["base"]
        if self._dragging == "left":
            cur["base"][0] = min(ev.xdata, rx - 1e-4)
        elif self._dragging == "right":
            cur["base"][1] = max(ev.xdata, lx + 1e-4)
        s.compute_peak_between_bases(cur)
        s.compute_auc(cur)
        self._update_plot()

    # ── rendering ─────────────────────────────────────────────────────────────

    def _update_top_panel(self, force=False):
        s = self.s
        p = s.p
        event = s.events[s.current_event]
        rec_key = s.get_recording_key(event)
        time, filtered, fs = s.load_trace(event)
        rec_events = [e for e in s.events if s.get_recording_key(e) == rec_key]

        if self._top_panel_key != rec_key or force:
            self._top_panel_key = rec_key
            self.ax_top.clear()
            self.ax_top.set_facecolor("#0d0d1a")
            for sp in self.ax_top.spines.values(): sp.set_edgecolor("#444466")
            self.ax_top.tick_params(colors="#aaaacc", labelsize=8)
            self.ax_top.xaxis.label.set_color("#aaaacc")
            self.ax_top.yaxis.label.set_color("#aaaacc")

            ds = max(1, len(time) // 20000)
            if ds > 1:
                n_blocks = len(time) // ds
                t_ds = time[:n_blocks*ds].reshape(n_blocks, ds)[:, ds//2]
                y_ds = filtered[:n_blocks*ds].reshape(n_blocks, ds)
                y_min = y_ds.min(axis=1); y_max = y_ds.max(axis=1)
                t_env = np.repeat(t_ds, 2)
                y_env = np.empty(len(t_ds)*2)
                y_env[0::2] = y_min; y_env[1::2] = y_max
                self.ax_top.plot(t_env, y_env, color="#4a6fa5", lw=0.5, alpha=0.9)
            else:
                self.ax_top.plot(time, filtered, color="#4a6fa5", lw=0.6, alpha=0.9)

            thr = event["threshold"]
            self.ax_top.axhline(thr, color="#e05c5c", lw=0.8, ls="--", alpha=0.7,
                                label=f"Threshold ({thr:.2f} pA)")
            for e in rec_events:
                color = "#55dd88" if e["status"] else "#dd5566"
                if e.get("manually_added"): color = "#ffaa33" if e["status"] else "#dd5566"
                if e.get("peak_t") is None:
                    continue
                t_pk = e["peak_t"]
                self.ax_top.axvline(t_pk, color=color, lw=0.6, alpha=0.5, zorder=2)

            self.ax_top.set_ylabel("Current (pA)", fontsize=8)
            self.ax_top.set_xlabel("Time (s)", fontsize=8)
            self.ax_top.set_ylim(p["ylim"])
            self.ax_top.legend(fontsize=7, loc="upper right",
                               facecolor="#1a1a2e", edgecolor="#444466",
                               labelcolor="#aaaacc")

        for artist in self.ax_top.lines[:]:
            if getattr(artist, "_cur_marker", False): artist.remove()
        for coll in self.ax_top.collections[:]:
            if getattr(coll, "_cur_marker", False): coll.remove()

        t_peak, _ = s.get_peak(event)
        ylims = self.ax_top.get_ylim()
        vl = self.ax_top.axvline(t_peak, color="#ffdd55", lw=1.5, zorder=5)
        vl._cur_marker = True
        tri = self.ax_top.scatter([t_peak], [ylims[1]*0.97], marker="v",
                                   color="#ffdd55", s=60, zorder=6, clip_on=False)
        tri._cur_marker = True

        rec_keys = list(dict.fromkeys(s.get_recording_key(e) for e in s.events))
        rec_n = rec_keys.index(rec_key) + 1
        n_rec = len(rec_keys)
        self.ax_top.set_title(
            f"Recording {rec_n}/{n_rec}  —  {event['file']}   "
            f"({len(rec_events)} events detected)",
            fontsize=9, color="#ddddff"
        )

    def _change_xrange(self, value):
        self.current_xmax = value
        self._update_plot()

    def _update_plot(self):
        s = self.s
        p = s.p
        event = s.events[s.current_event]
        time, filtered, fs = s.load_trace(event)

        t_peak, y_peak = s.get_peak(event)

        if event.get("auc") is None:
            s.compute_auc(event)

        # Center the view on the peak TIME (not on clamped sample indices) so
        # the peak stays exactly centered even when it sits near the start/end
        # of the trace. slider value is in ms.
        half_window_s = (self.current_xmax / 1000.0) / 2.0
        new_xlim = (t_peak - half_window_s, t_peak + half_window_s)
        self._view_xlim = list(new_xlim)

        self.ax_bot.clear()
        self.ax_bot.set_facecolor("#0d0d1a")
        for sp in self.ax_bot.spines.values(): sp.set_edgecolor("#444466")
        self.ax_bot.tick_params(colors="#aaaacc", labelsize=8)
        self.ax_bot.xaxis.label.set_color("#aaaacc")
        self.ax_bot.yaxis.label.set_color("#aaaacc")

        dt = np.median(np.diff(time))
        view_mask = (time >= new_xlim[0] - dt) & (time <= new_xlim[1] + dt)
        t = time[view_mask]; y = filtered[view_mask]
        if len(t) < 2: return

        self.ax_bot.plot(t, y, color="#7ecfff", lw=1.2)
        self.ax_bot.scatter(t_peak, y_peak, color="#ff6b6b", s=60, zorder=5)

        lx, rx = event["base"]
        ly = np.interp(lx, t, y); ry = np.interp(rx, t, y)
        self.ax_bot.scatter(lx, ly, color="#55ccff", s=60, zorder=6)
        self.ax_bot.scatter(rx, ry, color="#55ccff", s=60, zorder=6)

        mask = (t >= lx) & (t <= rx)
        if np.any(mask):
            t_seg = t[mask]; y_seg = y[mask]
            top    = np.column_stack([t_seg, y_seg])
            bottom = np.array([[rx, ry], [lx, ly]])
            verts  = np.vstack([top, bottom])
            poly   = Polygon(verts, closed=True, facecolor="#aa55ff", alpha=0.25, edgecolor=None)
            self.ax_bot.add_patch(poly)

        if self._add_peak_active:
            self.ax_bot.set_facecolor("#0a1a0a")
            self.ax_bot.text(0.5, 0.97, "Click on the signal to place a new peak",
                             transform=self.ax_bot.transAxes, ha="center", va="top",
                             color="#88ff99", fontsize=8, alpha=0.85)

        if event["status"]:
            label = "ACCEPTED"; title_color = "#88ff99"
            if not self._add_peak_active: self.ax_bot.set_facecolor("#0a1f0a")
        else:
            label = "REJECTED"; title_color = "#ff8888"
            if not self._add_peak_active: self.ax_bot.set_facecolor("#1f0a0a")

        manual_tag = "  [manual]" if event.get("manually_added") else ""
        self.ax_bot.set_title(
            f"Event {s.current_event+1}/{len(s.events)}  —  {label}{manual_tag}   "
            f"[← →: navigate  |  ↑: accept  |  ↓: reject  |  Q: save & quit]",
            fontsize=8.5, color=title_color
        )
        self.ax_bot.set_xlabel("Time (s)", fontsize=8)
        self.ax_bot.set_ylabel("Current (pA)", fontsize=8)
        # Apply limits last and lock autoscale OFF so that data added above
        # (trace, scatter markers, and especially the AUC Polygon) cannot
        # re-expand or shift the view. Without this, the patch's data extent
        # biases the axis and the peak drifts off-center when zooming.
        self.ax_bot.set_autoscale_on(False)
        self.ax_bot.set_ylim(p["ylim"])
        self.ax_bot.set_xlim(new_xlim[0], new_xlim[1])

        self._update_top_panel(force=False)
        self.canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _sem(vals):
    arr = np.array([v for v in vals if not np.isnan(v)])
    if len(arr) < 2: return 0.0
    return np.nanstd(arr, ddof=1) / np.sqrt(len(arr))

def _pvalue_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"

def _extract_features(wf, t, left_base_val=None, right_base_val=None, right_t_ms=None):
    if right_t_ms is not None and right_t_ms > 0:
        event_mask = (t >= 0) & (t <= right_t_ms)
    else:
        event_mask = (t >= 0)
    if not np.any(event_mask): event_mask = np.ones(len(t), dtype=bool)

    peak_idx_local = np.argmin(wf[event_mask])
    peak_idx = np.where(event_mask)[0][peak_idx_local]
    peak_val = wf[peak_idx]

    base_level = ((left_base_val + right_base_val) / 2.0
                  if left_base_val is not None and right_base_val is not None else 0.0)
    prominence = np.abs(peak_val - base_level)

    half_max = peak_val / 2.0
    below_masked = (wf <= half_max) & event_mask
    if np.any(below_masked):
        indices = np.where(below_masked)[0]
        width = t[indices[-1]] - t[indices[0]]
    else:
        width = np.nan

    ttp = max(t[peak_idx] - 0.0, 0.0)
    recovery = max(right_t_ms - t[peak_idx], 0.0) if right_t_ms is not None else np.nan
    return prominence, width, ttp, recovery


def run_results(params):
    """Full results computation + plotting. Runs on calling thread."""
    from collections import defaultdict
    root = params["root"]
    analysis_start_s  = params["analysis_start_s"]
    analysis_end_s    = params["analysis_end_s"]
    window_ms_pre     = params["window_ms_pre"]
    total_window_ms   = params["total_window_ms"]
    highpass_cutoff   = params["highpass_cutoff"]
    lowpass_cutoff    = params["lowpass_cutoff"]
    halfwidth_threshold_ms = params["halfwidth_threshold_ms"]
    ylim              = params["ylim"]
    accepted_only     = params["accepted_only"]
    split_by_halfwidth = params["split_by_halfwidth"]

    trace_cache = {}

    def load_filtered(rec_path, rec_file):
        key = (rec_path, rec_file)
        if key in trace_cache: return trace_cache[key]
        abf = pyabf.ABF(os.path.join(rec_path, rec_file)); abf.setSweep(0)
        fs = abf.dataRate; time = abf.sweepX; signal = abf.sweepY
        mask = (time >= analysis_start_s) & (time <= analysis_end_s)
        time = time[mask]; signal = signal[mask]
        signal = _highpass(signal, highpass_cutoff, fs)
        signal = _lowpass(signal, lowpass_cutoff, fs)
        trace_cache[key] = (time, signal, fs)
        return time, signal, fs

    event_features       = defaultdict(lambda: {"prominence":[], "width":[], "ttp":[], "rec":[], "base_dur":[], "auc":[]})
    condition_recordings = defaultdict(set)
    condition_waveforms  = defaultdict(list)
    recording_features   = []

    event_files = [
        os.path.join(subdir, "event_status.npy")
        for subdir, _, files in os.walk(root)
        if "event_status.npy" in files
    ]

    # ─────────────────────────────────────────────
    # AUTO-FALLBACK: if no saved analysis exists, run detection directly
    # ─────────────────────────────────────────────
    if len(event_files) == 0:
        print("[INFO] No event_status.npy found → running fresh detection")

        det_params = {
            "root": root,
            "analysis_start_s": analysis_start_s,
            "analysis_end_s": analysis_end_s,
            "threshold_sigma": 5,
            "min_amp": 6,
            "max_amp": 80,
            "noise_filter_window_ms": 5,
            "noise_filter_pos_voltage": 10,
        }

        session = AnalysisSession(det_params)
        session.discover_recordings()
        session.process_all()

        event_files = []
        tmp = defaultdict(list)

        for e in session.events:
            tmp[e["recording_path"]].append(e)

        for rec_path, rec_events in tmp.items():
            out_file = os.path.join(rec_path, "event_status.npy")
            np.save(out_file, {"events": rec_events}, allow_pickle=True)
            event_files.append(out_file)

        print(f"[INFO] Generated {len(event_files)} temporary event files")

    for event_file in event_files:
        try: saved = np.load(event_file, allow_pickle=True).item()
        except Exception: continue
        events = saved.get("events", [])
        if not events: continue

        condition = os.path.basename(os.path.dirname(event_file))
        if accepted_only: events = [e for e in events if e.get("status", True)]

        rec_groups = defaultdict(list)
        for e in events:
            rec_groups[(e["recording_path"], e["file"])].append(e)

        for (rec_path, rec_file), rec_events in rec_groups.items():
            time, signal, fs = load_filtered(rec_path, rec_file)
            pre_samples   = int(window_ms_pre   / 1000 * fs)
            total_samples = int(total_window_ms / 1000 * fs)
            waveforms = []

            for e in rec_events:
                if e.get("base") is None: continue
                left_t, right_t = e["base"]
                left_idx  = np.argmin(np.abs(time - left_t))
                right_idx = np.argmin(np.abs(time - right_t))
                lbv = float(signal[left_idx])  if left_idx  < len(signal) else None
                rbv = float(signal[right_idx]) if right_idx < len(signal) else None
                start = left_idx - pre_samples
                end   = start + total_samples
                if start < 0 or end >= len(signal): continue

                wf = signal[start:end].copy()
                wf -= np.mean(wf[:pre_samples])
                t_wf = (np.arange(wf.size) - pre_samples) / fs * 1000
                right_t_ms = (right_t - left_t) * 1000
                prom, width, ttp, rec = _extract_features(wf, t_wf, lbv, rbv, right_t_ms)
                auc = e.get("auc") or np.nan

                if split_by_halfwidth:
                    if np.isnan(width): continue
                    pop = "narrow" if width <= halfwidth_threshold_ms else "wide"
                    cond_label = f"{condition}_{pop}"
                else:
                    cond_label = condition

                for key, val in [("prominence", prom), ("width", width), ("ttp", ttp),
                                  ("rec", rec), ("base_dur", right_t_ms), ("auc", auc)]:
                    event_features[cond_label][key].append(val)
                waveforms.append((cond_label, wf))

            if not waveforms: continue
            wf_by_label = defaultdict(list)
            for lbl, wf in waveforms: wf_by_label[lbl].append(wf)
            duration_s = analysis_end_s - analysis_start_s
            for lbl, wf_list in wf_by_label.items():
                recording_features.append({"condition": lbl, "frequency": len(wf_list)/duration_s})
                condition_waveforms[lbl].extend(wf_list)
                condition_recordings[lbl].add((rec_path, rec_file))

    raw_frequency = defaultdict(list)
    for r in recording_features: raw_frequency[r["condition"]].append(r["frequency"])

    conditions = sorted(set(event_features.keys()) | set(raw_frequency.keys()))
    condition_N = {c: len(condition_recordings[c]) for c in conditions}
    if not conditions:
        raise RuntimeError("No valid conditions found — run analysis first.")

    feature_summary = {k: [] for k in ("condition","frequency","frequency_sem",
                                        "prominence","prominence_sem","width","width_sem",
                                        "ttp","ttp_sem","rec","rec_sem","auc","auc_sem")}
    for cond in conditions:
        feature_summary["condition"].append(cond)
        fv = raw_frequency.get(cond, [])
        feature_summary["frequency"].append(np.nanmean(fv) if fv else np.nan)
        feature_summary["frequency_sem"].append(_sem(fv))
        for key in ("prominence","width","ttp","rec","auc"):
            vals = [v for v in event_features[cond][key] if v is not None]
            feature_summary[key].append(np.nanmean(vals) if vals else np.nan)
            feature_summary[f"{key}_sem"].append(_sem(vals))

    feature_lookup = {}
    for i, cond in enumerate(feature_summary["condition"]):
        feature_lookup[cond] = {k: feature_summary[k][i] for k in feature_summary if k != "condition"}

    # ── Matplotlib style ──────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "figure.dpi": 150, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    _palette = ["#1d1d1d","#7B2CBF","#9D4EDD","#5A189A","#C77DFF",
                "#2196F3","#FF5722","#00897B"]

    # ── Plotting ──────────────────────────────────────────────────────────────
    def make_figures(group_conditions, group_name=""):
        if not group_conditions: return
        colors = _palette[:len(group_conditions)]
        n_wf_rows = len(group_conditions)
        fig = plt.figure(figsize=(16, 3.2 + 2.8 * n_wf_rows))
        if group_name: fig.suptitle(group_name, fontsize=13, fontweight="bold", y=0.995)

        outer    = gridspec.GridSpec(2, 1, height_ratios=[2.8, 2.8*n_wf_rows],
                                     hspace=0.55, figure=fig)
        top_gs   = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[0], wspace=0.45)
        bot_gs   = gridspec.GridSpecFromSubplotSpec(n_wf_rows, 1,
                                                    subplot_spec=outer[1], hspace=0.55)

        keys   = ["frequency","prominence","width","ttp","rec","auc"]
        labels = ["Frequency\n(Hz)","Prominence\n(pA)","Half-width\n(ms)",
                  "Time-to-peak\n(ms)","Recovery\n(ms)","AUC\n(pA·ms)"]
        x = np.arange(len(group_conditions))
        rng = np.random.default_rng(0)

        for col_i, (key, label) in enumerate(zip(keys, labels)):
            ax = fig.add_subplot(top_gs[col_i])
            means = np.array([feature_lookup[c][key]          for c in group_conditions], dtype=float)
            sems  = np.array([feature_lookup[c][f"{key}_sem"] for c in group_conditions], dtype=float)
            ax.bar(x, means, color=colors, alpha=0.85, width=0.55, zorder=2,
                   linewidth=0.6, edgecolor="white")
            ax.errorbar(x, means, yerr=sems, fmt="none", color="black",
                        capsize=3, capthick=0.8, elinewidth=0.8, zorder=3)

            for xi, cond, color in zip(x, group_conditions, colors):
                vals = (raw_frequency.get(cond, []) if key == "frequency"
                        else [v for v in event_features[cond][key] if v is not None and not np.isnan(v)])
                if not vals: continue
                jitter = rng.normal(0, 0.07, size=len(vals))
                ax.scatter(np.full(len(vals), xi) + jitter, vals,
                           color="white", edgecolors=color, linewidths=0.6,
                           s=14, alpha=0.75, zorder=4)

            if len(group_conditions) >= 2:
                data_by_cond = []
                for cond in group_conditions:
                    d = (raw_frequency.get(cond, []) if key == "frequency"
                         else [v for v in event_features[cond][key] if v is not None and not np.isnan(v)])
                    data_by_cond.append(d)
                pairs = list(combinations(range(len(group_conditions)), 2))
                data_max = np.nanmax([np.nanmax(d) if d else 0 for d in data_by_cond])
                bbase = data_max * 1.08; bgap = data_max * 0.12 if data_max else 0.12
                for pi, (i, j) in enumerate(pairs):
                    d1, d2 = data_by_cond[i], data_by_cond[j]
                    if len(d1) < 2 or len(d2) < 2: continue
                    _, p_val = ttest_ind(d1, d2, equal_var=False)
                    stars = _pvalue_stars(p_val)
                    yb = bbase + pi * bgap; h = bgap * 0.35
                    ax.plot([i,i,j,j],[yb,yb+h,yb+h,yb], lw=0.7, color="#444444")
                    ax.text((i+j)/2, yb+h*1.1, stars, ha="center", va="bottom",
                            fontsize=6.5, color="#222222")

            ax.set_title(label, fontsize=8.5, fontweight="bold", pad=4)
            ax.set_xticks(x)
            ax.set_xticklabels(group_conditions, rotation=35, ha="right", fontsize=7.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
            ax.tick_params(length=3)

        top_axes = fig.axes[:6]
        bb_freq  = top_axes[0].get_position()
        bb_left  = top_axes[1].get_position()
        bb_right = top_axes[5].get_position()
        header_y = bb_freq.y1 + 0.055; line_y = header_y - 0.005
        for label_txt, x0, x1 in [("Per-Recording", bb_freq.x0, bb_freq.x1),
                                    ("Per-Event", bb_left.x0, bb_right.x1)]:
            fig.text((x0+x1)/2, header_y, label_txt, ha="center", va="bottom",
                     fontsize=10, fontweight="bold", transform=fig.transFigure)
            fig.add_artist(Line2D([x0,x1],[line_y,line_y], transform=fig.transFigure,
                                  color="black", linewidth=1.2, clip_on=False))

        fs_ref = list(trace_cache.values())[0][2] if trace_cache else 10000
        pre_samples = int(window_ms_pre / 1000 * fs_ref)

        for row_idx, cond in enumerate(group_conditions):
            ax = fig.add_subplot(bot_gs[row_idx])
            wfs = np.array(condition_waveforms[cond])
            n_events = len(wfs)
            if n_events == 0:
                ax.set_title(f"{cond}   (no events)", fontsize=8.5, fontweight="bold",
                             color="#222222", pad=4)
                ax.set_ylim(ylim); continue

            t = (np.arange(wfs.shape[1]) - pre_samples) / fs_ref * 1000
            color = colors[row_idx]
            for wf in wfs:
                ax.plot(t, wf, color=color, alpha=max(0.03, min(0.12, 3/n_events)),
                        lw=0.6, rasterized=True)
            mean_wf = np.mean(wfs, axis=0)
            sem_wf  = np.std(wfs, axis=0) / np.sqrt(n_events)
            ax.fill_between(t, mean_wf-sem_wf, mean_wf+sem_wf, color=color, alpha=0.18, zorder=2)
            ax.plot(t, mean_wf, color=color, lw=2.2, zorder=3)
            ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax.set_title(f"{cond}   (n = {n_events} events,  N = {condition_N[cond]} recordings)",
                         fontsize=8.5, fontweight="bold", color="#222222", pad=4)
            ax.set_ylim(ylim)
            ax.set_ylabel("Current (pA)", fontsize=8.5)
            if row_idx < n_wf_rows - 1: ax.set_xticklabels([])
            else: ax.set_xlabel("Time from left base (ms)", fontsize=8.5)
            ax.tick_params(length=3)

        fig.patch.set_facecolor("white")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show(block=False)

        # Confirmation figure
        fig2, axes2 = plt.subplots(1, len(group_conditions),
                                   figsize=(3.5*len(group_conditions), 4), sharey=False)
        if len(group_conditions) == 1: axes2 = [axes2]
        for ax, cond, color in zip(axes2, group_conditions, colors):
            ttp_vals  = np.array(event_features[cond]["ttp"],      dtype=float)
            rec_vals  = np.array(event_features[cond]["rec"],      dtype=float)
            base_vals = np.array(event_features[cond]["base_dur"], dtype=float)
            valid = ~(np.isnan(ttp_vals) | np.isnan(rec_vals) | np.isnan(base_vals))
            ttp_v = ttp_vals[valid]; rec_v = rec_vals[valid]; base_v = base_vals[valid]
            if not len(base_v):
                ax.set_title(f"{cond} (no events)", fontsize=9, fontweight="bold"); continue
            summed = ttp_v + rec_v; residual = base_v - summed
            for xi, vals_i in zip([0,1], [summed, base_v]):
                m = np.nanmean(vals_i); s = np.nanstd(vals_i, ddof=1)/np.sqrt(len(vals_i))
                ax.bar(xi, m, color=color, alpha=0.8, width=0.55,
                       edgecolor="white", linewidth=0.6, zorder=2)
                ax.errorbar(xi, m, yerr=s, fmt="none", color="black",
                            capsize=3, capthick=0.8, elinewidth=0.8, zorder=3)
                jitter = np.random.default_rng(xi).normal(0, 0.06, size=len(vals_i))
                ax.scatter(np.full(len(vals_i), xi)+jitter, vals_i,
                           color="white", edgecolors=color, linewidths=0.6,
                           s=14, alpha=0.75, zorder=4)
            jitter = np.random.default_rng(99).normal(0, 0.06, size=len(residual))
            ax.scatter(np.full(len(residual), 2)+jitter, residual,
                       color=color, s=14, alpha=0.6, linewidths=0, zorder=4)
            ax.axhline(np.nanmean(residual), color="black", lw=1.2,
                       xmin=2/3+0.02, xmax=1.0-0.02, zorder=5)
            ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.6, zorder=3,
                       xmin=2/3, xmax=1.0)
            ax.set_xticks([0,1,2])
            ax.set_xticklabels(["Onset + Offset","Base dur.","Residual\n(base − sum)"], fontsize=8)
            ax.set_xlim(-0.6, 2.6)
            ax.set_title(cond, fontsize=9, fontweight="bold", pad=4)
            ax.set_ylabel("Time (ms)", fontsize=8.5)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.tick_params(length=3)
            ylo, yhi = ax.get_ylim()
            ax.text(2, ylo+(yhi-ylo)*0.05, f"mean = {np.nanmean(residual):.3f} ms",
                    ha="center", va="bottom", fontsize=7, color="#444444")

        title_suffix = f"  —  {group_name}" if group_name else ""
        fig2.suptitle("Confirmation: TTP + Recovery vs. Base-to-Base Duration" + title_suffix,
                      fontsize=10, fontweight="bold", y=1.01)
        fig2.patch.set_facecolor("white")
        plt.tight_layout()
        plt.show(block=False)

    if split_by_halfwidth:
        make_figures(sorted(c for c in conditions if c.endswith("_narrow")), "Narrow events")
        make_figures(sorted(c for c in conditions if c.endswith("_wide")),   "Wide events")
        other = sorted(c for c in conditions if not (c.endswith("_narrow") or c.endswith("_wide")))
        if other: make_figures(other, "Unsplit events")
    else:
        make_figures(conditions)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GUI  (PyQt5)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Palette: mid-gray charcoal with indigo/purple accents ──────────────────
DARK_BG   = "#21242b"   # app background — soft charcoal
PANEL_BG  = "#2e323d"   # card / panel surface, sits above the background
BORDER    = "#3a3f4b"   # subtle card / input borders
ACCENT    = "#7c6cf0"   # indigo/purple accent (header, progress, focus)
TEXT      = "#e3e3ea"   # primary text — near-white, easy on charcoal
TEXT_DIM  = "#9aa0ad"   # secondary / muted text
BTN_GREEN = "#3b8268"   # run — muted teal-green
BTN_BLUE  = "#3c4250"   # neutral secondary (Browse)
BTN_PURP  = "#6b4fd6"   # plot results — indigo
BTN_DIS   = "#363a44"   # disabled surface
BTN_RED   = "#a33a48"   # destructive (clear)
ENTRY_BG  = "#1e2128"   # input fields — slightly darker than panels


class VClampApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voltage Clamp Analysis Suite")
        self.setMinimumSize(680, 600)

        # State
        self._analysis_done = False
        self._analysis_session = None
        self._analysis_window = None
        self._worker = None

        self._param_vars = {}
        self._res_vars = {}

        self._build_ui()
        # Re-evaluate button availability whenever the directory changes
        # (typed or chosen via Browse), since Clear/Plot depend on whether
        # event_status.npy files exist there.
        self._root_edit.textChanged.connect(lambda _=None: self._refresh_buttons())
        self._refresh_buttons()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        central.setStyleSheet(f"background:{DARK_BG};")
        self.setCentralWidget(central)

        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ─ Header ─
        hdr = QtWidgets.QWidget()
        hdr.setObjectName("header")
        hdr.setStyleSheet(
            f"#header {{ background:{ACCENT};"
            f" border-bottom:1px solid #5a4bc4; }}")
        hv = QtWidgets.QVBoxLayout(hdr)
        hv.setContentsMargins(12, 14, 12, 14)
        hv.setSpacing(3)
        title = QtWidgets.QLabel("Current Events")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color:white; font-size:19px; font-weight:bold; background:transparent;"
            " letter-spacing:0.5px;")
        sub = QtWidgets.QLabel("Voltage Clamp Analysis Suite for pClamp")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color:#e6e2ff; font-size:11px; background:transparent;")
        hv.addWidget(title); hv.addWidget(sub)
        outer.addWidget(hdr)

        # ─ Scrollable body ─
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none; background:{DARK_BG};}}")
        outer.addWidget(scroll, stretch=1)

        body = QtWidgets.QWidget()
        body.setStyleSheet(f"background:{DARK_BG};")
        scroll.setWidget(body)
        bl = QtWidgets.QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 14)
        bl.setSpacing(8)

        # ── Data Directory ──
        bl.addWidget(self._section("📁  Data Directory"))
        dir_row = QtWidgets.QHBoxLayout()
        self._root_edit = QtWidgets.QLineEdit()
        self._root_edit.setStyleSheet(
            f"background:{ENTRY_BG}; color:{TEXT}; border:1px solid {BORDER};"
            " border-radius:5px; padding:6px; font-family:'Courier New', monospace;"
            " font-size:12px;")
        dir_row.addWidget(self._root_edit, stretch=1)
        browse = QtWidgets.QPushButton("Browse…")
        browse.setCursor(Qt.PointingHandCursor)
        browse.clicked.connect(self._browse_root)
        self._style_btn(browse, BTN_BLUE, TEXT, "#4a515f", "6px 12px")
        dir_row.addWidget(browse)
        bl.addLayout(dir_row)

        # ── Detection Parameters ──
        bl.addWidget(self._section("🔬  Detection Parameters"))
        det_params = [
            ("Min amplitude (pA)",    "min_amp",                 "6"),
            ("Max amplitude (pA)",    "max_amp",                 "80"),
            ("Threshold σ",           "threshold_sigma",         "5"),
            ("Analysis start (s)",    "analysis_start_s",        "0"),
            ("Analysis end (s)",      "analysis_end_s",          "60"),
            ("View window (ms)",      "window_samples_view_ms",  "400"),
            ("Noise filter pos (pA)", "noise_filter_pos_voltage","10"),
            ("Noise filter win (ms)", "noise_filter_window_ms",  "5"),
            ("Y-lim min (pA)",        "ylim_min",                "-40"),
            ("Y-lim max (pA)",        "ylim_max",                "5"),
        ]
        bl.addWidget(self._param_grid(det_params, self._param_vars))

        # ── Results Parameters ──
        bl.addWidget(self._section("Results Parameters"))
        res_params = [
            ("Pre-event window (ms)",     "window_ms_pre",          "100"),
            ("Total window (ms)",         "total_window_ms",        "200"),
            ("Highpass cutoff (Hz)",      "highpass_cutoff",        "1"),
            ("Lowpass cutoff (Hz)",       "lowpass_cutoff",         "1000"),
            ("Half-width threshold (ms)", "halfwidth_threshold_ms", "3.0"),
        ]
        res_panel = self._param_grid(res_params, self._res_vars)
        # add checkboxes inside the same panel
        chk_row = QtWidgets.QHBoxLayout()
        self._accepted_only = QtWidgets.QCheckBox("Accepted events only")
        self._accepted_only.setChecked(True)
        self._split_halfwidth = QtWidgets.QCheckBox("Split by half-width")
        for c in (self._accepted_only, self._split_halfwidth):
            c.setStyleSheet(f"""
                QCheckBox {{ color:{TEXT}; font-size:11px; background:transparent;
                             spacing:7px; }}
                QCheckBox::indicator {{ width:15px; height:15px; border-radius:4px;
                             border:1px solid {BORDER}; background:{ENTRY_BG}; }}
                QCheckBox::indicator:checked {{ background:{ACCENT};
                             border:1px solid {ACCENT}; }}
            """)
        chk_row.addWidget(self._accepted_only)
        chk_row.addSpacing(20)
        chk_row.addWidget(self._split_halfwidth)
        chk_row.addStretch(1)
        res_panel.layout().addLayout(chk_row)
        bl.addWidget(res_panel)

        # ── Action buttons ──
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 8, 0, 4)
        self._btn_analysis = QtWidgets.QPushButton("▶  Run Analysis")
        self._btn_analysis.setCursor(Qt.PointingHandCursor)
        self._btn_analysis.clicked.connect(self._run_analysis)
        self._style_btn(self._btn_analysis, BTN_GREEN, "white", "#4a9a7e",
                        "10px 22px", bold=True, size=14)
        self._btn_analysis.setEnabled(True)
        btn_row.addWidget(self._btn_analysis)
        btn_row.addSpacing(12)

        self._btn_clear = QtWidgets.QPushButton("🗑  Clear Analysis")
        self._btn_clear.clicked.connect(self._run_clear)
        self._style_btn(self._btn_clear, BTN_DIS, TEXT_DIM, "#c0485a",
                        "10px 22px", bold=True, size=14)
        self._btn_clear.setEnabled(False)
        btn_row.addWidget(self._btn_clear)
        btn_row.addSpacing(12)

        self._btn_results = QtWidgets.QPushButton("📈  Plot Results")
        self._btn_results.clicked.connect(self._run_results)
        self._style_btn(self._btn_results, BTN_DIS, TEXT_DIM, BTN_PURP,
                        "10px 22px", bold=True, size=14)
        self._btn_results.setEnabled(False)
        btn_row.addWidget(self._btn_results)
        btn_row.addStretch(1)
        bl.addLayout(btn_row)
        bl.addStretch(1)

        # ── Status bar ──
        sb = QtWidgets.QWidget()
        sb.setObjectName("statusBar")
        sb.setStyleSheet(
            f"#statusBar {{ background:{ENTRY_BG}; border-top:1px solid {BORDER}; }}")
        sbl = QtWidgets.QHBoxLayout(sb)
        sbl.setContentsMargins(12, 6, 12, 6)
        self._status_lbl = QtWidgets.QLabel(
            "Ready. Select a data directory and run analysis.")
        self._status_lbl.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:11px; background:transparent;")
        sbl.addWidget(self._status_lbl, stretch=1)
        self._progress = QtWidgets.QProgressBar()
        self._progress.setFixedWidth(180)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background:{PANEL_BG}; border:1px solid {BORDER};
                            border-radius:5px; height:14px; }}
            QProgressBar::chunk {{ background:{ACCENT}; border-radius:4px; }}
        """)
        sbl.addWidget(self._progress)
        outer.addWidget(sb)

    # ── UI helper builders ─────────────────────────────────────────────────────

    def _section(self, text):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 6, 0, 2)
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(
            f"color:{TEXT}; font-size:12px; font-weight:bold; background:transparent;")
        h.addWidget(lbl)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet(f"background:{BORDER}; max-height:1px; border:none;")
        h.addWidget(line, stretch=1)
        return w

    def _param_grid(self, params, store):
        panel = QtWidgets.QWidget()
        panel.setObjectName("paramCard")
        # Scope to objectName so the border/background doesn't cascade onto
        # child labels and inputs.
        panel.setStyleSheet(
            f"#paramCard {{ background:{PANEL_BG}; border:1px solid {BORDER};"
            f" border-radius:8px; }}")
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(14, 12, 14, 12)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for i, (label, key, default) in enumerate(params):
            row, col = divmod(i, 3)
            col_off = col * 2
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px; background:transparent;")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(lbl, row, col_off)
            edit = QtWidgets.QLineEdit(default)
            edit.setFixedWidth(70)
            edit.setValidator(QtGui.QDoubleValidator(self))
            edit.setStyleSheet(
                f"QLineEdit {{ background:{ENTRY_BG}; color:{TEXT};"
                f" border:1px solid {BORDER}; border-radius:5px; padding:4px;"
                f" font-family:'Courier New', monospace; font-size:11px; }}"
                f" QLineEdit:focus {{ border:1px solid {ACCENT}; }}")
            grid.addWidget(edit, row, col_off + 1)
            store[key] = edit
        for c in (1, 3, 5):
            grid.setColumnStretch(c, 1)
        vbox.addLayout(grid)
        return panel

    @staticmethod
    def _style_btn(btn, bg, fg, hover, padding, bold=False, size=10):
        weight = "bold" if bold else "normal"
        btn.setStyleSheet(f"""
            QPushButton {{
                background:{bg}; color:{fg}; border:none; border-radius:6px;
                padding:{padding}; font-size:{size}px; font-weight:{weight};
            }}
            QPushButton:hover:enabled {{ background:{hover}; color:white; }}
            QPushButton:disabled {{ background:{BTN_DIS}; color:{TEXT_DIM}; }}
        """)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _browse_root(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select root data directory")
        if d:
            self._root_edit.setText(d)

    def _get_analysis_params(self):
        v = self._param_vars
        f = lambda k: float(v[k].text())
        return {
            "root":                     self._root_edit.text(),
            "min_amp":                  f("min_amp"),
            "max_amp":                  f("max_amp"),
            "threshold_sigma":          f("threshold_sigma"),
            "analysis_start_s":         f("analysis_start_s"),
            "analysis_end_s":           f("analysis_end_s"),
            "window_samples_view_ms":   f("window_samples_view_ms"),
            "noise_filter_pos_voltage": f("noise_filter_pos_voltage"),
            "noise_filter_window_ms":   f("noise_filter_window_ms"),
            "ylim":                     (f("ylim_min"), f("ylim_max")),
        }

    def _get_results_params(self):
        v = self._res_vars
        f = lambda k: float(v[k].text())
        ap = self._get_analysis_params()
        return {
            "root":                   ap["root"],
            "analysis_start_s":       ap["analysis_start_s"],
            "analysis_end_s":         ap["analysis_end_s"],
            "ylim":                   ap["ylim"],
            "window_ms_pre":          f("window_ms_pre"),
            "total_window_ms":        f("total_window_ms"),
            "highpass_cutoff":        f("highpass_cutoff"),
            "lowpass_cutoff":         f("lowpass_cutoff"),
            "halfwidth_threshold_ms": f("halfwidth_threshold_ms"),
            "accepted_only":          self._accepted_only.isChecked(),
            "split_by_halfwidth":     self._split_halfwidth.isChecked(),
        }

    def _has_saved_analysis(self):
        """True if at least one event_status.npy exists under the chosen root."""
        root = self._root_edit.text().strip()
        if not root or not os.path.isdir(root):
            return False
        return any(
            "event_status.npy" in files
            for _, _, files in os.walk(root)
        )

    def _refresh_buttons(self):
        """Set button enabled/disabled state.

        Run Analysis is always available (with a valid dir). Clear Analysis and
        Plot Results are only available when one or more event_status.npy files
        exist under the selected directory.
        """
        # Run Analysis — always enabled (validity is checked on click).
        self._btn_analysis.setEnabled(True)
        self._btn_analysis.setCursor(Qt.PointingHandCursor)

        has_saved = self._has_saved_analysis()

        if has_saved:
            self._btn_results.setEnabled(True)
            self._btn_results.setCursor(Qt.PointingHandCursor)
            self._style_btn(self._btn_results, BTN_PURP, "white", "#8166e0",
                            "10px 22px", bold=True, size=14)
            self._btn_clear.setEnabled(True)
            self._btn_clear.setCursor(Qt.PointingHandCursor)
            self._style_btn(self._btn_clear, BTN_RED, "white", "#c0485a",
                            "10px 22px", bold=True, size=14)
        else:
            self._btn_results.setEnabled(False)
            self._btn_results.setCursor(Qt.ArrowCursor)
            self._style_btn(self._btn_results, BTN_DIS, TEXT_DIM, BTN_PURP,
                            "10px 22px", bold=True, size=14)
            self._btn_clear.setEnabled(False)
            self._btn_clear.setCursor(Qt.ArrowCursor)
            self._style_btn(self._btn_clear, BTN_DIS, TEXT_DIM, "#c0485a",
                            "10px 22px", bold=True, size=14)

    def _set_status(self, msg, progress=None):
        self._status_lbl.setText(msg)
        if progress is not None:
            self._progress.setValue(int(progress))

    # ── Analysis flow ─────────────────────────────────────────────────────────

    def _run_analysis(self):
        root = self._root_edit.text().strip()
        if not root or not os.path.isdir(root):
            QtWidgets.QMessageBox.critical(
                self, "No directory", "Please select a valid data directory.")
            return

        try:
            params = self._get_analysis_params()
        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Invalid parameter", str(e))
            return

        self._btn_analysis.setEnabled(False)
        self._set_status("Scanning for recordings…", 0)
        QtWidgets.QApplication.processEvents()

        # IMPORTANT: detection calls scipy.filtfilt -> numpy.linalg.solve (LAPACK).
        # On macOS with a NumPy built against Apple's Accelerate framework, LAPACK
        # called from a worker thread crashes with a bus error. Run detection on
        # the main thread instead. The UI updates between recordings via
        # processEvents() so the progress bar still moves.
        try:
            session = AnalysisSession(params)
            n_rec = session.discover_recordings()
            if n_rec == 0:
                self._on_no_recordings()
                return

            self._set_status(f"Found {n_rec} recordings. Running detection…", 5)
            QtWidgets.QApplication.processEvents()

            def progress_cb(i, total, fname):
                pct = int(5 + 90 * i / max(total, 1))
                msg = (f"Processing {fname} ({i}/{total})"
                       if fname != "done" else "Detection complete.")
                self._set_status(msg, pct)
                QtWidgets.QApplication.processEvents()

            session.process_all(progress_cb=progress_cb)
            session.init_bases()
            for e in session.events:
                if e.get("peak_t") is None:
                    session.compute_peak_between_bases(e)

            if len(session.events) == 0:
                self._on_no_events()
                return

            self._on_analysis_ready(session)
        except Exception as ex:
            self._on_analysis_error(str(ex))

    def _on_no_recordings(self):
        QtWidgets.QMessageBox.warning(
            self, "No recordings",
            "No .abf files found in the selected directory.")
        self._btn_analysis.setEnabled(True)
        self._set_status("No recordings found.", 0)

    def _on_no_events(self):
        QtWidgets.QMessageBox.warning(
            self, "No events",
            "Detection found no events. Check your parameters.")
        self._btn_analysis.setEnabled(True)
        self._set_status("No events detected.", 0)

    def _on_analysis_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Analysis error", msg)
        self._btn_analysis.setEnabled(True)
        self._set_status("Error during analysis.", 0)

    def _on_analysis_ready(self, session):
        self._analysis_session = session
        n_ev = len(session.events)
        self._set_status(
            f"Opening analysis window — {n_ev} events across "
            f"{len(session.recordings)} recordings.", 100)

        self._analysis_window = AnalysisWindow(session, parent=self)
        self._analysis_window.closed.connect(self._on_analysis_closed)
        self._analysis_window.show()
        self._set_status(
            f"Analysis window open — {n_ev} events. "
            "Use ← → to navigate, ↑ ↓ to accept/reject.", 100)

    def _on_analysis_closed(self, n_saved):
        if n_saved is not None:
            self._analysis_done = True
            self._refresh_buttons()
            self._set_status(
                f"Analysis saved ({n_saved} recording(s)). You can now plot results.", 100)
            QtWidgets.QMessageBox.information(
                self, "Saved",
                f"Analysis saved for {n_saved} recording(s).\n"
                "You can now click 'Plot Results'.")
        else:
            self._set_status("Analysis window closed (not saved).", 0)
        self._btn_analysis.setEnabled(True)
        self._analysis_window = None

    # ── Clear flow ────────────────────────────────────────────────────────────

    def _run_clear(self):
        root = self._root_edit.text().strip()
        if not root or not os.path.isdir(root):
            QtWidgets.QMessageBox.critical(
                self, "No directory", "Please select a valid data directory.")
            return

        files = [
            os.path.join(subdir, "event_status.npy")
            for subdir, _, fnames in os.walk(root)
            if "event_status.npy" in fnames
        ]
        if not files:
            QtWidgets.QMessageBox.information(
                self, "Nothing to clear",
                "No event_status.npy files were found under this directory.")
            self._refresh_buttons()
            return

        # Guarded confirmation: default button is "No", and the user must
        # explicitly confirm this destructive, irreversible action.
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Clear analysis?")
        box.setText(
            f"This will permanently delete {len(files)} saved analysis "
            f"file(s) (event_status.npy) under:\n\n{root}\n\n"
            "This cannot be undone. Detection will have to be re-run.")
        box.setInformativeText("Are you sure you want to delete them?")
        yes = box.addButton("Delete files", QtWidgets.QMessageBox.DestructiveRole)
        no  = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(no)
        box.exec_()
        if box.clickedButton() is not yes:
            self._set_status("Clear cancelled.", 0)
            return

        deleted, failed = 0, []
        for fp in files:
            try:
                os.remove(fp)
                deleted += 1
            except Exception as ex:
                failed.append(f"{fp}: {ex}")

        # A cleared analysis means results are no longer available.
        self._analysis_done = False
        self._refresh_buttons()

        if failed:
            QtWidgets.QMessageBox.warning(
                self, "Partly cleared",
                f"Deleted {deleted} file(s); {len(failed)} could not be removed:\n\n"
                + "\n".join(failed))
            self._set_status(
                f"Cleared {deleted} file(s); {len(failed)} failed.", 0)
        else:
            self._set_status(f"Cleared {deleted} analysis file(s).", 0)
            QtWidgets.QMessageBox.information(
                self, "Cleared",
                f"Deleted {deleted} analysis file(s). You can run a fresh analysis.")

    # ── Results flow ──────────────────────────────────────────────────────────

    def _run_results(self):
        root = self._root_edit.text().strip()
        if not root or not os.path.isdir(root):
            QtWidgets.QMessageBox.critical(
                self, "No directory", "Please select a valid data directory.")
            return

        found = any(
            "event_status.npy" in files
            for _, _, files in os.walk(root)
        )
        if not found:
            QtWidgets.QMessageBox.warning(
                self, "No saved analysis",
                "No event_status.npy files found.\n"
                "Please run and save the analysis first.")
            return

        try:
            params = self._get_results_params()
        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Invalid parameter", str(e))
            return

        self._btn_results.setEnabled(False)
        self._set_status("Loading events and computing features…", 20)

        # IMPORTANT: matplotlib figure creation and plt.show() must run on the
        # main (GUI) thread. On macOS, doing this from a QThread causes a hard
        # bus error / crash in Cocoa. So we run results synchronously here.
        # Let the status text repaint before the (briefly) blocking work.
        QtWidgets.QApplication.processEvents()
        try:
            run_results(params)
            self._set_status("Results plotted successfully.", 100)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Results error", str(ex))
            self._set_status("Error during results computation.", 0)
        finally:
            self._refresh_buttons()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = VClampApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()