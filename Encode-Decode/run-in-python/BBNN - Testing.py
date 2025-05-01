#!/usr/bin/env python3
"""
gui_matlab_states_creative.py

A modern, creative GUI for running MATLAB battery-state tests,
exporting figures, and previewing them in-app, using CustomTkinter.

© Created by Abdullah Ahmed
License: MIT
"""

import os
import glob
import threading
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk  # PIL.Image.LANCZOS available here

# ─── Paths & States ─────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode"
OUT_DIR  = r"C:\Users\msi-pc\Desktop\Li-ion paper 2\Model\Encode-Decode\run-in-python"
STATES   = ['I','II','III','IV','V','VI','VII','VIII','IX']

# ─── Tooltip Helper ─────────────────────────────────────────────────────────
class CreateToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
    def show(self, e=None):
        if self.tip or not self.text: return
        x, y = e.x_root + 10, e.y_root + 10
        self.tip = ctk.CTkToplevel(self.widget)
        self.tip.overrideredirect(True)
        self.tip.geometry(f"+{x}+{y}")
        label = ctk.CTkLabel(
            self.tip,
            text=self.text,
            text_color="#333",
            fg_color="#f9f9f9",
            corner_radius=5,
            padx=5, pady=2
        )
        label.pack()
    def hide(self, e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

# ─── MATLAB Runner (thread target) ──────────────────────────────────────────
def run_matlab(state, on_complete):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    ws    = os.path.join(BASE_DIR, f"ANN_State_{state}", "Workspace", "workspace.mat")
    mfile = os.path.join(BASE_DIR, f"ANN_State_{state}", "Testing.m")
    eng.load(ws, nargout=0)
    eng.run(mfile, nargout=0)
    save_cmd = f"""
    hs = findall(0,'Type','figure');
    for k=1:length(hs)
      saveas(hs(k), fullfile('{OUT_DIR}', sprintf('fig_{state}_%d.png', k)));
    end
    """
    eng.eval(save_cmd, nargout=0)
    eng.quit()
    on_complete()

# ─── Main Application ───────────────────────────────────────────────────────
class BatteryTesterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        # ---- Appearance ----
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("green")
        self.title("⚡ Battery State Tester ⚡")
        self.geometry("900x600")
        self.resizable(False, False)

        # ---- Menu Bar ----
        self._build_menu()

        # ---- Layout ----
        self._build_sidebar()
        self._build_main_area()

    def _build_menu(self):
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.destroy)
        menu_bar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menu_bar)

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=180, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        ctk.CTkLabel(
            self.sidebar,
            text="States",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=10)
        self.state_var = tk.StringVar(value=STATES[0])
        for st in STATES:
            btn = ctk.CTkButton(
                self.sidebar,
                text=st,
                width=120,
                command=lambda s=st: self._select_state(s)
            )
            btn.pack(pady=4)

    def _build_main_area(self):
        self.main = ctk.CTkFrame(self)
        self.main.pack(side="right", expand=True, fill="both", padx=20, pady=20)

        # Header + controls
        hdr = ctk.CTkFrame(self.main, fg_color=None)
        hdr.pack(fill="x")
        self.header = ctk.CTkLabel(
            hdr,
            text="🔋 Battery State Testing",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.header.pack(side="left")

        self.mode_switch = ctk.CTkSwitch(
            hdr,
            text="Dark Mode",
            command=self._toggle_mode
        )
        self.mode_switch.pack(side="right", padx=10)
        CreateToolTip(self.mode_switch, "Toggle light/dark appearance")

        self.btn_run = ctk.CTkButton(
            hdr,
            text="▶ Run",
            fg_color="#1ABC9C",
            command=self._on_run
        )
        self.btn_run.pack(side="right", padx=5)
        CreateToolTip(self.btn_run, "Run the MATLAB test for the selected state")

        self.btn_exit = ctk.CTkButton(
            hdr,
            text="✖ Exit",
            fg_color="#E74C3C",
            command=self.destroy
        )
        self.btn_exit.pack(side="right", padx=5)
        CreateToolTip(self.btn_exit, "Close this application")

        # Progress & status
        self.progress = ctk.CTkProgressBar(self.main, mode="indeterminate")
        self.progress.pack(fill="x", pady=(10,5))
        self.status = ctk.CTkLabel(
            self.main,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status.pack(anchor="w")

        # Preview pane
        self.preview_frame = ctk.CTkScrollableFrame(
            self.main,
            label_text="Figure Previews"
        )
        self.preview_frame.pack(expand=True, fill="both", pady=10)

    def _select_state(self, state):
        self.state_var.set(state)
        self.header.configure(text=f"🔋 Testing State {state}")

    def _toggle_mode(self):
        mode = "Dark" if self.mode_switch.get() else "Light"
        ctk.set_appearance_mode(mode)

    def _on_run(self):
        state = self.state_var.get()
        self.btn_run.configure(state="disabled")
        self.status.configure(text=f"Running State {state}...", text_color="#2980B9")
        self.progress.start()

        # clear old previews
        for w in self.preview_frame.winfo_children():
            w.destroy()

        # run MATLAB in background
        threading.Thread(
            target=run_matlab,
            args=(state, self._on_complete),
            daemon=True
        ).start()

    def _on_complete(self):
        self.progress.stop()
        state = self.state_var.get()
        pngs = sorted(glob.glob(os.path.join(OUT_DIR, f"fig_{state}_*.png")))
        if not pngs:
            self.status.configure(text="No figures produced", text_color="#E67E22")
            messagebox.showinfo("No Figures", "No MATLAB figures were produced.")
        else:
            self.status.configure(text="Displaying figures", text_color="#27AE60")
            for img_path in pngs:
                # use LANCZOS instead of ANTIALIAS
                img = Image.open(img_path).resize((200,125), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                lbl = ctk.CTkLabel(
                    self.preview_frame,
                    image=photo,
                    text=os.path.basename(img_path),
                    compound="top"
                )
                lbl.image = photo
                lbl.pack(side="left", padx=10, pady=10)

        self.btn_run.configure(state="normal")

    def _show_about(self):
        messagebox.showinfo(
            "About Battery State Tester",
            "Battery State Tester\n\n"
            "A CustomTkinter-based GUI for running MATLAB battery-state tests.\n\n"
            "© Created by Abdullah Ahmed\n"
            "License: MIT"
        )

if __name__ == "__main__":
    app = BatteryTesterApp()
    app.mainloop()
