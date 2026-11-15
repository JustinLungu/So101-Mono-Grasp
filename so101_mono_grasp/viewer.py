"""Tiny live image window with keyboard and mouse input, built on tkinter.

lerobot installs opencv-python-headless, so cv2.imshow / setMouseCallback don't exist. tkinter does.
Use it as a polling loop:

    v = Viewer("title")
    while v.alive:
        v.show(frame_rgb, "status text")
        for key in v.keys():      # tkinter keysyms: "space", "Return", "q", "u", "Escape", ...
            ...
        for u, v_ in v.clicks():  # in IMAGE pixel coordinates (display scaling undone)
            ...
"""

from __future__ import annotations

import tkinter as tk
from collections import deque
from tkinter import simpledialog

import numpy as np
from PIL import Image, ImageTk


class Viewer:
    def __init__(self, title: str, max_width: int = 1280):
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root, cursor="crosshair")
        self.label.pack()
        self.status = tk.Label(self.root, anchor="w", justify="left", font=("monospace", 10))
        self.status.pack(fill="x")
        self.max_width = max_width
        self.scale = 1.0
        self.alive = True
        self._keys: deque[str] = deque()
        self._clicks: deque[tuple[float, float]] = deque()
        self._photo = None
        self.root.bind("<Key>", lambda e: self._keys.append(e.keysym))
        self.label.bind("<Button-1>", lambda e: self._clicks.append((e.x / self.scale, e.y / self.scale)))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def show(self, frame_rgb: np.ndarray, status: str = "") -> None:
        if not self.alive:
            return
        img = Image.fromarray(frame_rgb)
        self.scale = min(1.0, self.max_width / img.width)
        if self.scale < 1.0:
            img = img.resize((round(img.width * self.scale), round(img.height * self.scale)))
        self._photo = ImageTk.PhotoImage(img)
        self.label.configure(image=self._photo)
        self.status.configure(text=status)
        self.pump()

    def pump(self) -> None:
        if self.alive:
            try:
                self.root.update()
            except tk.TclError:
                self.alive = False

    def keys(self) -> list[str]:
        out = list(self._keys)
        self._keys.clear()
        return out

    def clicks(self) -> list[tuple[float, float]]:
        out = list(self._clicks)
        self._clicks.clear()
        return out

    def ask(self, title: str, prompt: str) -> str | None:
        """Modal text box. None if cancelled."""
        return simpledialog.askstring(title, prompt, parent=self.root)

    def close(self) -> None:
        if self.alive:
            self.alive = False
            self.root.destroy()
