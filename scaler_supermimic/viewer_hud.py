"""Readable pixel-sized HUD panels inside the native MuJoCo window."""
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class LargeHUD:
    def __init__(self, scale=200):
        self.size = round(12 * scale / 100)
        bundled = Path(__file__).resolve().parents[1] / "assets/fonts/DejaVuSans.ttf"
        system = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        path = bundled if bundled.exists() else system
        self.font = ImageFont.truetype(str(path), self.size)
        self.last_update = 0

    def panel(self, title, labels, values, width):
        lines = [(title, (100, 220, 255))]
        for key, value in zip(labels.splitlines(), values.splitlines()):
            text = f"{key}: {value}" if key else value
            line = ""
            for char in text:
                if self.font.getlength(line + char) > width - 24 and line:
                    lines.append((line, (245, 245, 245)))
                    line = ""
                line += char
            lines.append((line, (255, 150, 110) if "FAILED" in text else (245, 245, 245)))
        step = self.size + 6
        image = Image.new("RGB", (width, 20 + len(lines)*step), (20, 25, 32))
        draw = ImageDraw.Draw(image)
        for i, (line, color) in enumerate(lines):
            draw.text((12, 8 + i*step), line, font=self.font, fill=color)
        return np.asarray(image)

    def update(self, handle, left, right):
        now = time.monotonic()
        if now-self.last_update < .1:
            return
        viewport = handle.viewport
        if viewport is None or viewport.width < 100 or viewport.height < 100:
            return
        self.last_update = now
        width = min(round(self.size*23), max(200, viewport.width-24))
        l = self.panel("SuperMimic / UFO LAFAN", *left, width)
        r = self.panel("LAFAN source", *right, min(round(self.size*16), width))
        x, top = viewport.left + 5, viewport.bottom + viewport.height - 5
        images = [(mujoco.MjrRect(x, max(viewport.bottom, top-l.shape[0]), l.shape[1], l.shape[0]), l)]
        if viewport.width > width+r.shape[1]+32:
            rx, ry = viewport.left+viewport.width-r.shape[1]-8, top-r.shape[0]
        else:
            rx, ry = x, max(viewport.bottom, top-l.shape[0]-r.shape[0]-8)
        images.append((mujoco.MjrRect(rx, ry, r.shape[1], r.shape[0]), r))
        # The native viewer renders user texts before user images. drawPixels
        # inherits the scene's depth test, whereas mjr_overlay initializes 2D
        # rendering (including disabling depth testing) on the render thread.
        # Keep an empty native overlay every frame, underneath the left panel,
        # so our pixel-sized fonts cannot be occluded by any scene geometry.
        handle.set_texts([(mujoco.mjtFont.mjFONT_NORMAL,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, "", "")])
        handle.set_images(images)
