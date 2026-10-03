from __future__ import annotations

"""Load and render logo/logo.png as animated Rich Text frames for Textual splash.

Implements the spinning ring logo with blue-white gradient.
Outer ring is fixed, inner content rotates via PIL Image.rotate.
Gradient uses fixed screen-coordinate bounds (no clock-hand sweep).
"""

import math
from pathlib import Path
from rich.text import Text

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from app.core.paths import find_project_root

SOURCE_IMAGE = find_project_root() / "logo" / "logo.png"

LINES = 30
FRAME_STEP = 15
FPS_INTERVAL = 0.08
CLOCKWISE = True

WHITE = (255, 255, 255)
CAS_BLUE = (23, 73, 148)


def has_pil() -> bool:
    return HAS_PIL


def logo_exists() -> bool:
    return SOURCE_IMAGE.exists()


def _fallback_frame() -> Text:
    return Text("", style="dim")


def ansi_rgb(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def real_rotation_angle(deg: int, clockwise: bool = True) -> int:
    """PIL rotates counter-clockwise for positive angles; clockwise needs negation."""
    return -deg if clockwise else deg


def crop_white_border(img: Image.Image, threshold: int = 245, padding: int = 6) -> Image.Image:
    """Auto-crop white borders with a small padding."""
    gray = img.convert("L")
    w, h = gray.size
    pix = gray.load()

    xs = []
    ys = []
    for y in range(h):
        for x in range(w):
            if pix[x, y] < threshold:
                xs.append(x)
                ys.append(y)

    if not xs:
        return img

    left = max(0, min(xs) - padding)
    right = min(w, max(xs) + padding + 1)
    top = max(0, min(ys) - padding)
    bottom = min(h, max(ys) + padding + 1)

    return img.crop((left, top, right, bottom))


def load_source_image(image_path: Path | None = None) -> Image.Image:
    """Load and composite onto white background, then crop white borders."""
    path = image_path or SOURCE_IMAGE
    img = Image.open(path).convert("RGBA")
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white_bg, img).convert("RGB")
    img = crop_white_border(img)
    return img


def make_clean_ring_frame(
    src_gray: Image.Image,
    angle: float,
) -> Image.Image:
    """Generate one frame: fixed outer ring + rotated inner content.

    The outer ring is drawn fresh every frame (always circular, never rotated).
    Only the center content (within preserve_r) is rotated.
    """
    w, h = src_gray.size
    cx = w / 2.0
    cy = h / 2.0
    radius = min(w, h) / 2.0 - 4

    preserve_r = 0.67 * radius
    ring_inner = 0.74 * radius
    ring_outer = 0.97 * radius

    frame = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(frame)

    draw.ellipse(
        (cx - ring_outer, cy - ring_outer, cx + ring_outer, cy + ring_outer),
        fill=0,
    )
    draw.ellipse(
        (cx - ring_inner, cy - ring_inner, cx + ring_inner, cy + ring_inner),
        fill=255,
    )

    rotated = src_gray.rotate(
        angle,
        resample=Image.BICUBIC,
        center=(cx, cy),
        fillcolor=255,
    )

    frame_pix = frame.load()
    rot_pix = rotated.load()

    for y in range(h):
        for x in range(w):
            rr = math.hypot(x + 0.5 - cx, y + 0.5 - cy)
            if rr <= preserve_r:
                frame_pix[x, y] = rot_pix[x, y]

    return frame


def render_outline_fill(
    mask_img: Image.Image,
    lines: int = LINES,
    max_width: int | None = None,
) -> list[str]:
    """Render outline_fill style ASCII art from a grayscale image.

    Uses block characters (█▓▒░) for fill and directional characters (│─╱╲)
    for outlines based on Sobel gradient direction.
    Returns exactly `lines` rows.
    """
    src_w, src_h = mask_img.size
    target_h = lines
    target_w = max(1, int((src_w / src_h) * target_h / 0.5))

    if max_width is not None and target_w > max_width:
        target_w = max_width

    gray = mask_img.resize((target_w, target_h), Image.LANCZOS)
    pix = gray.load()

    values = [
        [(255 - pix[x, y]) / 255.0 for x in range(target_w)]
        for y in range(target_h)
    ]

    def val(x: int, y: int) -> float:
        if 0 <= x < target_w and 0 <= y < target_h:
            return values[y][x]
        return 0.0

    result: list[str] = []

    for y in range(target_h):
        row: list[str] = []
        for x in range(target_w):
            c = values[y][x]

            if c < 0.10:
                row.append(" ")
                continue

            gx = val(x + 1, y) - val(x - 1, y)
            gy = val(x, y + 1) - val(x, y - 1)
            edge = abs(gx) + abs(gy)

            if c > 0.72:
                row.append("█")
            elif c > 0.52:
                row.append("▓")
            elif c > 0.32:
                if edge > 0.20:
                    if abs(gx) > abs(gy) * 1.3:
                        row.append("│")
                    elif abs(gy) > abs(gx) * 1.3:
                        row.append("─")
                    else:
                        row.append("╱" if gx * gy < 0 else "╲")
                else:
                    row.append("▒")
            else:
                if edge > 0.18:
                    if abs(gx) > abs(gy) * 1.3:
                        row.append("│")
                    elif abs(gy) > abs(gx) * 1.3:
                        row.append("─")
                    else:
                        row.append("╱" if gx * gy < 0 else "╲")
                else:
                    row.append("░")

        result.append("".join(row).rstrip())

    return result


def colorize_blue_white_gradient(
    ascii_lines: list[str],
    width: int | None = None,
    height: int | None = None,
) -> Text:
    """Apply blue-white gradient using FIXED screen-coordinate bounds.

    Gradient direction: top-left white → bottom-right blue.
    t = 0.55 * nx + 0.45 * ny (fixed diagonal, never shifts per-frame).
    """
    h = height or len(ascii_lines)
    w = width or max((len(line) for line in ascii_lines), default=1)

    text = Text()
    for y, line in enumerate(ascii_lines):
        padded = line.ljust(w)
        for x, ch in enumerate(padded):
            if ch == " ":
                text.append(" ")
                continue
            nx = x / max(1, w - 1)
            ny = y / max(1, h - 1)
            t = 0.55 * nx + 0.45 * ny
            r = int(WHITE[0] + (CAS_BLUE[0] - WHITE[0]) * t)
            g = int(WHITE[1] + (CAS_BLUE[1] - WHITE[1]) * t)
            b = int(WHITE[2] + (CAS_BLUE[2] - WHITE[2]) * t)
            color = ansi_rgb(r, g, b)
            text.append(ch, style=color)
        if y < len(ascii_lines) - 1:
            text.append("\n")
    return text


def build_frames(
    image_path: Path | None = None,
    lines: int = LINES,
    frame_step: int = FRAME_STEP,
    clockwise: bool = CLOCKWISE,
    max_width: int | None = None,
) -> list[Text]:
    """Build all rotation frames of the logo as Rich Text objects.

    Each frame:
    1. Rotates inner content via PIL (clockwise when CLOCKWISE=True).
    2. Draws fixed outer ring.
    3. Renders to ASCII with outline_fill characters.
    4. Applies FIXED-BOUNDS gradient (no clock-hand sweep).
    Result: exactly `lines` rows per frame.
    """
    if not HAS_PIL:
        return [_fallback_frame()]

    path = image_path or SOURCE_IMAGE
    if not path.exists():
        return [_fallback_frame()]

    try:
        img = load_source_image(path)
        src_gray = img.convert("L")
    except Exception:
        return [_fallback_frame()]

    frames: list[Text] = []
    for deg in range(0, 360, frame_step):
        angle = real_rotation_angle(deg, clockwise)
        frame_img = make_clean_ring_frame(src_gray, angle)
        ascii_lines = render_outline_fill(frame_img, lines=lines, max_width=max_width)

        while len(ascii_lines) < lines:
            ascii_lines.append("")
        ascii_lines = ascii_lines[:lines]

        fixed_w = max(len(line) for line in ascii_lines) if ascii_lines else 1
        frame = colorize_blue_white_gradient(ascii_lines, width=fixed_w, height=lines)
        frames.append(frame)

    return frames


def load_logo_text(lines: int = LINES) -> Text:
    """Load a single static logo frame (first frame, no rotation)."""
    frames = build_frames(lines=lines)
    return frames[0] if frames else _fallback_frame()


def load_logo_frames(
    image_path: Path | None = None,
    lines: int = LINES,
    frame_step: int = FRAME_STEP,
    clockwise: bool = CLOCKWISE,
    max_width: int | None = None,
) -> list[Text]:
    """Load all animated logo frames."""
    return build_frames(
        image_path=image_path,
        lines=lines,
        frame_step=frame_step,
        clockwise=clockwise,
        max_width=max_width,
    )


def _pixel_to_char(gray: int, alpha: int = 255) -> str:
    if alpha < 128:
        return " "
    darkness = 255 - gray
    CHARS = " .:-=+*#%@"
    idx = int(darkness / 256 * len(CHARS))
    return CHARS[min(idx, len(CHARS) - 1)]