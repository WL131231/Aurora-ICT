"""Aurora-ICT 흰색 다이아몬드 아이콘 생성 (PNG / ICO / ICNS).

사이트(/ict/) 로고와 동일 도형:
    외곽 다이아몬드 (stroke) + 내부 다이아몬드 (fill, opacity 0.6)

색 — 흰색 톤 (Aurora 본체는 어두운 다이아몬드, ICT 는 흰색으로 차별화).

실행:
    python scripts/make_icon.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

# viewBox 28×28 좌표 (사이트 SVG 동일)
OUTER = [(14, 2), (26, 14), (14, 26), (2, 14)]
INNER = [(14, 8), (20, 14), (14, 20), (8, 14)]
STROKE_W = 1.5  # SVG stroke-width


def _render(size: int) -> Image.Image:
    """단일 사이즈 다이아몬드 렌더링."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size / 28
    outer = [(x * s, y * s) for x, y in OUTER]
    inner = [(x * s, y * s) for x, y in INNER]
    # outline: max(1, round(stroke * scale))
    stroke = max(1, round(STROKE_W * s))
    draw.polygon(outer, outline=(255, 255, 255, 255), width=stroke)
    draw.polygon(inner, fill=(255, 255, 255, 153))  # opacity 0.6
    return img


def main() -> None:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    imgs = {sz: _render(sz) for sz in sizes}

    # 256 PNG (preview / Linux)
    imgs[256].save(ASSETS / "aurora.png")
    # 64 PNG (런처용 작은 버전)
    imgs[64].save(ASSETS / "aurora-launcher.png")

    # Windows ICO — 다중 사이즈 한 파일
    imgs[256].save(
        ASSETS / "aurora.ico",
        format="ICO",
        sizes=[(sz, sz) for sz in sizes],
    )
    imgs[256].save(
        ASSETS / "aurora-launcher.ico",
        format="ICO",
        sizes=[(sz, sz) for sz in sizes],
    )

    # macOS ICNS — Pillow ICNS plugin 은 256/512/1024 지원
    icns_sizes = [16, 32, 64, 128, 256, 512]
    icns_imgs = [imgs[sz] if sz in imgs else _render(sz) for sz in icns_sizes]
    icns_imgs[-1].save(  # 512 base
        ASSETS / "aurora.icns",
        format="ICNS",
        append_images=icns_imgs[:-1],
    )
    icns_imgs[-1].save(
        ASSETS / "aurora-launcher.icns",
        format="ICNS",
        append_images=icns_imgs[:-1],
    )

    for f in ASSETS.glob("aurora*"):
        if f.suffix in {".png", ".ico", ".icns"}:
            print(f"  {f.name:30s} {f.stat().st_size:>8d} bytes")


if __name__ == "__main__":
    main()
