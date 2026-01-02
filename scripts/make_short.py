#!/usr/bin/env python3
import argparse
import os
import random
import textwrap
from pathlib import Path

from moviepy import ColorClip, CompositeVideoClip, TextClip


def pick_font() -> str:
    # Use a real font FILE PATH (what fc-list shows)
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise RuntimeError("No known font file found. Try: sudo apt install fonts-dejavu-core")


def wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(s, width=width))


def bullets_for(topic: str) -> list[str]:
    t = topic.lower()
    if "ai tool" in t or "ai tools" in t:
        return [
            "ChatGPT / Claude → drafts, rewrites, ideas",
            "Perplexity → research with sources fast",
            "Notion AI / Copilot → summaries + action items",
        ]
    return [
        "Hook: here’s the punchline",
        "3 quick points",
        "CTA: follow for more",
    ]


def make_text(
    text: str,
    *,
    font_path: str,
    font_size: int,
    color: str = "white",
    size=(960, None),
    method: str = "caption",
    text_align: str = "center",
    stroke_color: str = "black",
    stroke_width: int = 2,
):
    # MoviePy 2.1.2 expects keyword args (text=, font=, font_size=)
    return TextClip(
        text=text,
        font=font_path,
        font_size=font_size,
        color=color,
        size=size,
        method=method,
        text_align=text_align,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--out", default="outputs/short.mp4")
    ap.add_argument("--duration", type=float, default=18.0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    W, H = 1080, 1920
    duration = float(args.duration)
    font_path = pick_font()

    bg = ColorClip(size=(W, H), color=(15, 15, 18)).with_duration(duration)

    title_txt = wrap(args.topic.strip(), 18)
    title = (
        make_text(title_txt, font_path=font_path, font_size=78, size=(980, None))
        .with_duration(duration)
        .with_position(("center", 260))
    )

    bullet_lines = [f"• {wrap(b, 26)}" for b in bullets_for(args.topic)]
    body_txt = "\n\n".join(bullet_lines)
    body = (
        make_text(body_txt, font_path=font_path, font_size=56, size=(980, None))
        .with_duration(duration)
        .with_position(("center", 650))
    )

    footer_txt = random.choice(["Follow for more ✅", "Save this ✅", "Comment 'AI' for links ✅"])
    footer = (
        make_text(footer_txt, font_path=font_path, font_size=52, size=(980, None))
        .with_duration(duration)
        .with_position(("center", 1700))
    )

    video = CompositeVideoClip([bg, title, body, footer], size=(W, H)).with_duration(duration)

    video.write_videofile(
        str(out_path),
        fps=30,
        codec="libx264",
        audio=False,
        preset="medium",
        threads=4,
    )

    print(f"\n✅ Wrote: {out_path}\nFont used: {font_path}\n")


if __name__ == "__main__":
    main()
