"""Render a real saved model smoke result as a terminal product demonstration."""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    """Show measured model outputs, including poor summaries, without embellishment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/eval/functional-smoke.json")
    parser.add_argument("--output", default="docs/model-demo.gif")
    args = parser.parse_args()
    record = json.loads(Path(args.input).read_text())
    slides = []
    for question, item in record["qa"].items():
        result = item["result"]
        lines = [
            "Question: " + question,
            "Answer: " + result["answer"],
            f"Heuristic support score: {result['support_score']:.3f}",
        ]
        if result["sources"]:
            source = result["sources"][0]
            lines += [
                f"Source: {source['paper']} | {source['section']} | PDF page {source['page']}",
                "Passage: " + source["passage"],
            ]
        slides.append(lines)
    summary = record["summary"]
    slides.append(
        [
            "Real whole-paper model summary: " + summary["summary"],
            f"Mapped source windows: {len(summary['map_summaries'])}",
            f"Hierarchical reduction rounds: {summary['reduce_rounds']}",
            "This small model output can be poor. Verify against the full source audit.",
            f"Study notes generated: {record['notes_bytes']} characters",
        ]
    )
    frames = []
    for lines in slides:
        frame = Image.new("RGB", (1000, 410), "#111827")
        draw = ImageDraw.Draw(frame)
        draw.text(
            (24, 22),
            "Research Paper Assistant | recorded pretrained-model CPU smoke",
            fill="#67e8f9",
        )
        draw.text(
            (24, 50),
            "Actual output on Attention Is All You Need; scores are uncalibrated.",
            fill="#fbbf24",
        )
        y = 95
        for line in lines:
            for wrapped in textwrap.wrap(line, width=130):
                draw.text((24, y), wrapped, fill="#f3f4f6")
                y += 18
            y += 12
        frames.append(frame)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=4000, loop=0)


if __name__ == "__main__":
    main()
