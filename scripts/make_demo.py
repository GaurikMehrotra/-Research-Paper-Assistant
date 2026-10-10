"""Record real infrastructure checks as a GIF; this is not a pretrained-model demo."""

import argparse
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    """Render captured command outcomes, failing rather than fabricating success frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="python")
    parser.add_argument("--output", default="docs/infrastructure-demo.gif")
    args = parser.parse_args()
    checks = [
        [args.python, "-m", "ruff", "check", "."],
        [args.python, "-m", "pytest", "-q", "--disable-warnings"],
        [
            args.python,
            "-c",
            "import json,urllib.request; "
            "print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')))",
        ],
        [
            args.python,
            "-c",
            "import json,urllib.request; "
            "r=json.load(urllib.request.urlopen('http://127.0.0.1:7860/config')); "
            "print('Gradio components:',len(r['components']))",
        ],
    ]
    frames = []
    for command in checks:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        frame = Image.new("RGB", (900, 330), "#111827")
        draw = ImageDraw.Draw(frame)
        draw.text((24, 20), "RESEARCH PAPER ASSISTANT | infrastructure checks", fill="#67e8f9")
        draw.text(
            (24, 50),
            "Real command output. Offline tests use injected model adapters.",
            fill="#d1d5db",
        )
        draw.text(
            (24, 76), "No pretrained-model quality or product demo is claimed.", fill="#fbbf24"
        )
        lines = textwrap.wrap("$ " + " ".join(command[1:]), width=110)
        lines += [""] + result.stdout.strip().splitlines()[:8]
        for number, line in enumerate(lines):
            draw.text((24, 118 + number * 17), line[:140], fill="#f3f4f6")
        frames.append(frame)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=2300, loop=0)
    print(f"Saved actual infrastructure demonstration: {path}")


if __name__ == "__main__":
    main()
