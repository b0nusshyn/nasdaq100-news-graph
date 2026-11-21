"""Render an offline SVG learning curve without optional plotting dependencies."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
COLORS = {42: "#2369ac", 43: "#b45b16"}
METRICS = {"accuracy": "Accuracy", "within_day_auc": "Within-day AUC", "mcc": "MCC"}


def render(records):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="800" viewBox="0 0 1120 800">',
           '<rect width="1120" height="800" fill="white"/>',
           '<g font-family="system-ui, sans-serif" fill="#162332">',
           '<text x="50" y="39" font-size="24" font-weight="bold">Five-day training history learning curves</text>',
           '<text x="50" y="65" font-size="13">Historical development only • seed 42 blue • seed 43 orange</text>']
    maximum = max(r["train_rows"] for r in records) * 1.05
    for row, (metric, title) in enumerate(METRICS.items()):
        v = [r["assessment"][metric] for r in records]
        lo, hi = min(v), max(v)
        pad = (hi - lo or .01) * .15
        low, high = lo - pad, hi + pad
        for w in (0, 1):
            left, top, width, height = 75 + 545*w, 115 + row*215, 430, 142
            svg.append(f'<text x="{left}" y="{top-15}" font-size="15">{2022+w} · {title}</text>')
            for i in range(4):
                y = top + i*height/3
                val = high - i*(high-low)/3
                svg.append(f'<path d="M{left} {y}h{width}" stroke="#dbe3eb"/>')
                svg.append(f'<text x="{left-7}" y="{y+4}" font-size="11" text-anchor="end">{val:.3f}</text>')
            for xval in (0, 25000, 50000, 75000, 100000):
                if xval > maximum:
                    continue
                x = left + width*xval/maximum
                svg.append(f'<text x="{x}" y="{top+height+16}" font-size="11" text-anchor="middle">'
                           f'{xval//1000}k</text>')
            for seed, color in COLORS.items():
                fit = sorted((r for r in records if r["fold"] == w and r["seed"] == seed), key=lambda r: r["fraction"])
                pts = [(left+width*r["train_rows"]/maximum, top+height*(high-r["assessment"][metric])/(high-low))
                       for r in fit]
                line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                svg.append(f'<polyline points="{line}" stroke="{color}" stroke-width="2" fill="none"/>')
                for x, y in pts:
                    svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{color}"/>')
    return "\n".join([*svg, '<text x="560" y="773" text-anchor="middle">Labelled training ticker-days</text>',
                      "</g></svg>"])


if __name__ == "__main__":
    runs = [r for r in map(json.loads, (ROOT/"runs/ledger.jsonl").read_text().splitlines()) if "skipped" not in r]
    (ROOT / "learning_curves.svg").write_text(render(runs))
