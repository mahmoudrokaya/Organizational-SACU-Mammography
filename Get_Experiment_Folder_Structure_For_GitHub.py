# Get_Experiment_Folder_Structure_For_GitHub.py

from pathlib import Path
from datetime import datetime

ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

OUT_TXT = ROOT / "Experiment_Folder_Structure_For_GitHub.txt"
OUT_MD = ROOT / "Experiment_Folder_Structure_For_GitHub.md"

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    "venv",
    "env",
    ".venv",
}

EXCLUDE_EXTS = {
    ".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    ".npy", ".npz", ".pkl", ".joblib", ".pt", ".pth", ".h5", ".hdf5",
}

MAX_DEPTH = 8


def should_skip(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    if path.is_file() and path.suffix.lower() in EXCLUDE_EXTS:
        return True
    return False


def file_size(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    if size < 1024**3:
        return f"{size / 1024**2:.1f} MB"
    return f"{size / 1024**3:.1f} GB"


def build_tree(root: Path):
    lines = []
    lines.append(f"Experiment folder structure")
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Root: {root}")
    lines.append("=" * 100)
    lines.append("")

    def walk(current: Path, prefix: str = "", depth: int = 0):
        if depth > MAX_DEPTH:
            lines.append(prefix + "└── ... depth limit reached")
            return

        try:
            items = sorted(
                [p for p in current.iterdir() if not should_skip(p)],
                key=lambda p: (p.is_file(), p.name.lower())
            )
        except PermissionError:
            lines.append(prefix + "└── [Permission denied]")
            return

        for i, item in enumerate(items):
            connector = "└── " if i == len(items) - 1 else "├── "
            next_prefix = prefix + ("    " if i == len(items) - 1 else "│   ")

            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                walk(item, next_prefix, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{item.name} ({file_size(item)})")

    lines.append(f"{root.name}/")
    walk(root)

    return lines


def summarize_files(root: Path):
    rows = []

    for path in root.rglob("*"):
        if should_skip(path) or not path.is_file():
            continue

        rel = path.relative_to(root)

        rows.append({
            "relative_path": str(rel),
            "folder": str(rel.parent),
            "file_name": path.name,
            "extension": path.suffix.lower(),
            "size": file_size(path),
        })

    return rows


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Root does not exist: {ROOT}")

    tree_lines = build_tree(ROOT)
    summary_rows = summarize_files(ROOT)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(tree_lines))

    md = []
    md.append("# Experiment Folder Structure for GitHub Preparation")
    md.append("")
    md.append(f"Generated: `{datetime.now()}`")
    md.append("")
    md.append(f"Root: `{ROOT}`")
    md.append("")
    md.append("## Folder Tree")
    md.append("")
    md.append("```text")
    md.extend(tree_lines)
    md.append("```")
    md.append("")
    md.append("## File Inventory")
    md.append("")
    md.append("| Relative path | Extension | Size |")
    md.append("|---|---:|---:|")

    for r in summary_rows:
        md.append(f"| `{r['relative_path']}` | `{r['extension']}` | {r['size']} |")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("=" * 100)
    print("EXPERIMENT FOLDER STRUCTURE EXPORTED")
    print("-" * 100)
    print(f"TXT: {OUT_TXT}")
    print(f"MD:  {OUT_MD}")
    print("=" * 100)


if __name__ == "__main__":
    main()