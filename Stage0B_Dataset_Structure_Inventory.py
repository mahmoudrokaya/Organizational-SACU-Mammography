r"""
Stage0B_Dataset_Structure_Inventory.py

Purpose
-------
Generate a complete folder/file structure inventory for all downloaded
mammography datasets and save the report inside the datasets directory.

Output
------
D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments\datasets\Dataset_Structure_Report.txt
"""

from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATASET_ROOT = Path(
    r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments\datasets"
)

OUTPUT_FILE = DATASET_ROOT / "Dataset_Structure_Report.txt"

# ---------------------------------------------------------------------
# Tree Generation
# ---------------------------------------------------------------------
def build_tree(root_path: Path):
    lines = []

    def walk(path: Path, prefix=""):
        items = sorted(
            list(path.iterdir()),
            key=lambda p: (not p.is_dir(), p.name.lower())
        )

        for i, item in enumerate(items):
            is_last = i == len(items) - 1

            connector = "└── " if is_last else "├── "

            if item.is_dir():
                lines.append(
                    f"{prefix}{connector}{item.name}/"
                )

                extension = "    " if is_last else "│   "
                walk(item, prefix + extension)

            else:
                try:
                    size_mb = item.stat().st_size / (1024 * 1024)
                    lines.append(
                        f"{prefix}{connector}{item.name} "
                        f"({size_mb:.2f} MB)"
                    )
                except Exception:
                    lines.append(
                        f"{prefix}{connector}{item.name}"
                    )

    lines.append(f"{root_path.name}/")
    walk(root_path)

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Dataset Statistics
# ---------------------------------------------------------------------
def collect_statistics(root_path: Path):
    total_files = 0
    total_dirs = 0
    total_size = 0

    extensions = {}

    for p in root_path.rglob("*"):
        if p.is_dir():
            total_dirs += 1

        elif p.is_file():
            total_files += 1

            try:
                total_size += p.stat().st_size
            except Exception:
                pass

            ext = p.suffix.lower()

            if ext == "":
                ext = "[NO_EXTENSION]"

            extensions[ext] = extensions.get(ext, 0) + 1

    stats = []
    stats.append("=" * 100)
    stats.append("DATASET INVENTORY SUMMARY")
    stats.append("=" * 100)
    stats.append(f"Root Folder : {root_path}")
    stats.append(f"Generated   : {datetime.now()}")
    stats.append(f"Folders     : {total_dirs:,}")
    stats.append(f"Files       : {total_files:,}")
    stats.append(f"Size (GB)   : {total_size / (1024**3):,.2f}")
    stats.append("")

    stats.append("FILE TYPES")
    stats.append("-" * 100)

    for ext, count in sorted(
        extensions.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        stats.append(f"{ext:<15} {count:,}")

    stats.append("")
    return "\n".join(stats)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    print("=" * 80)
    print("GENERATING DATASET STRUCTURE REPORT")
    print("=" * 80)

    summary = collect_statistics(DATASET_ROOT)
    tree = build_tree(DATASET_ROOT)

    report = (
        summary
        + "\n"
        + "=" * 100
        + "\nFULL DIRECTORY TREE\n"
        + "=" * 100
        + "\n"
        + tree
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print()
    print("REPORT GENERATED SUCCESSFULLY")
    print(f"Saved to: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    main()