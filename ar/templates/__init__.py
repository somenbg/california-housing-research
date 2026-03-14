"""Template management — list, scaffold, and validate templates."""

import os
import shutil
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent

AVAILABLE_TEMPLATES = {
    "tabular": {
        "dir": "tabular",
        "description": "Tabular data (regression/classification) with PyTorch MLP",
        "model_class": "tabular",
        "works_on": "cpu, mps, cuda",
    },
}


def list_templates() -> dict:
    return AVAILABLE_TEMPLATES


def scaffold_template(template_name: str, dest_dir: str) -> list[str]:
    """Copy template files to the destination directory.

    Returns list of created file paths.
    """
    if template_name not in AVAILABLE_TEMPLATES:
        available = ", ".join(AVAILABLE_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")

    info = AVAILABLE_TEMPLATES[template_name]
    src_dir = TEMPLATES_DIR / info["dir"]

    if not src_dir.exists():
        raise FileNotFoundError(f"Template directory not found: {src_dir}")

    created = []
    for src_file in sorted(src_dir.iterdir()):
        if src_file.name.startswith("__"):
            continue
        dest_file = os.path.join(dest_dir, src_file.name)
        if os.path.exists(dest_file):
            print(f"  Skipping {src_file.name} (already exists)")
            continue
        shutil.copy2(src_file, dest_file)
        created.append(dest_file)
        print(f"  Created {src_file.name}")

    return created
