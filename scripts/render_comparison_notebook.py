#!/usr/bin/env python3
"""Build and execute the full comparison notebook after all source jobs complete."""
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/final_comparison.ipynb"


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/build_comparison_notebook.py")],
                   cwd=ROOT, check=True)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(notebook, timeout=1200, kernel_name="dllm-eval",
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        # Preserve executed cells for diagnosis if a later cell fails.
        nbformat.write(notebook, NOTEBOOK)
    nbformat.validate(notebook)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} cells: {NOTEBOOK}")


if __name__ == "__main__":
    main()
