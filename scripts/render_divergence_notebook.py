#!/usr/bin/env python3
"""Rebuild and execute the divergence analysis with current validated data."""
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'notebooks/divergence_analysis.ipynb'
subprocess.run([sys.executable,str(ROOT/'scripts/build_divergence_notebook.py')],cwd=ROOT,check=True)
notebook=nbformat.read(TARGET,as_version=4)
client=NotebookClient(notebook,timeout=1200,kernel_name='dllm-eval',
                      resources={'metadata':{'path':str(ROOT)}})
try:
    client.execute()
finally:
    nbformat.write(notebook,TARGET)
nbformat.validate(notebook)
print('executed',sum(c.cell_type=='code' for c in notebook.cells),'cells',TARGET)
