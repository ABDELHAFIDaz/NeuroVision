"""Rewrite the /workspace/NeuroVision paths stored in mlflow.db after you download it from the VM.

Run once from the project root on your PC:
    uv run python fix_mlflow_paths.py
"""
import sqlite3
from pathlib import Path

OLD = "file:///workspace/NeuroVision"
NEW = Path.cwd().as_uri()

con = sqlite3.connect("mlflow.db")
con.execute("UPDATE experiments SET artifact_location = replace(artifact_location, ?, ?)", (OLD, NEW))
con.execute("UPDATE runs SET artifact_uri = replace(artifact_uri, ?, ?)", (OLD, NEW))
con.commit()
print("Paths now point to", NEW)