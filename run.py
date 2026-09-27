#!/usr/bin/env python3
"""
Single-command entrypoint for the full TRACE-X Bitcoin forensic pipeline.

Usage:
    python run.py

This performs: ingestion -> preprocessing -> correlation -> graph -> features ->
ML (Isolation Forest) -> clustering (DBSCAN) -> risk scoring -> explanations ->
evaluation -> cached outputs.

After this completes, launch the dashboard with:
    streamlit run dashboard/app.py
"""
from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
