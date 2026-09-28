#!/usr/bin/env python3
"""
Pipeline entrypoint (no dashboard). Same command as before:

    python run.py                      # run -- or reuse -- the run for paths.raw_data
    python run.py --input data/raw/x.json
    python run.py --force              # new run even if an identical completed run exists

Results go to runs/<RUN_ID>/ (see README "Runs"). A completed run for the same
dataset + pipeline version + config is reused, not recomputed.

To run the pipeline AND start the dashboard in one step, use ./run.sh.
"""
import sys

from src.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
