"""
Legacy wrapper for the maintained Airtable cron entrypoint.

Running this file executes the root-level `cron.py` script so legacy usage
inherits the current logging and failure-handling behavior.
"""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cron import main


if __name__ == "__main__":
    main()
