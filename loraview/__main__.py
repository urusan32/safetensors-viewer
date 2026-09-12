import os
import sys

if __package__ in (None, ""):
    # Invoked as `python3 loraview ...` (directory as script): there is no
    # package context, so relative imports fail. Put the repo root on the path
    # and import via the absolute package name instead.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from loraview.cli import main
else:
    from .cli import main

raise SystemExit(main())
