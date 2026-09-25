"""Allows `python -m pneumonia_cxr <command>` as an alternative to `pcxr <command>`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
