"""One-off smoke test: run every Streamlit page headlessly via AppTest and
report any exceptions. Not part of the pytest suite (kept as a manual dev
utility) - see tests/integration/test_streamlit_app.py for the real,
CI-covered version of these checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGES = sorted((Path(__file__).resolve().parents[1] / "app" / "pages").glob("*.py"))
HOME = Path(__file__).resolve().parents[1] / "app" / "Home.py"

failed = False
for page in [HOME, *PAGES]:
    at = AppTest.from_file(str(page), default_timeout=120)
    at.run()
    if at.exception:
        failed = True
        print(f"FAIL: {page.name}")
        for e in at.exception:
            print("  ", e)
    else:
        print(f"OK:   {page.name}")

sys.exit(1 if failed else 0)
