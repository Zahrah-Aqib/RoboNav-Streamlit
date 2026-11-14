"""Headless checks for Streamlit's default screen and generated PDF report."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from streamlit.testing.v1 import AppTest
from experiments import run_experiments
from report import make_report_pdf


def main():
    # Run the app in the test harness without opening a browser.
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=90)
    app.run()
    if app.exception:
        raise AssertionError("The Streamlit app raised an exception: " + str(app.exception))
    # The site must open on Home, with the React-inspired welcome heading.
    # This prevents future edits from accidentally dropping the landing page.
    if not any("Watch your robot" in element.value
               for element in app.markdown):
        raise AssertionError("The RoboNav welcome homepage did not render.")
    print("Original React-inspired Streamlit homepage loaded successfully.")

    # Keep the existing simulator accessible from the native sidebar.
    app.sidebar.radio[0].set_value("Grid Simulator").run()
    if app.exception:
        raise AssertionError("Grid Simulator failed after navigating from Home: "
                             + str(app.exception))
    if not any("2D Grid Simulator" in element.value
               for element in app.header):
        raise AssertionError("Grid Simulator page is unavailable.")
    print("Existing Grid Simulator navigation works.")

    # The generated report must use real computed Python measurements.
    result = run_experiments([10], [0], [1], include_u_trap=True)
    document = make_report_pdf(result)
    assert document.startswith(b"%PDF"), "The experimental report is not a valid PDF."
    assert len(document) > 2000, "Report PDF appears unexpectedly small."
    print("Measured experimental report produced successfully.")


if __name__ == "__main__":
    main()
