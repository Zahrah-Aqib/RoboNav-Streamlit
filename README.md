# RoboNav Streamlit | AI Lab Project

Robot Navigation in an Obstacle-Filled Environment: A Comparative Study of Greedy Best-First Search and A* Search

This is the standalone Python Streamlit + NumPy + Matplotlib project for free Streamlit Community Cloud hosting. Its **Home page recreates the layout and content of the original React RoboNav homepage**, with a decorative 3D robot (and CSS fallback), feature cards and navigation to the real Python simulator. The original React/Three.js app remains a separate project.

## Features

- Original React-inspired homepage: "Watch your robot think its way home," decorative Three.js robot, simulator/comparison buttons, four feature cards, metric strip and three-step explanation.
- Default entry page is **Home**; existing simulator, comparison, experiments and methodology remain accessible via the sidebar.

- From-scratch Python 3 A* and Greedy Best-First Search, priority frontier (heapq), visited tracking, Manhattan/Euclidean heuristics.
- Configurable 2D obstacle map, custom start/goal, random seed and obstacle density, JSON/ASCII input and manual obstacle editor.
- Matplotlib step-by-step frontier, explored-node, route, and moving-robot visualizations.
- Matched algorithm comparison and experiments at 10x10, 20x20, 50x50, densities 10%, 20%, 30%, several random seeds.
- U-shaped Greedy trap and percentage of non-optimal Greedy routes.
- Real CSV, PNG, and measured PDF experimental report downloads.
- No Kaggle dataset, ML training, database, JavaScript build or separate API server.

## Windows setup

Open PowerShell in the project root:

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py

Open http://localhost:8501. The site opens on Home; the sidebar lets you select Grid Simulator, Algorithm Comparison, Experiment Lab or Methodology.

## Free Streamlit deployment

1. The repository is https://github.com/Zahrah-Aqib/RoboNav-Streamlit.
2. Sign in at https://share.streamlit.io with GitHub.
3. Open the existing https://robonav-navigation.streamlit.app app and select **Reboot app** if GitHub's latest commit has not loaded automatically.
4. Keep branch `main`, main file `streamlit_app.py`, and Python 3.12.
5. Verify the Home page, sidebar navigation, and experiment reports.

## Project files

    streamlit_app.py
    home/robot_hero.html
    requirements.txt
    .streamlit/config.toml
    backend/algorithms/search.py
    backend/algorithms/maps.py
    backend/experiments.py
    backend/report.py
    backend/tests/
    sample_maps/
    scripts/check_streamlit.py
    docs/
    README.md

## Tests

    .\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
    .\.venv\Scripts\python.exe scripts/check_streamlit.py

## Proposal and limitations

The submitted proposal names Python 3, a 2D static grid, Greedy and A* implemented from scratch, a priority-queue frontier, Manhattan and Euclidean heuristics, graphical search tracing and measured experiments. It proposes Matplotlib/Pygame/Tkinter as alternative Python visualization libraries; this project uses Matplotlib.

Navigation allows four-direction movements costing 1 per step. Eight-direction movement is optional in the proposal and is not implemented. Execution time excludes plotting. The final report PDF should be reviewed and supplemented with the team's interpretation, working app screenshots, live URL, and a demonstration video if requested.

See docs/STREAMLIT_SETUP.md and docs/VIVA_GUIDE.md.
