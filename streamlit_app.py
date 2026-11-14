"""RoboNav: Streamlit and Matplotlib companion to the React website.

Uses exactly the same backend/algorithms/search.py as the React FastAPI
frontend. Every figure and result comes from that Python search algorithm.
"""
from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from algorithms.maps import from_ascii, make_open_grid, make_u_trap
from algorithms.search import run_search, validate_map
from experiments import paired_runs, run_experiments
from report import make_report_pdf

st.set_page_config(page_title="RoboNav | Robot Navigation", page_icon="🤖", layout="wide")
PALETTE = ["#111d30", "#4a5670", "#126e86", "#aa7a29",
           "#19c7bf", "#6cca7c", "#f17c83", "#fff39b"]
CMAP = ListedColormap(PALETTE)
NORM = BoundaryNorm(np.arange(-0.5, 8.5, 1), 8)


def random_grid(rows: int, cols: int, density: float, seed: int) -> dict:
    """Generate a repeatable obstacle map from a fixed random seed."""
    grid = make_open_grid(rows, cols)
    rng = random.Random(seed)
    start = grid["start"]
    goal = grid["goal"]
    grid["obstacles"] = [
        r * cols + c for r in range(rows) for c in range(cols)
        if (r, c) not in (
            (start["row"], start["col"]),
            (goal["row"], goal["col"]),
        ) and rng.random() < density
    ]
    return grid


def map_plot(grid: dict, result: dict | None = None,
             step: int = 0, title: str = "Robot navigation"):
    """Draw exploration, frontier, path and moving robot on the 2D grid."""
    rows, cols = grid["rows"], grid["cols"]
    board = np.zeros((rows, cols), dtype=np.uint8)
    for index in grid["obstacles"]:
        board[index // cols, index % cols] = 1

    if result is not None and step:
        frames = result["frames"]
        for frame in frames[:min(step, len(frames))]:
            p = frame["explored"]
            board[p["row"], p["col"]] = 2

        if 0 < step <= len(frames):
            for p in frames[step - 1]["frontier"]:
                board[p["row"], p["col"]] = 3

        walked = max(0, step - len(frames))
        if walked and result["path"]:
            for p in result["path"][:walked]:
                board[p["row"], p["col"]] = 4
            robot = result["path"][min(walked - 1, len(result["path"]) - 1)]
            board[robot["row"], robot["col"]] = 7

    start, goal = grid["start"], grid["goal"]
    board[start["row"], start["col"]] = 5
    board[goal["row"], goal["col"]] = 6

    fig, ax = plt.subplots(
        figsize=(min(11, max(5, cols * 0.46)),
                 min(9, max(4, rows * 0.46))), dpi=105
    )
    fig.patch.set_facecolor("#091626")
    ax.set_facecolor("#091626")
    ax.imshow(board, cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#64748b", alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(np.arange(cols) if cols <= 20 else np.arange(0, cols, 5))
    ax.set_yticks(np.arange(rows) if rows <= 20 else np.arange(0, rows, 5))
    ax.tick_params(colors="#cbd5e1", labelsize=8)
    ax.set_xlabel("Column", color="#cbd5e1")
    ax.set_ylabel("Row", color="#cbd5e1")
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.text(start["col"], start["row"], "S", ha="center",
            va="center", color="#04111d", fontweight="bold")
    ax.text(goal["col"], goal["row"], "G", ha="center",
            va="center", color="#04111d", fontweight="bold")
    fig.tight_layout()
    return fig


def display_plot(grid, result=None, step=0, title="Robot navigation"):
    """Display a Matplotlib figure and close it to avoid memory leaks."""
    fig = map_plot(grid, result, step, title)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def sidebar_map():
    """A fully configurable 2D map, including checkbox obstacle editing."""
    st.sidebar.subheader("Grid map")
    source = st.sidebar.selectbox(
        "Map source",
        ["Random", "Empty", "U-shaped trap", "Upload JSON", "ASCII map"]
    )
    if source == "U-shaped trap":
        grid = make_u_trap()
        st.sidebar.caption("Known case: A* cost 15, Greedy cost 39 (Manhattan).")
    elif source == "Upload JSON":
        uploaded = st.sidebar.file_uploader("Choose a grid JSON", type="json")
        if uploaded is None:
            grid = make_open_grid()
        else:
            try:
                grid = json.load(uploaded)
                validate_map(grid)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                st.sidebar.error("Invalid JSON map: " + str(error))
                st.stop()
    elif source == "ASCII map":
        text = st.sidebar.text_area(
            "S=start, G=goal, #=wall, .=empty",
            "S.....\n..##..\n......\n..##..\n.....G",
            height=150
        )
        try:
            grid = from_ascii([s.strip() for s in text.splitlines() if s.strip()])
        except ValueError as error:
            st.sidebar.error(str(error))
            st.stop()
    else:
        rows = int(st.sidebar.number_input("Rows", 2, 100, 12))
        cols = int(st.sidebar.number_input("Columns", 2, 100, 16))
        if source == "Empty":
            grid = make_open_grid(rows, cols)
        else:
            density = st.sidebar.slider("Obstacle density", 0, 60, 20, 5) / 100
            seed = int(st.sidebar.number_input("Random seed", 0, 100000, 12))
            grid = random_grid(rows, cols, density, seed)

    if st.sidebar.checkbox("Customize start and goal", False):
        for label, key in [("Start", "start"), ("Goal", "goal")]:
            row = st.sidebar.number_input(
                label + " row", 0, grid["rows"] - 1, int(grid[key]["row"])
            )
            col = st.sidebar.number_input(
                label + " column", 0, grid["cols"] - 1, int(grid[key]["col"])
            )
            grid[key] = {"row": int(row), "col": int(col)}

    # A native Streamlit checkbox table acts as a direct wall editor.
    if grid["rows"] <= 25 and grid["cols"] <= 25:
        if st.sidebar.checkbox("Manually edit obstacle cells", False):
            st.sidebar.caption("Check or uncheck cells to place or erase walls.")
            matrix = np.zeros((grid["rows"], grid["cols"]), dtype=bool)
            for index in grid["obstacles"]:
                matrix[index // grid["cols"], index % grid["cols"]] = True
            data = pd.DataFrame(
                matrix,
                index=["R" + str(r) for r in range(grid["rows"])],
                columns=["C" + str(c) for c in range(grid["cols"])],
            )
            edited = st.data_editor(
                data, use_container_width=True, height=min(560, 80 + grid["rows"] * 35),
                key="map_editor_" + source
            )
            grid["obstacles"] = [
                r * grid["cols"] + c
                for r in range(grid["rows"]) for c in range(grid["cols"])
                if bool(edited.iloc[r, c])
            ]

    # Make sure start/goal cells are never obstacles.
    forbidden = {
        grid["start"]["row"] * grid["cols"] + grid["start"]["col"],
        grid["goal"]["row"] * grid["cols"] + grid["goal"]["col"],
    }
    grid["obstacles"] = sorted(set(grid["obstacles"]) - forbidden)
    try:
        validate_map(grid)
    except (ValueError, KeyError, TypeError) as error:
        st.sidebar.error(str(error))
        st.stop()
    st.sidebar.download_button(
        "Save this map as JSON", json.dumps(grid, indent=2),
        file_name="robonav-map.json", mime="application/json",
        use_container_width=True,
    )
    return grid


def stats_cards(result):
    """Display actual backend measurements, never placeholder values."""
    stat = result["stats"]
    c = st.columns(4)
    c[0].metric("Path cost", stat["pathCost"] if stat["pathCost"] is not None else "No path")
    c[1].metric("Nodes expanded", stat["nodesExpanded"])
    c[2].metric("Maximum frontier", stat["maxFrontier"])
    c[3].metric("Time (ms)", f'{stat["executionTimeMs"]:.3f}')


def step_control(result, key):
    """Select a search step or one step of robot travel."""
    search_frames = len(result["frames"])
    max_step = search_frames + len(result["path"])
    st.caption(
        "First " + str(search_frames) + " frames show frontier exploration; "
        "subsequent frames move the robot along the discovered path."
    )
    step = st.slider("Animation step", 0, max_step, 0, key=key)
    if 0 < step <= search_frames:
        frame = result["frames"][step - 1]
        p = frame["explored"]
        st.caption(
            "Selected (" + str(p["row"]) + "," + str(p["col"]) + ")"
            " | g=" + str(frame["g"]) + " | h=" + str(frame["h"])
            + " | f=" + str(frame["f"])
        )
        st.markdown("**Actual frontier list after this expansion**")
        details = pd.DataFrame(frame["frontierDetails"])
        if details.empty:
            st.info("Frontier is empty.")
        else:
            st.dataframe(details, hide_index=True, use_container_width=True)
    return step


def simulator():
    st.header("2D Grid Simulator")
    st.write("Configure obstacles and see real frontier and final route visualizations.")
    grid = sidebar_map()
    algo = st.selectbox("Algorithm", ["A* Search", "Greedy Best-First Search"])
    heuristic = st.selectbox("Heuristic", ["manhattan", "euclidean"])
    signature = json.dumps(grid, sort_keys=True) + algo + heuristic
    if st.session_state.get("sim_signature") != signature:
        st.session_state["sim_result"] = None
        st.session_state["sim_signature"] = signature
        st.session_state.pop("sim_step", None)

    if st.button("Run navigation", type="primary"):
        key = "astar" if algo == "A* Search" else "greedy"
        st.session_state["sim_result"] = run_search(grid, key, heuristic)
        st.session_state.pop("sim_step", None)

    result = st.session_state.get("sim_result")
    if result:
        stats_cards(result)
        if result["status"] != "found":
            st.warning("No route exists. The frontier was exhausted.")
        step = step_control(result, "sim_step")
        display_plot(grid, result, step, algo)
    else:
        display_plot(grid, title="Input map (S=start, G=goal)")


def comparison():
    st.header("A* vs Greedy: same grid, fair comparison")
    grid = sidebar_map()
    heuristic = st.selectbox("Shared heuristic", ["manhattan", "euclidean"])
    signature = json.dumps(grid, sort_keys=True) + heuristic
    if st.session_state.get("comparison_signature") != signature:
        st.session_state["comparison_result"] = None
        st.session_state["comparison_signature"] = signature
        st.session_state.pop("compare_step", None)
    if st.button("Run both algorithms", type="primary"):
        st.session_state["comparison_result"] = paired_runs(grid, heuristic)
        st.session_state.pop("compare_step", None)
    result = st.session_state.get("comparison_result")
    if not result:
        display_plot(grid, title="Shared grid")
        return
    a, g = result["astar"], result["greedy"]
    length = max(len(a["frames"]) + len(a["path"]),
                 len(g["frames"]) + len(g["path"]))
    step = st.slider("Shared animation step", 0, length, 0, key="compare_step")
    left, right = st.columns(2)
    for col, name, run in [
        (left, "A* | f(n)=g(n)+h(n)", a),
        (right, "Greedy | f(n)=h(n)", g)
    ]:
        with col:
            st.subheader(name)
            display_plot(grid, run, step, name)
            stats_cards(run)
    ac, gc = a["stats"]["pathCost"], g["stats"]["pathCost"]
    if ac is not None and gc is not None:
        difference = gc - ac
        st.info("Greedy cost - A* cost = " + str(difference)
                + (" (Greedy is non-optimal)." if difference > 0 else " (equal costs)."))
    else:
        st.warning("No path was found by one or both algorithms.")


def experiment_lab():
    st.header("Experiment Lab")
    st.write("Matched maps with reproducible random seeds, plus a U-shaped trap.")
    sizes = st.multiselect("Grid sizes", [10, 20, 50, 100], [10, 20, 50])
    densities = st.multiselect("Obstacle density (%)", [0, 10, 20, 30, 40], [10, 20, 30])
    nseeds = st.slider("Number of seeds", 1, 8, 3)
    heuristic = st.selectbox("Heuristic", ["manhattan", "euclidean"], key="lab_heuristic")
    include_trap = st.checkbox("Add U-shaped Greedy trap", value=True)
    if st.button("Run real experiments", type="primary"):
        if not sizes or not densities:
            st.error("Choose at least one size and one obstacle density.")
        else:
            with st.spinner("Running measured Python searches..."):
                st.session_state["batch_result"] = run_experiments(
                    sizes, [d / 100 for d in densities],
                    list(range(1, nseeds + 1)), heuristic, include_trap
                )
    data = st.session_state.get("batch_result")
    if not data:
        st.info("Choose settings and run the experiment batch.")
        return
    overall = data["overall"]
    c = st.columns(3)
    c[0].metric("Maps tested", overall["mapsTested"])
    c[1].metric("Solved by both", overall["comparableRuns"])
    rate = overall["greedyNonOptimalPct"]
    c[2].metric("Non-optimal Greedy %", str(rate) + "%" if rate is not None else "N/A")
    st.caption(
        "Non-optimal Greedy % = 100 × (maps where Greedy cost > A* cost)"
        " / (maps solved by both algorithms). Unsolved maps are reported separately."
    )
    summary = pd.DataFrame(data["summary"])
    rows = pd.DataFrame(data["rows"])
    st.subheader("Group averages")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.subheader("Every measured run")
    st.dataframe(rows, hide_index=True, use_container_width=True, height=320)
    st.download_button("Download all runs (CSV)", rows.to_csv(index=False),
                       "robonav_runs.csv", "text/csv")
    st.download_button("Download group averages (CSV)", summary.to_csv(index=False),
                       "robonav_summary.csv", "text/csv")

    metrics = ["avgNodesExpanded", "avgPathCost", "avgMaxFrontier",
               "avgExecutionTimeMs"]
    metric = st.selectbox("Graph metric", metrics)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for size in sizes:
        for algo, color in (("astar", "#1ab6d2"), ("greedy", "#e9a52c")):
            samples = sorted(
                (r for r in data["summary"] if r["size"] == size
                 and r["algorithm"] == algo and r[metric] is not None),
                key=lambda r: r["density"],
            )
            if samples:
                ax.plot([100 * r["density"] for r in samples],
                        [r[metric] for r in samples], marker="o",
                        color=color, label=str(size) + "x" + str(size) + " " + algo)
    ax.set_title("Search performance vs obstacle density")
    ax.set_xlabel("Obstacle density (%)")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=180, bbox_inches="tight")
    st.download_button("Download Matplotlib chart (PNG)", png.getvalue(),
                       "robonav_chart.png", "image/png")
    # Build a real PDF report from the same measured rows and chart.
    pdf = make_report_pdf(data, png.getvalue())
    st.download_button("Download experimental report (PDF)", pdf,
                       "RoboNav_Experimental_Report.pdf", "application/pdf")
    plt.close(fig)
    if include_trap:
        st.subheader("Hand-designed U-shaped obstacle test")
        case = [p for p in data["comparisons"]
                if p["mapName"] == "U-shaped Greedy trap"]
        st.dataframe(pd.DataFrame(case), hide_index=True, use_container_width=True)


def methodology():
    st.header("Project methodology")
    st.markdown("""
    **State:** current (row, column); **start/goal:** designated free cells;
    **actions:** up, down, left, right; **transition cost:** 1 per move;
    **goal test:** selected cell equals destination.

    **Greedy Best-First Search:** f(n) = h(n), expanding the frontier node
    with the lowest estimated remaining cost.

    **A* Search:** f(n) = g(n) + h(n), including known travel cost.
    Its path is optimal on this static, four-direction, unit-cost grid
    under Manhattan or Euclidean heuristics.

    Both use Python heapq and a visited/explored set, with cost updates in A*.
    Actual search traces are visualized in Streamlit using Matplotlib and NumPy.

    **Experimental methodology:** Both algorithms use identical grids for
    every size, obstacle density and random seed. Compare path costs,
    nodes expanded, maximum frontier size and execution time.
    Separately test a U-shaped obstacle trap and unsolvable grids.
    Report the rate of non-optimal Greedy paths using successful matched runs
    as the denominator.

    **Final submission:** GitHub code, requirements, sample inputs,
    PDF project report with actual graphs/results, deployed app link,
    screenshots and a short demo video if requested.
    """)
    st.markdown(
        "Code repository: https://github.com/Zahrah-Aqib/RoboNav-Streamlit"
    )



# -----------------------------------------------------------------------------
# Original React-inspired homepage
# -----------------------------------------------------------------------------
# The layout and text follow the ORIGINAL React homepage in
# Final-RoboNav-Website/src/pages/Home.tsx. Unlike RoboNav-3D, this is not a
# scroll-animated website. Navigation stays native to Streamlit.
def open_page(page_name: str) -> None:
    """Update the sidebar radio from a homepage action button."""
    st.session_state["rn_page"] = page_name


def homepage() -> None:
    """Welcome page with the original React hero, robot, cards and workflow."""
    st.markdown(
        """
<style>
/* Only injected on the Home page, so existing simulator widgets keep their
   usual Streamlit styling. Colors mirror the original React navy/cyan theme. */
.rn-kicker {
  display: inline-block; padding: 7px 12px; border-radius: 20px;
  color: #67e8f9; font-size: .76rem; font-weight: 650; letter-spacing: .06em;
  border: 1px solid #22d3ee55; background: #0e749022; margin-bottom: 15px;
}
.rn-title {
  margin: 0 0 20px; font-family: "Space Grotesk", Inter, system-ui, sans-serif;
  font-weight: 780; font-size: clamp(2.5rem, 4.1vw, 4.3rem);
  line-height: 1.07; letter-spacing: -.045em; color: #f8fafc;
}
.rn-title span { color: #67e8f9; text-shadow: 0 0 22px #22d3ee33; }
.rn-lead {
  max-width: 570px; color: #a9b8cc; font-size: clamp(.97rem, 1.35vw, 1.14rem);
  line-height: 1.7; margin: 0 0 20px;
}
.rn-heading {
  font-weight: 760; font-size: clamp(1.45rem, 2.1vw, 1.95rem);
  letter-spacing: -.025em; color: white; margin: 35px 0 20px;
}
.rn-heading::before {
  content: ""; display:inline-block; width: 25px; height: 3px;
  background: #22d3ee; border-radius: 2px; vertical-align: middle;
  margin-right: 11px;
}
.rn-feature-icon {
  width: 43px; height: 43px; display: grid; place-items:center;
  background: #0e749028; border: 1px solid #22d3ee66;
  color: #67e8f9; border-radius: 12px; font-size: 1.4rem;
  margin-bottom: 10px;
}
.rn-feature-title { font-weight: 740; color: #f1f5f9; font-size: 1.07rem; margin-bottom: 5px; }
.rn-feature-desc { color: #9aaac1; font-size: .86rem; line-height: 1.6; min-height: 66px; }
.rn-metric-strip {
  display: grid; grid-template-columns: repeat(3, minmax(0,1fr));
  gap: 20px; border: 1px solid #274160cc; border-radius: 15px;
  background: #0a1224bf; padding: 24px; margin: 10px 0 28px;
}
.rn-metric-title { font-weight: 730; color: #f1f5f9; }
.rn-metric-desc { color: #96a6bd; font-size: .84rem; margin-top: 5px; }
.rn-step { border: 1px solid #274160cc; background: #0a1224bf;
  border-radius: 16px; padding: 22px; height: 100%; min-height: 155px; }
.rn-step-no { color: #67e8f9; letter-spacing: .12em; font-size: .72rem; font-weight: 760; }
.rn-step-title { color: #f1f5f9; font-size: 1.12rem; font-weight: 740; margin: 10px 0 7px; }
.rn-step-desc { color: #97a7bd; font-size: .86rem; line-height: 1.62; }
.rn-note { color: #91a1b9; font-size: .85rem; margin-top: 20px; }
.rn-note span { color: #fbbf24; }
.rn-footer { border-top: 1px solid #274160aa; padding: 24px 0 9px;
  margin-top: 45px; color: #8092ac; font-size: .82rem; }
/* Avoid cards/columns becoming unreadable at narrower widths. */
@media(max-width: 740px) {
  .rn-metric-strip { grid-template-columns: 1fr; gap: 14px; padding: 18px; }
  .rn-feature-desc { min-height: 0; }
  .rn-title { font-size: clamp(2.3rem, 10vw, 3.2rem); }
}
</style>
        """,
        unsafe_allow_html=True,
    )

    # Hero closely follows React's original copy and two-column structure.
    hero_copy, hero_robot = st.columns([1.16, 1], gap="large",
                                       vertical_alignment="center")
    with hero_copy:
        st.markdown(
            '<div class="rn-kicker">Robotics path-planning visualizer</div>'
            '<h1 class="rn-title">Watch your robot<br>'
            '<span>think its way home.</span></h1>'
            '<p class="rn-lead">RoboNav is an interactive laboratory for '
            'Greedy Best-First Search and A*. Draw a world, run both planners, '
            'and compare how each one explores, step by step, cell by cell.</p>',
            unsafe_allow_html=True,
        )
        primary, secondary = st.columns(2, gap="small")
        with primary:
            st.button("Launch the simulator", type="primary",
                      use_container_width=True, key="home_launch",
                      on_click=open_page, args=("Grid Simulator",))
        with secondary:
            st.button("Greedy vs A*", use_container_width=True,
                      key="home_compare", on_click=open_page,
                      args=("Algorithm Comparison",))
    with hero_robot:
        # Decorative Three.js port of the original React geometric robot.
        # Built-in CSS fallback renders when WebGL/CDN is unavailable.
        components.html(
            (ROOT / "home" / "robot_hero.html").read_text(encoding="utf-8"),
            height=370, scrolling=False,
        )

    st.markdown('<h2 class="rn-heading">What you can do</h2>',
                unsafe_allow_html=True)
    features = (
        ("▦", "Grid Simulator",
         "Paint obstacles, move the start and goal, and watch a search unfold cell by cell.",
         "Open simulator", "Grid Simulator"),
        ("⚖", "Algorithm Comparison",
         "Greedy Best-First versus A* side by side: explored cells, path cost, and speed.",
         "Compare algorithms", "Algorithm Comparison"),
        ("⚗", "Experiment Lab",
         "Batch maps through both planners and export the results as CSV for your report.",
         "Enter the lab", "Experiment Lab"),
        ("▤", "Documentation",
         "Learn the search rules, input formats and project methodology.",
         "Read the method", "Methodology"),
    )
    cards = st.columns(4, gap="medium")
    for i, (symbol, name, desc, action, page_name) in enumerate(features):
        with cards[i]:
            with st.container(border=True):
                st.markdown(
                    '<div class="rn-feature-icon">' + symbol + '</div>'
                    '<div class="rn-feature-title">' + name + '</div>'
                    '<div class="rn-feature-desc">' + desc + '</div>',
                    unsafe_allow_html=True,
                )
                st.button(action, use_container_width=True, key="home_card_" + str(i),
                          on_click=open_page, args=(page_name,))

    # Same performance metrics shown on the original React homepage.
    st.markdown(
        '<div class="rn-metric-strip">'
        '<div><div class="rn-metric-title">Path cost</div>'
        '<div class="rn-metric-desc">Total steps from start to goal</div></div>'
        '<div><div class="rn-metric-title">Nodes expanded</div>'
        '<div class="rn-metric-desc">How much of the map was explored</div></div>'
        '<div><div class="rn-metric-title">Execution time</div>'
        '<div class="rn-metric-desc">Wall-clock time reported by the planner</div></div>'
        '</div>', unsafe_allow_html=True,
    )

    st.markdown('<h2 class="rn-heading">How the project fits together</h2>',
                unsafe_allow_html=True)
    steps = (
        ("01", "Design a world",
         "Draw obstacles or generate them randomly, pick a heuristic, and set the grid size."),
        ("02", "Run both planners",
         "Python search functions execute each planner and record every expanded cell."),
        ("03", "Compare and export",
         "Inspect step-by-step results, study the metrics, and export CSV or PDF reports."),
    )
    for col, (number, title, desc) in zip(st.columns(3, gap="medium"), steps):
        with col:
            st.markdown(
                '<div class="rn-step"><div class="rn-step-no">' + number + '</div>'
                '<div class="rn-step-title">' + title + '</div>'
                '<div class="rn-step-desc">' + desc + '</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<p class="rn-note"><span>●</span> Search and experiments run '
        'directly in Python inside Streamlit; no separate API is needed.</p>'
        '<div class="rn-footer">RoboNav · Educational robot navigation '
        'with real Python A* and Greedy search. &nbsp; '
        'The homepage robot is decorative; the navigation grid is 2D.</div>',
        unsafe_allow_html=True,
    )


# The original React website starts at Home; the original Streamlit app now
# matches that visitor journey while keeping every existing working page.
PAGES = ["Home", "Grid Simulator", "Algorithm Comparison",
         "Experiment Lab", "Methodology"]
if st.session_state.get("rn_page") not in PAGES:
    st.session_state["rn_page"] = "Home"

st.sidebar.title("RoboNav")
st.sidebar.caption("Python-native robot navigation simulator")
page = st.sidebar.radio("Pages", PAGES, key="rn_page")
st.sidebar.divider()
st.sidebar.caption("Greedy Best-First Search and A* | AI Lab proposal")

if page == "Home":
    homepage()
else:
    st.title("RoboNav | Python Navigation")
    st.caption("Streamlit + NumPy + Matplotlib • Real A* and Greedy search")

    if page == "Grid Simulator":
        simulator()
    elif page == "Algorithm Comparison":
        comparison()
    elif page == "Experiment Lab":
        experiment_lab()
    else:
        methodology()
