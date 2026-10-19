# Example EPA SWMM 5.2 Networks (`.inp`)

This directory contains ready-to-upload EPA SWMM 5.2 storm sewer network models (`FLOW_UNITS = CMS`) for testing and exploring **StormLab** in the web application (`http://127.0.0.1:8000/analysis/`) or via the Python API.

| File | Conduits | Nodes | Subcatchments | Geometry Mode | Description |
|---|---:|---:|---:|---|---|
| [`demo_network_coordinates.inp`](demo_network_coordinates.inp) | 21 | 23 | 7 | Geographic (`[COORDINATES]` & `[Polygons]`) | Full geographic storm sewer network with two collector branches (`O1`, `O2`), circular pipes ($\varnothing 300\text{--}800\text{ mm}$), subcatchment polygons, and land-cover tags. |
| [`demo_network_schematic.inp`](demo_network_schematic.inp) | 21 | 23 | 7 | Automatic Topological (`auto_layout`) | Identical hydraulic topology without `[COORDINATES]` or `[Polygons]` tables, demonstrating StormLab's automatic topological graph layout (*Schematic layout — not model geometry*). |

## Quick Usage in the Web UI

1. Start the application (`make run` or `docker compose up` inside `app/`).
2. Open `http://127.0.0.1:8000/analysis/` (log in or register a local account).
3. Select `examples/demo_network_coordinates.inp`, choose the frost-zone depth (e.g., `1.2 m`), and click **Run Analysis**.
4. Explore the **Interactive Network Map** (`Checks passed` and `Filling` modes), inspect the **Conduits / Nodes / Subcatchments** tables, export the session to CSV, or click any conduit to view its **Longitudinal Profile SVG** diagram.
