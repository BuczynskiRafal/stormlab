# StormLab — Stormwater Network Analysis & Validation Platform

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**StormLab** is an engineering and research platform for hydraulic simulation, standards-based validation, and interactive visual diagnostics of gravity storm sewer networks built on **EPA SWMM 5.2** and **Django 5.1**.

> **Note on Doctoral Research Artifacts:**  
> This repository hosts the public source code of the StormLab web application and hydraulic analysis engine developed as part of the doctoral dissertation by **Rafał Buczyński**. The synthetic SWMM network datasets, reproducible training pipeline, experimental benchmarks, and pretrained Graph Neural Network (GraphSAGE) / MLP model weights will be published in this repository upon completion of the formal dissertation publication process.

---

## Key Capabilities

- **Automated EPA SWMM 5.2 Simulation & Parsing:**  
  Uploads and validates `.inp` sewer network models (requiring metric `FLOW_UNITS = CMS`), executes simulations in isolated temporary workspaces via `pyswmm`, and extracts conduit, node, and subcatchment metrics from `.inp` and `.rpt` reports.
- **Standards-Compliant Hydraulic & Geometric Validation:**  
  Evaluates every conduit against **PN-EN 752:2008** and **PN-81/B-03020** engineering criteria:
  - Maximum filling ratio $h/D$ as a function of diameter (`app/sa/core/round.py::max_filling`),
  - Minimum and maximum flow velocity bounds ($0.8 \le v \le 5.0\text{ m/s}$),
  - Minimum self-cleansing slope derived from the tractive shear stress condition $\tau = 1.5\text{ Pa}$ (`app/sa/core/round.py::min_slope`) and maximum allowable slope (`max_slope`),
  - Ground cover and frost-zone depth verification ($0.8\text{--}1.6\text{ m}$),
  - Simple operational band check ($0.2 \le h/D \le 0.7$ and $0.8 \le v \le 3.0\text{ m/s}$).
- **Interactive SVG Network Map:**  
  Renders the complete sewer network topology (junctions, outfalls, storage units, conduits, and subcatchment links) with automatic force-directed fallback layout when node coordinates are absent, interactive pan/zoom, hydraulic status highlighting, and per-element inspection cards.
- **Server-Rendered Longitudinal Conduit Profiles:**  
  Generates engineering longitudinal profile SVG diagrams (`viewBox 0 0 880 340`) displaying ground surface elevation, frost protection depth, pipe crown and invert levels, water surface profile, and adjacent upstream/downstream conduit navigation.
- **Pluggable ML / GNN Recommendation Architecture:**  
  Includes the inference engine for 9-class hydraulic rehabilitation recommendations (`valid`, `diameter_increase`, `diameter_reduction`, `slope_increase`, `slope_reduction`, `depth_increase`, `tank`, `seepage_boxes`, `pump`) and 12-class subcatchment classification. When pretrained model weights are not installed locally, the platform runs in full hydraulic validation and visualization mode.
- **Session History & CSV Export:**  
  Persists calculation sessions, supports side-by-side session comparison, and exports full conduit hydraulic and validation tables to CSV.

---

## Repository Structure

| Path | Description |
|---|---|
| `app/` | Django web application (`storm_analysis`, `sa`, `accounts`), templates, static assets (`app/static/`), and Docker configuration. |
| `app/sa/core/` | Core hydraulic calculations (`round.py`, `valid_round.py`, `services.py`), feature engineering (`conduits.py`, `nodes.py`, `subcatchments.py`), network geometry extraction (`geometry.py`), and GNN/MLP model serving (`graph_constructor.py`, `predictor.py`). |
| `app/sa/core/tests/` | `pytest` unit and integration suite for hydraulic calculations, feature contracts, graph construction, and model serving. |
| `app/sa/tests/` | Django view, service, form, model, and Node.js frontend harness tests. |

---

## Quick Start (Local Development)

1. **Create and activate a Python 3.12 virtual environment:**
   ```bash
   python3.12 -m venv venv
   ./venv/bin/pip install -r requirements.txt -r app/requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp app/.env.example app/.env
   ```

3. **Apply database migrations and start the development server:**
   ```bash
   make migrate
   make run
   ```
   Open `http://127.0.0.1:8000/` in your browser.

---

## Running with Docker

```bash
cd app
docker compose up --build
```

---

## Running Tests & Static Analysis

- **Core hydraulic & graph unit tests (`pytest`):**
  ```bash
  make test-core
  ```
- **Django web application & frontend harness tests:**
  ```bash
  make test-django
  ```
- **Linting & static type checking (`ruff` & `pyright`):**
  ```bash
  make lint
  make typecheck
  ```

---

## License & Citation

Licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.  
Citation metadata is provided in [CITATION.cff](CITATION.cff).
