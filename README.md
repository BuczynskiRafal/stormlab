# StormLab

[![CI](https://github.com/BuczynskiRafal/stormlab/actions/workflows/main.yaml/badge.svg)](https://github.com/BuczynskiRafal/stormlab/actions/workflows/main.yaml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Decision support for storm sewer design. StormLab runs an EPA SWMM simulation of a
network, checks every conduit against design criteria and uses a graph neural network
to recommend an action for each conduit.

![Network map coloured by the recommended action](docs/images/network_map_recommendations.png)

## What it does

- **Simulation.** Runs EPA SWMM 5.2 on an uploaded `.inp` model.
- **Hydraulic verification.** Seven checks per conduit: filling, velocity, slope, depth and cover.
- **Recommendations.** A graph neural network reads each conduit together with the
  conduits that drain into it and scores nine actions, from *no change* to
  *larger diameter* or *retention tank*.
- **Catchment categories.** Read from the model's tags, or assigned by a classifier.
- **Results.** Interactive network map, tables with CSV export and a longitudinal
  profile for every conduit.

## How it works

```mermaid
flowchart LR
    inp["SWMM model (.inp)"] --> sim["EPA SWMM 5.2"]
    sim --> checks["7 design checks per conduit"]
    sim --> cats["Catchment categories"]
    sim --> feats["Conduit graph, 66 features per conduit"]
    feats --> gnn["Graph model, 5 runs averaged"]
    gnn --> res["9 action probabilities per conduit"]
    checks --> ui["Map, tables, conduit panel"]
    cats --> ui
    res --> ui
```

Every conduit becomes a node of a directed graph; an edge joins two conduits that meet
in a manhole and points with the flow. For each conduit the model reads its own 66
physical features and a weighted average over the conduits that drain into it, up to
four links upstream. The simulation supplies the hydraulics; the model interprets them.

![What the model reads for one conduit of the Rakowiec network](docs/images/method_receptive_field.png)

On held-out network families the graph model names the main action for 89.3% of
conduits (90.7% with the five runs averaged, as served), against 87.6% for the same
network without neighbours. How the model was trained, which experiments led to it and
where it falls short is described in [docs/method.md](docs/method.md) and, with an
interactive view of the graph, on the application's **Method** page.

## Quick start

Requires Python 3.12.

```bash
python3.12 -m venv venv
make install
cp app/.env.example app/.env
make migrate
make run
```

Open <http://127.0.0.1:8000>, sign up and follow the tour below.

On macOS, `make install` also re-signs the SWMM libraries: the published wheel carries
invalid code signatures and the system refuses to load it otherwise.

## Tour

1. Go to **Analysis**, pick a frost zone and choose **Try with Demo Data**. It loads the
   storm sewer network of the Rakowiec estate in Wrocław (62 conduits).
2. On the **Map** tab, switch between recommendations, passed checks and filling. Click a
   conduit, node or subcatchment for its key values.
3. Browse the conduit, node and subcatchment tables, or export them to CSV.
4. Open a conduit's details to see its profile, the seven checks and the recommended
   action with the model's score for every class.
5. Open **Method** to see, on the same network, which conduits the model reads for the
   one you select.
6. Upload your own `.inp` file (flow units: CMS). More examples are in [`examples/`](examples/).

| Filling | Conduit panel |
|---|---|
| ![Network map coloured by filling](docs/images/network_map_filling.png) | ![Conduit profile, checks and recommendation](docs/images/conduit_panel.png) |

## Models

The repository ships the trained models the application uses: the recommendation
network (five training runs, averaged) and the catchment classifier. Training code is
not part of this repository.

## Documentation

| Document | Content |
|---|---|
| [docs/method.md](docs/method.md) | The method: conduit graph, features, model, data, training, experiments, results and limits |
| [docs/architecture.md](docs/architecture.md) | The code: components, path of one analysis, recommendation path, data model, tests |
| [docs/deployment.md](docs/deployment.md) | Production setup with Docker Compose |
| [examples/README.md](examples/README.md) | Example networks |

## Deployment

A production image and a Docker Compose setup are included; see
[docs/deployment.md](docs/deployment.md).

## Development

```bash
make test        # unit, contract and Django tests
make lint        # ruff
make typecheck   # pyright
```

Changes go through topic branches merged into `develop`; `main` holds released states.

## License and citation

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
To cite the software, use [CITATION.cff](CITATION.cff).
