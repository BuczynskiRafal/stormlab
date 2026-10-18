"""Graph construction utilities for SWMM conduit networks."""

from collections import defaultdict
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)


class SWMMGraphConstructor:
    """
    Constructor for graphs where nodes are conduits and edges are hydraulic connections.
    """

    def __init__(self, conduits_data: pd.DataFrame, conduit_order: list[str] | None = None):
        self.conduits_data = self._prepare_conduits_data(conduits_data, conduit_order)
        self.conduit_to_idx = {}
        self.idx_to_conduit = {}

    def _prepare_conduits_data(self, conduits_data: pd.DataFrame, conduit_order: list[str] | None) -> pd.DataFrame:
        data = conduits_data.copy()
        if "Name" not in data.columns:
            data["Name"] = data.index.astype(str)

        data["Name"] = data["Name"].astype(str)

        if conduit_order is not None:
            order = [str(name) for name in conduit_order]
            if data["Name"].duplicated().any():
                duplicates = data.loc[data["Name"].duplicated(), "Name"].unique().tolist()
                raise ValueError(f"Duplicate conduit names make conduit_order ambiguous: {duplicates}")

            indexed = data.set_index("Name", drop=False)
            missing = [name for name in order if name not in indexed.index]
            if missing:
                raise ValueError(f"conduit_order contains names missing from conduits_data: {missing}")
            data = indexed.loc[order].reset_index(drop=True)

        return data.reset_index(drop=True)

    def build_conduit_graph(self) -> Tuple[sp.csr_matrix, None]:
        """
        Build a raw directed graph where nodes are conduits and edges follow flow direction.
        """
        n_conduits = len(self.conduits_data)
        conduits_reset = self.conduits_data.reset_index(drop=True).copy()
        self.conduit_to_idx = {row["Name"]: idx for idx, row in conduits_reset.iterrows()}
        self.idx_to_conduit = {idx: name for name, idx in self.conduit_to_idx.items()}

        clean_data = conduits_reset.dropna(subset=["InletNode", "OutletNode", "Name"])
        if len(clean_data) < len(conduits_reset):
            dropped = len(conduits_reset) - len(clean_data)
            logger.warning(
                "Removed %s conduits with NaN in node columns (%.1f%%)",
                dropped,
                dropped / len(conduits_reset) * 100 if len(conduits_reset) else 0,
            )

        all_inlet_nodes = set(clean_data["InletNode"].astype(str).str.strip().str.lower())
        all_outlet_nodes = set(clean_data["OutletNode"].astype(str).str.strip().str.lower())
        all_inlet_nodes.discard("nan")
        all_outlet_nodes.discard("nan")
        common_nodes = all_inlet_nodes.intersection(all_outlet_nodes)

        logger.debug(
            "Node diagnostics: conduits=%s/%s, inlet_nodes=%s, outlet_nodes=%s, common_nodes=%s",
            len(clean_data),
            n_conduits,
            len(all_inlet_nodes),
            len(all_outlet_nodes),
            len(common_nodes),
        )

        adjacency = sp.lil_matrix((n_conduits, n_conduits), dtype=np.float32)
        outlet_to_conduits = defaultdict(set)

        for _, row in clean_data.iterrows():
            outlet_node = str(row["OutletNode"]).strip().lower()
            if outlet_node and outlet_node != "nan":
                conduit_idx = self.conduit_to_idx[row["Name"]]
                outlet_to_conduits[outlet_node].add(conduit_idx)

        edge_count = 0
        no_match_count = 0
        added_edges = set()
        connections_debug = []

        for _, row in clean_data.iterrows():
            inlet_node = str(row["InletNode"]).strip().lower()
            current_conduit_name = row["Name"]
            if not inlet_node or inlet_node == "nan":
                no_match_count += 1
                continue

            current_conduit_idx = self.conduit_to_idx[current_conduit_name]
            if inlet_node not in outlet_to_conduits:
                no_match_count += 1
                if no_match_count <= 5:
                    logger.debug("No upstream outlet match for conduit %s via inlet node %s", current_conduit_name, inlet_node)
                continue

            upstream_conduits = outlet_to_conduits[inlet_node]
            for upstream_conduit_idx in upstream_conduits:
                if upstream_conduit_idx == current_conduit_idx:
                    continue

                edge_key = (upstream_conduit_idx, current_conduit_idx)
                if edge_key in added_edges:
                    continue

                adjacency[upstream_conduit_idx, current_conduit_idx] = 1.0
                added_edges.add(edge_key)
                edge_count += 1

                if len(connections_debug) < 10:
                    upstream_name = self.idx_to_conduit[upstream_conduit_idx]
                    connections_debug.append(f"{upstream_name} -> {current_conduit_name} (via {inlet_node})")

        logger.info(
            "Built raw conduit graph: nodes=%s, directed_edges=%s, no_match=%s",
            n_conduits,
            edge_count,
            no_match_count,
        )
        if connections_debug:
            logger.debug("Sample conduit graph connections: %s", connections_debug)
        else:
            logger.warning("No hydraulic conduit connections found in raw graph.")

        adjacency_csr = adjacency.tocsr()
        n_components = self._weak_component_count(adjacency_csr)

        if n_conduits and n_components > 0.9 * n_conduits:
            logger.warning(
                "Graph is highly disconnected (%s weak components for %s conduits); trying fallback connections.",
                n_components,
                n_conduits,
            )
            extra_edges = self._try_spatial_connections(clean_data, adjacency, added_edges)
            edge_count += extra_edges

            if extra_edges < 10:
                extra_edges2 = self._try_feature_based_connections(clean_data, adjacency, added_edges, max_edges=50)
                edge_count += extra_edges2
                logger.info("Added %s feature-similarity fallback connections.", extra_edges2)

            adjacency_csr = adjacency.tocsr()

        adjacency_csr.setdiag(0)
        adjacency_csr.eliminate_zeros()
        csr_mat = sp.csr_matrix(adjacency_csr)
        self._analyze_connectivity_detailed(csr_mat)
        return csr_mat, None

    def _weak_component_count(self, adjacency: sp.csr_matrix) -> int:
        shape = adjacency.shape
        if not shape or shape[0] == 0:
            return 0
        return int(connected_components(adjacency, directed=True, connection="weak", return_labels=False))

    def _analyze_connectivity_detailed(self, adjacency: sp.csr_matrix) -> None:
        """Detailed graph connectivity logging."""
        shape = adjacency.shape
        n_nodes = shape[0] if shape else 0
        if n_nodes == 0:
            logger.info("Connectivity analysis: empty graph.")
            return

        n_components, labels = connected_components(adjacency, directed=True, connection="weak", return_labels=True)
        component_sizes = sorted(np.bincount(labels).tolist(), reverse=True) if labels.size else []
        in_degrees = np.asarray(adjacency.sum(axis=0)).ravel()
        out_degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        isolated = np.count_nonzero((in_degrees == 0) & (out_degrees == 0))
        density = adjacency.nnz / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

        logger.info(
            "Connectivity analysis: components=%s, largest_components=%s, isolated=%s (%.1f%%), density=%.4f",
            n_components,
            component_sizes[:5],
            isolated,
            isolated / n_nodes * 100,
            density,
        )
        logger.debug(
            "Degree statistics: mean_in=%.2f, mean_out=%.2f, max_in=%.2f, max_out=%.2f",
            float(np.mean(in_degrees)) if in_degrees.size else 0.0,
            float(np.mean(out_degrees)) if out_degrees.size else 0.0,
            float(np.max(in_degrees)) if in_degrees.size else 0.0,
            float(np.max(out_degrees)) if out_degrees.size else 0.0,
        )

    def _try_spatial_connections(self, clean_data: pd.DataFrame, adjacency: sp.lil_matrix, added_edges: set) -> int:
        """Attempt to add fallback connections based on spatial proximity."""
        spatial_cols = ["X1", "Y1", "X2", "Y2", "Length"]
        available_spatial = [col for col in spatial_cols if col in clean_data.columns]
        if len(available_spatial) < 2:
            logger.warning("Not enough spatial data available to make spatial fallback connections.")
            return 0

        extra_edges = 0
        if "Length" in clean_data.columns:
            sortable_data = clean_data.assign(_LengthNumeric=pd.to_numeric(clean_data["Length"], errors="coerce"))
            clean_data_sorted = sortable_data.dropna(subset=["_LengthNumeric"]).sort_values("_LengthNumeric")

            for i in range(len(clean_data_sorted) - 1):
                row1 = clean_data_sorted.iloc[i]
                row2 = clean_data_sorted.iloc[i + 1]
                length1 = row1["_LengthNumeric"]
                length2 = row2["_LengthNumeric"]

                if abs(length1 - length2) / max(length1, 1) < 0.2:
                    idx1 = self.conduit_to_idx[row1["Name"]]
                    idx2 = self.conduit_to_idx[row2["Name"]]
                    if idx1 == idx2:
                        continue

                    edge_key = (idx1, idx2)
                    if edge_key not in added_edges:
                        adjacency[idx1, idx2] = 0.5
                        added_edges.add(edge_key)
                        extra_edges += 1

                        if extra_edges >= 20:
                            break

        logger.info("Added %s spatial fallback connections.", extra_edges)
        return extra_edges

    def _try_feature_based_connections(
        self, clean_data: pd.DataFrame, adjacency: sp.lil_matrix, added_edges: set, max_edges: int = 50
    ) -> int:
        """Add fallback connections based on conduit feature similarity."""
        feature_cols = ["Diameter", "Length", "Roughness"]
        available_features = [col for col in feature_cols if col in clean_data.columns]
        if not available_features or len(clean_data) < 2:
            logger.warning("No suitable features available for fallback comparison.")
            return 0

        normalized_data = clean_data[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        for col in available_features:
            if normalized_data[col].std() > 0:
                normalized_data[col] = (normalized_data[col] - normalized_data[col].mean()) / normalized_data[col].std()

        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(normalized_data.values))
        extra_edges = 0

        for i in range(len(clean_data)):
            for j in range(i + 1, len(clean_data)):
                if distances[i, j] >= 0.5:
                    continue

                row1 = clean_data.iloc[i]
                row2 = clean_data.iloc[j]
                idx1 = self.conduit_to_idx[row1["Name"]]
                idx2 = self.conduit_to_idx[row2["Name"]]
                if idx1 == idx2:
                    continue

                edge_key1 = (idx1, idx2)
                edge_key2 = (idx2, idx1)

                if edge_key1 not in added_edges:
                    adjacency[idx1, idx2] = 0.3
                    added_edges.add(edge_key1)
                    extra_edges += 1

                if edge_key2 not in added_edges and extra_edges < max_edges:
                    adjacency[idx2, idx1] = 0.3
                    added_edges.add(edge_key2)
                    extra_edges += 1

                if extra_edges >= max_edges:
                    return extra_edges

        return extra_edges


def build_adjacency_from_dfc(dfc: pd.DataFrame, conduit_order: list[str] | None = None) -> sp.csr_matrix:
    """Build raw conduit adjacency in dfc row order unless conduit_order is provided."""
    data = dfc.copy()
    if "Name" not in data.columns:
        data["Name"] = data.index.astype(str)

    constructor = SWMMGraphConstructor(data, conduit_order=conduit_order)
    adjacency, _ = constructor.build_conduit_graph()
    return adjacency
