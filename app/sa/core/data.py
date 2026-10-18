"""
DataManager - Main orchestrator class for SWMM model data processing.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import swmmio as sw

from .conduits import ConduitFeatureEngineeringService, RecommendationService
from .enums import RecommendationCategory
from .nodes import NodeFeatureEngineeringService
from .predictor import GNN_CONFIG, _LazyModelProxy, get_gnn_recommendation, get_recommendation, gnn_enabled, recommendation
from .round import max_slope, min_slope  # Re-export for backwards compatibility
from .services import HydraulicCalculationsService, SimulationRunnerService, TraceAnalysisService
from .subcatchments import SubcatchmentFeatureEngineeringService

# Re-exports for backwards compatibility
__all__ = [
    "DataManager",
    "HydraulicCalculationsService",
    "ConduitFeatureEngineeringService",
    "SubcatchmentFeatureEngineeringService",
    "NodeFeatureEngineeringService",
    "RecommendationService",
    "TraceAnalysisService",
    "SimulationRunnerService",
    "min_slope",
    "max_slope",
]

logger = logging.getLogger(__name__)


class DataManager(sw.Model):
    """
    Main class that combines various services/classes responsible for different areas
    and manages data flow. Following SOLID principles, its purpose is mainly to orchestrate
    logic rather than implementing all logic in a single class.
    """

    def __init__(self, inp_file_path: str, crs: Optional[str] = None, include_rpt: bool = True, zone: float = 1.2):
        super().__init__(inp_file_path, crs=crs, include_rpt=include_rpt)
        self._frost_zone: float = zone
        self.frost_zone = zone

        # ---------------------------
        # DataFrames from swmmio model
        # ---------------------------
        self.dfs = self._get_df_safe(self.subcatchments.dataframe)
        self.dfn = self._get_df_safe(self.nodes.dataframe)
        # Fix for pandas merge error - ensure consistent column types
        conduits_df = self.conduits()
        if hasattr(conduits_df, "InletNode") and hasattr(conduits_df, "OutletNode"):
            conduits_df["InletNode"] = conduits_df["InletNode"].astype(str)
            conduits_df["OutletNode"] = conduits_df["OutletNode"].astype(str)
        self.dfc = self._get_df_safe(conduits_df)

        # ---------------------------
        # Initialization of services
        # ---------------------------
        self.subcatchment_service = SubcatchmentFeatureEngineeringService(self.dfs, self)
        self.node_service = NodeFeatureEngineeringService(self.dfn, self.dfs)
        self.conduit_service = ConduitFeatureEngineeringService(dfc=self.dfc, dfn=self.dfn, frost_zone=self.frost_zone)

        # MLP Recommendation Service is always available for fallback or experiments
        self.mlp_recommendation_service = RecommendationService(self.dfc, model=recommendation, model_name="MLP")
        logger.info("MLP Recommendation Service initialized.")

        # Default the active service to MLP so recommendations() can never run against an
        # unset service if post-simulation reinitialization fails; calculate() upgrades this
        # to the GNN (or a fresh MLP) once the model is reinitialized with the report file.
        self.recommendation_service = self.mlp_recommendation_service
        self.fallback_reason: Optional[str] = None

        # Store the GNN model if it's available, but don't initialize a service here
        self.gnn_model = get_gnn_recommendation() if gnn_enabled() else None
        if self.gnn_model is not None:
            logger.info("GNN model is available and loaded.")
        else:
            logger.info("GNN model is not available.")

        self.simulation_service = SimulationRunnerService(self.inp.path)
        self.trace_analysis_service = TraceAnalysisService(self)

    @property
    def frost_zone(self) -> float:
        return self._frost_zone

    @frost_zone.setter
    def frost_zone(self, value: float) -> None:
        if not (0.8 <= value <= 1.6):
            raise ValueError("Frost zone must be between 0.8 and 1.6 meters")
        self._frost_zone = value

    def _get_df_safe(self, df_source):
        """
        Safely retrieve DataFrame from swmmio object or existing DataFrame.
        """
        df = getattr(df_source, "dataframe", df_source)
        if df is None:
            raise ValueError("DataFrame source is None")
        return df.copy()

    def _ensure_conduit_node_columns_are_strings(self) -> None:
        if self.dfc is None:
            return
        for column in ("InletNode", "OutletNode"):
            if column in self.dfc.columns:
                self.dfc[column] = self.dfc[column].astype(str)

    def _join_node_report_results(self) -> None:
        """Attach simulated node metrics that ``swmmio.Model.nodes()`` omits.

        ``nodes()`` combines node definitions from the INP file, but it does not
        include the node summary tables stored only in the simulation report.
        """
        rpt_path = getattr(getattr(self, "rpt", None), "path", None)
        if self.dfn is None or not rpt_path or not Path(rpt_path).is_file():
            return

        node_names = (
            self.dfn["Name"].astype(str)
            if "Name" in self.dfn.columns
            else pd.Series(self.dfn.index.astype(str), index=self.dfn.index)
        )

        try:
            depth_summary = sw.dataframe_from_rpt(rpt_path, "Node Depth Summary")
            flooding_summary = sw.dataframe_from_rpt(rpt_path, "Node Flooding Summary")
        except (OSError, ValueError, KeyError, IndexError) as exc:
            logger.warning("Could not read node results from report %s: %s", rpt_path, exc)
            return

        if "MaxNodeDepth" in depth_summary.columns:
            max_depth_by_name = depth_summary["MaxNodeDepth"].rename(index=str)
            self.dfn["MaxNodeDepth"] = node_names.map(max_depth_by_name)

        flooding_columns = ("HoursFlooded", "TotalFloodVol")
        if all(column in flooding_summary.columns for column in flooding_columns):
            flooding_summary = flooding_summary.rename(index=str)
            for column in flooding_columns:
                values_by_name = flooding_summary[column]
                self.dfn[column] = node_names.map(values_by_name).fillna(0.0)

    def _select_recommendation_service(self) -> None:
        gnn_is_enabled = gnn_enabled()
        gnn_model = get_gnn_recommendation() if gnn_is_enabled else None
        adjacency = None
        conduit_order = None
        fallback_reason: Optional[str] = None

        if not gnn_is_enabled:
            fallback_reason = "GNN_ENABLED=false"
        elif gnn_model is None:
            fallback_reason = "GNN model unavailable"

        if gnn_model is not None:
            try:
                self._ensure_conduit_node_columns_are_strings()
                spec = getattr(gnn_model, "spec", None)
                try:
                    from stormlab.models.gnn_spec import GnnSpec, GnnSpecC1

                    is_stormlab_spec = isinstance(spec, (GnnSpec, GnnSpecC1))
                except ImportError:
                    is_stormlab_spec = False

                if is_stormlab_spec and spec is not None:
                    from stormlab.graph.order import canonical_order
                    from stormlab.graph.build import build_graph
                    from stormlab.graph.khop import operator_adjacency

                    conduit_order = canonical_order(self.dfc, "inp")
                    graph = build_graph(self.dfc, conduit_order, direction="downstream")
                    source = graph.adjacency.transpose().tocsr() if spec.direction == "upstream" else graph.adjacency
                    adjacency = operator_adjacency(source, operator=spec.operator, max_hops=spec.hops)
                else:
                    from .gnn import build_adjacency_from_dfc, preprocess_adjacency

                    raw_adjacency = build_adjacency_from_dfc(self.dfc)
                    adjacency = preprocess_adjacency(raw_adjacency, max_hops=GNN_CONFIG["max_hops"])
            except Exception as e:
                logger.warning("GNN graph construction failed -> MLP fallback: %s", e)
                fallback_reason = f"graph_construction_error: {e}"
                gnn_model, adjacency, conduit_order = None, None, None

        rpt_path = getattr(getattr(self, "rpt", None), "path", None)
        inp_path = getattr(getattr(self, "inp", None), "path", None)
        if not rpt_path and inp_path:
            candidate_rpt = inp_path.replace(".inp", ".rpt")
            if os.path.isfile(candidate_rpt):
                rpt_path = candidate_rpt

        self.fallback_reason = fallback_reason

        if gnn_model is not None and adjacency is not None:
            logger.info("Using GNN model for recommendations (GraphSAGE)")
            self.recommendation_service = RecommendationService(
                self.dfc,
                model=gnn_model,
                model_name="GNN",
                adjacency=adjacency,
                conduit_order=conduit_order,
                inp_path=inp_path,
                rpt_path=rpt_path,
            )
        else:
            logger.info("Using MLP model for recommendations (fallback - GNN unavailable or disabled)")
            self.recommendation_service = RecommendationService(
                self.dfc,
                model=get_recommendation(),
                model_name="MLP",
                inp_path=inp_path,
                rpt_path=rpt_path,
            )

    def __enter__(self):
        self.calculate()
        self.feature_engineering()
        self.recommendations()
        self._round_float_columns()
        self._drop_unused()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Exception occurred: {exc_val}")
        return False

    ############################################################################
    #                   HELPER METHODS (round, drop, etc.)
    ############################################################################
    def _round_float_columns(self) -> None:
        """
        Rounds float columns in dfs, dfn, and dfc to 2 decimal places.
        Excludes confidence columns and node report metrics whose source
        precision must be preserved for persistence.
        """
        confidence_cols = [f"confidence_{cat.value}" for cat in RecommendationCategory]
        report_node_cols = {"MaxNodeDepth", "HoursFlooded", "TotalFloodVol"}

        for df in [self.dfs, self.dfn, self.dfc]:
            if df is not None:
                float_cols = df.select_dtypes(include=["float"]).columns
                # Preserve model confidences and the report's source precision.
                cols_to_round = [col for col in float_cols if col not in confidence_cols and col not in report_node_cols]
                if cols_to_round:
                    df[cols_to_round] = df[cols_to_round].round(2)

    def _drop_unused(self) -> None:
        """
        Removes unused columns from DataFrames to maintain cleanliness.
        """
        unused_conduits = [
            "OutOffset",
            "InitFlow",
            "Barrels",
            "Shape",
            "InOffset",
            "coords",
            "Geom2",
            "SlopeFtPerFt",
            "Type",
        ]
        unused_nodes = ["coords", "StageOrTimeseries"]
        unused_subcatchments = ["coords"]

        if self.dfc is not None:
            self.dfc.drop(columns=unused_conduits, inplace=True, errors="ignore")
        if self.dfn is not None:
            self.dfn.drop(columns=unused_nodes, inplace=True, errors="ignore")
        if self.dfs is not None:
            self.dfs.drop(columns=unused_subcatchments, inplace=True, errors="ignore")

    ############################################################################
    #                              MAIN OPERATIONS
    ############################################################################
    def calculate(self) -> None:
        """
        Runs the simulation to update model values.
        """
        self.simulation_service.run_simulation()

        # Reinitialize parent sw.Model to include newly generated .rpt file
        reinitialized = False
        try:
            super().__init__(self.inp.path, crs=self.crs, include_rpt=True)

            # Update DataManager's DataFrames with new data from reinitialized model
            new_conduits_df = self.conduits()
            if new_conduits_df is not None and not new_conduits_df.empty:
                self.dfc = new_conduits_df.copy()
                self._ensure_conduit_node_columns_are_strings()

            new_nodes_df = self.nodes()
            if new_nodes_df is not None and not new_nodes_df.empty:
                self.dfn = new_nodes_df.copy()
                self._join_node_report_results()

            new_subcatch_df = self.subcatchments()
            if new_subcatch_df is not None and not new_subcatch_df.empty:
                self.dfs = new_subcatch_df.copy()

            # Reinitialize services with updated DataFrames
            self.subcatchment_service = SubcatchmentFeatureEngineeringService(self.dfs, self)
            self.node_service = NodeFeatureEngineeringService(self.dfn, self.dfs)
            self.conduit_service = ConduitFeatureEngineeringService(dfc=self.dfc, dfn=self.dfn, frost_zone=self.frost_zone)

            reinitialized = True

        except Exception as e:
            logger.warning(f"Could not reinitialize with report file: {e}")

        self.reinitialized_with_rpt = reinitialized
        if reinitialized:
            self._select_recommendation_service()

    def feature_engineering(self) -> None:
        """
        Calls individual feature engineering stages for subcatchments,
        nodes, and conduits.
        """
        # Subcatchments
        self.subcatchment_service.subcatchments_classify(categories=True)

        # Nodes - zmienione z nodes_subcatchment_name() na nodes_subcatchment_info()
        self.node_service.nodes_subcatchment_info()

        # Conduits
        self.conduit_service.calculate_filling()
        self.conduit_service.filling_is_valid()
        self.conduit_service.velocity_is_valid()
        self.conduit_service.slope_per_mile()
        self.conduit_service.max_depth()
        self.conduit_service.calculate_max_depth()
        self.conduit_service.calculate_ground_elevation()
        self.conduit_service.ground_cover()
        self.conduit_service.max_ground_cover_is_valid()
        self.conduit_service.min_ground_cover_is_valid()
        self.conduit_service.min_conduit_diameter()
        self.conduit_service.diameter_features()

        self.conduit_service.conduits_subcatchment_info()
        self.conduit_service.propagate_subcatchment_info()

        self.conduit_service.normalize_roughness()
        self.conduit_service.normalize_max_velocity()
        self.conduit_service.normalize_depth()
        self.conduit_service.normalize_filling()
        self.conduit_service.normalize_max_q()
        self.conduit_service.normalize_ground_cover()
        self.conduit_service.normalize_slope()
        self.conduit_service.slopes_is_valid()
        self.conduit_service.slope_increase()
        self.conduit_service.slope_reduction()
        self.conduit_service.encode_sbc_category()

    def recommendations(self) -> None:
        """
        Generates recommendations using the 'recommendation' model.
        """
        model = getattr(self.recommendation_service, "model", "unset")
        if isinstance(model, _LazyModelProxy):
            model = model._getter()
        if model is None:
            logger.warning("No recommendation model loaded; populating empty recommendation columns.")
            if self.dfc is not None:
                self.dfc["recommendation"] = "-"
                for category in RecommendationCategory:
                    self.dfc[f"confidence_{category.value}"] = 0.0
            return
        self.recommendation_service.recommendations()

    ############################################################################
    #      ROUTING / OVERFLOW / RECOMMENDATION METHODS
    ############################################################################
    def all_traces(self):
        """
        Returns all traces from the SWMM model.
        """
        return self.trace_analysis_service.all_traces()

    def overflowing_pipes(self) -> pd.DataFrame:
        """
        Returns overflowing pipes (ValMaxFill == 0).
        """
        return self.trace_analysis_service.overflowing_pipes()

    def overflowing_traces(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Returns all traces related to overflowing pipes.
        """
        return self.trace_analysis_service.overflowing_traces()

    def place_to_change(self) -> List[str]:
        """
        Determines which nodes require technical intervention.
        """
        return self.trace_analysis_service.place_to_change()
