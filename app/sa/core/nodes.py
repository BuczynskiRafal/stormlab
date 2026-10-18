"""
Feature engineering services for nodes.
"""

import pandas as pd


class NodeFeatureEngineeringService:
    """
    Maps subcatchment information to nodes based on the 'Outlet' column from dfs.
    """

    def __init__(self, dfn: pd.DataFrame, dfs: pd.DataFrame):
        self.dfn = dfn
        self.dfs = dfs

    def nodes_subcatchment_name(self) -> None:
        """
        Legacy method that calls nodes_subcatchment_info() for backward compatibility.
        """
        self.nodes_subcatchment_info()

    def nodes_subcatchment_info(self) -> None:
        """
        Maps subcatchment information (ID and category) to nodes based on the 'Outlet' column from dfs.

        Adds two columns to nodes dataframe:
        - Subcatchment: The ID of the connected subcatchment
        - SbcCategory: The category of the connected subcatchment
        """
        if self.dfs is None or self.dfn is None:
            return

        # Initialize columns with default values
        self.dfn["Subcatchment"] = "-"
        self.dfn["SbcCategory"] = "-"

        # Create mappings for subcatchment name and category if columns exist
        if "Outlet" in self.dfs.columns:
            # Get subcatchment ID mapping
            name_mapping = self.dfs.reset_index().set_index("Outlet")["Name"].to_dict()

            # Get subcatchment category mapping if category column exists
            category_mapping = {}
            if "category" in self.dfs.columns:
                category_mapping = self.dfs.reset_index().set_index("Outlet")["category"].to_dict()

            # Assign values to nodes that are outlets for subcatchments
            for node_id in self.dfn.index:
                if node_id in name_mapping:
                    self.dfn.at[node_id, "Subcatchment"] = name_mapping[node_id]
                    if node_id in category_mapping:
                        self.dfn.at[node_id, "SbcCategory"] = category_mapping[node_id]
