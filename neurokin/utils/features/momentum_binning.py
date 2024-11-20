import pandas as pd
import numpy as np
from typing import List, Dict, Any
from neurokin.utils.features.core import FeatureExtraction


class SpeedBinning(FeatureExtraction):
    """
    Computes the velocity of bodyparts seperated in the three dimensions -> 3 values (x,y,z) based on sliding window binning
    Input: df with positon data (i.e. DLC output), source_marker_ids: List of markers for which velocity should be computed
    Output: df with velocity data for input markers
    Binning method must be defined, either 'mean' or 'median'
    """

    @property
    def input_type(self) -> str:
        return "markers"

    @property
    def default_values(self) -> Dict[str, Any]:
        default_values = {"window_size": 3, "fps": 80}
        return default_values

    @property
    def default_value_types(self) -> Dict[str, List[type]]:
        default_types = {"window_size": [int], "fps": [int]}
        return default_types

    def _run_feature_extraction(
        self,
        source_marker_ids: List[str],
        marker_df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.DataFrame:

        coords_df = self._copy_filtered_columns_of_df(
            df_to_filter=marker_df,
            marker_id_filter=source_marker_ids,
            coords_filter=["x", "y", "z"],
        )

        if params['smoothing_method'] not in ['mean', 'median']:
            raise ValueError("Smoothing method must be 'mean' or 'median'")

            # Apply sliding window smoothing
        if params['smoothing_method'] == 'mean':
            smoothed_df = coords_df.rolling(window=params["window_size"], min_periods=1, center=True).mean()
        elif params['smoothing_method'] == 'median':
            smoothed_df = coords_df.rolling(window=params["window_size"], min_periods=1, center=True).median()
        else: raise ValueError("Smoothing method must be 'mean' or 'median'")

        # Compute distances in smoothed dataframe
        distances = np.sqrt((smoothed_df.diff() ** 2).sum(axis=1))

        #Convert to cm/s assuming distance is in cm, since we compare row by row, we technically divide by 1
        speeds = distances * params["fps"]

        df_speed = self.convert_singleindex_to_multiindex_df(
                scorer="scorer",
                bodypart=source_marker_ids,
                axis="speed",
                data=speeds)

        return df_speed