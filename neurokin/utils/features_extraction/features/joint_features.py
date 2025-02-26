from typing import Dict

import numpy as np
import pandas as pd
from typeguard import typechecked

from neurokin.utils.features_extraction.commons import angle
from neurokin.utils.features_extraction.core import FeatureExtraction


class Angle(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        for joint, bodyparts in target_bodyparts.items():
            target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                     marker in coord]
            feat = angle(df[target_markers_coords].values)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_angle"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat


class AngleVelocity(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, fs: float, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        for joint, bodyparts in target_bodyparts.items():
            target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                     marker in coord]
            feat = self.angle_velocity(df[target_markers_coords].values, fs)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_angle_velocity"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat

    def angle_velocity(self, vectors, fs):
        angles = angle(vectors)
        angle_velocity = np.gradient(angles, 1 / fs)
        return angle_velocity


class AngleAcceleration(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, fs: float, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        for joint, bodyparts in target_bodyparts.items():
            target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                     marker in coord]
            feat = self.angle_acceleration(df[target_markers_coords].values, fs)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_angle_acceleration"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat

    def angle_acceleration(self, vectors, fs):
        angles = angle(vectors)
        angle_velocity = np.gradient(angles, 1 / fs)
        angle_acceleration = np.gradient(angle_velocity, 1 / fs)

        return angle_acceleration


class AngleCorrelation(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        for joint, bodyparts in target_bodyparts.items():
            target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                     marker in coord]
            feat = self.angle_correlation(df[target_markers_coords].values)
            df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_angle_correlation"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat

    def angle_correlation(self, vectors):
        angles = angle(vectors)
        angle_correlation = np.corrcoef(angles)
        return angle_correlation


class CustomJointFeatures(FeatureExtraction):
    extraction_target = "joints"

    @typechecked
    def compute_feature(self, df: pd.DataFrame, target_bodyparts: Dict, **kwargs):
        bodyparts_coordinates = df.columns.tolist()
        df_feat_list = []
        df_feat = pd.DataFrame()
        try:
            feature_names = kwargs.get("feature_names")
        except KeyError:
            raise KeyError(f"No name found for the Custom Join Function. "
                           f"Please add valid feature_names in the config file.")
        for name in feature_names:
            try:
                func = kwargs.get("custom_features")[name]
            except KeyError:
                raise KeyError(f"No function found with name: {name}, please provide a valid name and function "
                               f"name when calling extract_features")
            for joint, bodyparts in target_bodyparts.items():
                target_markers_coords = [coord for marker in bodyparts for coord in bodyparts_coordinates if
                                         marker in coord]
                feat = func(df[target_markers_coords].values)
                df_feat_list.append(pd.DataFrame(feat, columns=[f"{joint}_{name}"]))

        if df_feat_list:
            df_feat = pd.concat(df_feat_list, axis=1)
        return df_feat
