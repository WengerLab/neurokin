from numpy.typing import ArrayLike
from neurokin.utils.kinematics import kinematics_processing, c3d_import_export, event_detection
from neurokin.utils.helper import load_config
from neurokin.utils.features import features_extraction, binning
import pandas as pd
from matplotlib import pyplot as plt
from dlc2kinematics.preprocess import smooth_trajectory


class KinematicDataRun:
    """
    This class represents the kinematics data recorded in a single run.
    """

    def __init__(self, path, configpath):
        self.path = path

        self.config = load_config.read_config(configpath)

        self.trial_roi_start: int
        self.trial_roi_end: int
        self.fs: float
        self.condition: str

        self.markers_df = pd.DataFrame()
        self.gait_param = pd.DataFrame()
        self.stepwise_gait_features = pd.DataFrame()
        self.features_df: pd.MultiIndex = None

        self.left_mtp_lift: ArrayLike = None
        self.left_mtp_land: ArrayLike = None
        self.left_mtp_max: ArrayLike = None
        self.right_mtp_lift: ArrayLike = None
        self.right_mtp_land: ArrayLike = None
        self.right_mtp_max: ArrayLike = None
        self.bodyparts: ArrayLike = None
        self.scorer: str = "scorer"

    def load_kinematics(self,
                        correct_shift: bool = False,
                        correct_tilt: bool = False,
                        to_shift: ArrayLike = None,
                        to_tilt: ArrayLike = None,
                        shift_reference_marker: str = "",
                        tilt_reference_marker: str = ""):
        """
        Loads the kinematics from a c3d file into a dataframe with timeframes as rows and markers as columns

        :param correct_shift: bool should there be a correction in the shift of one of the axis?
        :param correct_tilt: bool should there be a correction in the tilt (linear trend) of one of the axis?
        :param to_shift: which columns to perform the shift on if correct_shift is true
        :param to_tilt: which columns to perform the tilt on if correct_shift is true
        :param shift_reference_marker: which marker to use as a reference trajectory to compute the shift
        :param tilt_reference_marker: which marker to use as a reference trajectory to compute the tilt
        :return:
        """

        self.trial_roi_start, self.trial_roi_end, self.fs, self.markers_df = c3d_import_export.import_c3d(self.path)

        if correct_shift:

            if shift_reference_marker not in self.markers_df.columns.get_level_values("bodyparts").tolist():
                raise ValueError("The shift reference marker " + shift_reference_marker + " is not among the markers."
                                 + "\n Please select one among the following: \n" +
                                 self.markers_df.columns.tolist())

            if not set(to_shift).issubset(self.markers_df.columns.get_level_values("bodyparts").tolist()):
                raise ValueError("Some or all columns to shift are not among the markers. You selected: \n"
                                 + " ,".join(str(x) for x in to_shift)
                                 + "\n Please select them among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            self.markers_df = kinematics_processing.shift_correct(self.markers_df, shift_reference_marker, to_shift)

        if correct_tilt:

            if tilt_reference_marker not in self.markers_df.columns.get_level_values("bodyparts").tolist():
                raise ValueError("The tilt reference marker " + tilt_reference_marker + " is not among the markers."
                                 + "\n Please select one among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            if not set(to_tilt).issubset(self.markers_df.columns.get_level_values("bodyparts").tolist()):
                raise ValueError("Some or all columns to tilt are not among the markers. You selected: \n"
                                 + " ,".join(str(x) for x in to_tilt)
                                 + "\n Please select them among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            self.markers_df = kinematics_processing.tilt_correct(self.markers_df, tilt_reference_marker, to_tilt)

        return

    def get_c3d_compliance(self, smooth=False, filter_window=3, order=1):
        """
        Converts a DeepLabCut-like dataframe (MultiIndex) to a simple pandas dataframe. Saves the scorer identity and
        the bodyparts in the corresponding class attributes.

        :param smooth: whether to smooth the coordinates or not
        :param filter_window: how big should the filter be
        :param order: order of the polynomial to fit the data
        :return:
        """
        df_ = self.markers_df
        bodyparts = df_.columns.get_level_values("bodyparts").unique().to_list()
        scorer = df_.columns.get_level_values(0)[0]
        if smooth:
            df_ = smooth_trajectory(
                df_,
                bodyparts,
                filter_window,
                order,
                deriv=0,
                save=False,
                output_filename=None,
                destfolder=None,
            )

        self.markers_df = df_
        self.bodyparts = bodyparts
        self.scorer = scorer
        return

    def compute_gait_cycles_bounds(self, left_marker, right_marker, axis="z"):
        """
        Computes the lifting and landing frames of both feet using a left and a right marker, respectively.
        To increase robustness of the cycle estimation it first low-passes the signal.

        :param left_marker: reference marker for the left foot, typically the left mtp
        :param right_marker: reference marker for the right foot, typically the right mtp
        :param recording_fs: sample frequency of the recording, used for low-passing.
        :param axis: axis to use to trace the movement and set the cycle bounds
        :return:
        """

        if left_marker not in self.markers_df.columns.get_level_values("bodyparts").unique():
            raise ValueError("The left reference marker " + left_marker + " is not among the markers."
                             + "\n Please select one among the following: \n" +
                             ", ".join(str(x) for x in self.markers_df.columns.get_level_values("bodyparts").unique()))

        if right_marker not in self.markers_df.columns.get_level_values("bodyparts").unique():
            raise ValueError("The right reference marker " + right_marker + " is not among the markers."
                             + "\n Please select one among the following: \n" +
                             ", ".join(str(x) for x in self.markers_df.columns.get_level_values("bodyparts").unique()))

        self.left_mtp_lift, self.left_mtp_land, self.left_mtp_max = event_detection.get_toe_lift_landing(
            self.markers_df[self.scorer][left_marker][axis], self.fs)
        self.right_mtp_lift, self.right_mtp_land, self.right_mtp_max = event_detection.get_toe_lift_landing(
            self.markers_df[self.scorer][right_marker][axis], self.fs)

        return

    def plot_step_partition(self, step_left, step_right, ax_l, ax_r):
        """
        Plots the step partition of a gait trace (ideally toe or foot trace). Fetches the class attributes so the
        steps computation has to happen before

        :param step_left: name of the left trace body part
        :param step_right: name of the right trace body part
        :param ax_l: ax to plot the left trace on
        :param ax_r: ax to plot the right trace on
        :return: axes
        """
        step_trace_l = self.markers_df[self.scorer][step_left]["z"]
        ax_l.plot(step_trace_l)
        ax_l.vlines(self.left_mtp_lift, min(step_trace_l), max(step_trace_l), colors="green")
        ax_l.vlines(self.left_mtp_land, min(step_trace_l), max(step_trace_l), colors="red")
        ax_l.set_title("Left side")

        step_trace_r = self.markers_df[self.scorer][step_right]["z"]
        ax_r.plot(step_trace_r)
        ax_r.vlines(self.right_mtp_lift, min(step_trace_r), max(step_trace_r), colors="green")
        ax_r.vlines(self.right_mtp_land, min(step_trace_r), max(step_trace_r), colors="red")
        ax_r.set_title("Right side")

        ax_l.set_xlim(0, len(step_trace_l))
        ax_r.set_xlim(0, len(step_trace_r))

        return ax_l, ax_r

    def print_step_partition(self, step_left, step_right, output_folder="./"):
        """
        Creates and saves the step partition plot. See plot_step_partition method for details.

        :param step_left: name of the left trace body part
        :param step_right: name of the right trace body part
        :param output_folder: where to save the figure
        :return:
        """
        fig, axs = plt.subplots(2, 1)
        fig.tight_layout(pad=2.0)
        filename = output_folder + self.path.split("/")[-1] + "_steps_partition.png"
        axs[0], axs[1] = self.plot_step_partition(step_left, step_right, axs[0], axs[1])
        plt.savefig(filename, facecolor="white")
        plt.close()

    def extract_features(self):
        """
        Computes features on the markers dataframe based on the config file

        :return:
        """
        features = self.config["features"]
        skeleton = self.config["skeleton"]

        new_features = features_extraction.extract_features(features=features,
                                                            bodyparts=self.bodyparts,
                                                            skeleton=skeleton,
                                                            markers_df=self.markers_df)

        if self.features_df is not None:
            self.features_df = pd.concat((self.features_df, new_features), axis=1)
        else:
            self.features_df = new_features

    def get_binned_features(self, window=50, overlap=25):
        """
        Bins features based on the passed windows and overlap

        :param window: window to compute the binning on
        :param overlap: how much should 2 subsequent windows overlab by
        :return: binned dataframe
        """
        reformat_df = binning.parse_df_features_for_binning(self.markers_df, self.features_df)
        test = binning.get_easy_metrics_on_bins(reformat_df, window=window, overlap=overlap)
        return test

    def get_trace_height(self, marker, axis="z", window=50, overlap=25):
        """
        Computes height traces of a given marker

        :param marker: which marker to consider
        :param axis: x,y, or z, default z
        :param window: window to compute the binning on
        :param overlap: how much should 2 subsequent windows overlab by
        :return:
        """
        test = binning.get_step_height_on_bins(self.markers_df,
                                               window=window,
                                               overlap=overlap,
                                               marker=marker,
                                               axis=axis)
        return test

    def get_step_fwd_movement_on_bins(self, marker, axis="y", window=50, overlap=25):
        """
        Computes forward movement traces of a given marker

        :param marker: which marker to consider
        :param axis: x,y, or z, default z
        :param window: window to compute the binning on
        :param overlap: how much should 2 subsequent windows overlab by
        :return:
        """
        test = binning.get_step_fwd_movement_on_bins(self.markers_df,
                                               window=window,
                                               overlap=overlap,
                                               marker=marker,
                                               axis=axis)
        return test

    def gait_param_to_csv(self, output_folder="./"):
        """
        Writes the gait_param dataframe to a csv file with the name [INPUT_FILENAME]+_gait_param.csv

        :return:
        """
        self.gait_param.to_csv(output_folder + self.path.split("/")[-1].replace(".c3d", "_gait_param.csv"))
        return

    def stepwise_gait_features_to_csv(self, output_folder="./"):
        """
        Writes the gait_param dataframe to a csv file with the name [INPUT_FILENAME]+_gait_param.csv

        :return:
        """
        self.stepwise_gait_features.to_csv(
            output_folder + self.path.split("/")[-1].replace(".c3d", "stepwise_feature.csv"))
        return

    def get_angles_features(self, features_df, **kwargs):
        left_df, right_df = self.split_in_unilateral_df(**kwargs)
        left_features = pd.DataFrame()
        right_features = pd.DataFrame()

        left_features = kinematics_processing.get_angle_features(left_df, self.left_mtp_land,
                                                                 features_df)  # TODO update features df with all angle features

        self.stepwise_gait_features = left_features.join(right_features)

    def split_in_unilateral_df(self, left_side="", right_side="",
                               name_starts_with=False, name_ends_with=False,
                               expected_columns_number=None,
                               left_columns=None, right_columns=None):
        """
        Returns dataframes split by the side, left or right.

        :param left_side:  target side name, left
        :param right_side: target side name, right
        :param name_starts_with: if all columns of a side start with, specify here. E.g. "l_" or "r_"
        :param name_ends_with: if all columns of a side end with, specify here. E.g. "l_" or "r_"
        :param expected_columns_number: expected number of columns, used as a sanity check
        :param left_columns: optional full list of names of left columns
        :param right_columns: optional full list of names of right columns
        :return:
        """
        # self.stepwise_gait_features = 0
        left_df = pd.DataFrame()
        right_df = pd.DataFrame()
        if left_side:
            left_df = kinematics_processing.get_unilateral_df(df=self.gait_param, side=left_side,
                                                              name_starts_with=name_starts_with,
                                                              name_ends_with=name_ends_with,
                                                              column_names=left_columns,
                                                              expected_columns_number=expected_columns_number)

        if right_side:
            right_df = kinematics_processing.get_unilateral_df(df=self.gait_param, side=right_side,
                                                               name_starts_with=name_starts_with,
                                                               name_ends_with=name_ends_with,
                                                               column_names=right_columns,
                                                               expected_columns_number=expected_columns_number)

        return left_df, right_df
