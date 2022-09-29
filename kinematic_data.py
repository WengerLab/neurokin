import numpy as np
from numpy.typing import ArrayLike
from utils.kinematics import c3d_import_export, event_detection, kinematics_processing
from utils.helper import load_config
import pandas as pd
from matplotlib import pyplot as plt


class KinematicDataRun:
    """
    This class represents the kinematics data recorded in a single run.
    """

    def __init__(self, path, configpath):
        self.path = path

        self.config = load_config.read_config(configpath)

        self.gait_cycles_start: ArrayLike
        self.gait_cycles_end: ArrayLike
        self.fs: float  #TODO set fs
        self.condition: str

        self.markers_df: pd.DataFrame
        self.gait_param = pd.DataFrame()

        self.left_mtp_lift: ArrayLike = None
        self.left_mtp_land: ArrayLike = None
        self.left_mtp_max: ArrayLike = None
        self.right_mtp_lift: ArrayLike = None
        self.right_mtp_land: ArrayLike = None
        self.right_mtp_max: ArrayLike = None

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

        self.markers_df = c3d_import_export.import_c3d(self.path)

        if correct_shift:

            if shift_reference_marker not in self.markers_df.columns.tolist():
                raise ValueError("The shift reference marker " + shift_reference_marker + " is not among the markers."
                                 + "\n Please select one among the following: \n" +
                                 self.markers_df.columns.tolist())

            if not set(to_shift).issubset(self.markers_df.columns.tolist()):
                raise ValueError("Some or all columns to shift are not among the markers. You selected: \n"
                                 + " ,".join(str(x) for x in to_shift)
                                 + "\n Please select them among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            self.markers_df = kinematics_processing.shift_correct(self.markers_df, shift_reference_marker, to_shift)

        if correct_tilt:

            if tilt_reference_marker not in self.markers_df.columns.tolist():
                raise ValueError("The tilt reference marker " + tilt_reference_marker + " is not among the markers."
                                 + "\n Please select one among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            if not set(to_tilt).issubset(self.markers_df.columns.tolist()):
                raise ValueError("Some or all columns to tilt are not among the markers. You selected: \n"
                                 + " ,".join(str(x) for x in to_tilt)
                                 + "\n Please select them among the following: \n" +
                                 ", ".join(str(x) for x in self.markers_df.columns.tolist()))

            self.markers_df = kinematics_processing.tilt_correct(self.markers_df, tilt_reference_marker, to_tilt)
        return

    def compute_gait_cycles_bounds(self, left_marker, right_marker, recording_fs):
        """
        Computes the lifting and landing frames of both feet using a left and a right marker, respectively.
        To increase robustness of the cycle estimation it first low-passes the signal.
        :param left_marker: reference marker for the left foot, typically the left mtp
        :param right_marker: reference marker for the left foot, typically the right mtp
        :param recording_fs: sample frequency of the recording, used for low-passing.
        :return:
        """

        if left_marker not in self.markers_df.columns.tolist():
            raise ValueError("The left reference marker " + left_marker + " is not among the markers."
                             + "\n Please select one among the following: \n" +
                             ", ".join(str(x) for x in self.markers_df.columns.tolist()))

        if right_marker not in self.markers_df.columns.tolist():
            raise ValueError("The right reference marker " + right_marker + " is not among the markers."
                             + "\n Please select one among the following: \n" +
                             ", ".join(str(x) for x in self.markers_df.columns.tolist()))

        self.left_mtp_lift, self.left_mtp_land, self.left_mtp_max = event_detection.get_toe_lift_landing(
            self.markers_df[left_marker], recording_fs)
        self.right_mtp_lift, self.right_mtp_land, self.right_mtp_max = event_detection.get_toe_lift_landing(
            self.markers_df[right_marker], recording_fs)

        return

    def print_step_partition(self):
        filename_l = self.path.split("/")[-1] + "_left_step.png"
        step_trace_l = self.markers_df["lmtp_y"]
        plt.plot(step_trace_l)
        plt.vlines(self.left_mtp_lift, min(step_trace_l), max(step_trace_l), colors="green")
        plt.vlines(self.left_mtp_land, min(step_trace_l), max(step_trace_l), colors="red")
        plt.savefig(filename_l)
        plt.close()

        filename_r = self.path.split("/")[-1] + "_right_step.png"
        step_trace_r = self.markers_df["rmtp_y"]
        plt.plot(step_trace_r)
        plt.vlines(self.right_mtp_lift, min(step_trace_r), max(step_trace_r), colors="green")
        plt.vlines(self.right_mtp_land, min(step_trace_r), max(step_trace_r), colors="red")
        plt.savefig(filename_r)
        plt.close()

    def compute_angles_joints(self):
        """
        It refers to the joints listed in the config under angles > joints to set a corresponding column in the
        gait_param dataset. It should be able to support both 3d and 2d spaces.
        :return:
        """
        for key, value in self.config["angles"]["joints"].items():
            names = kinematics_processing.get_marker_coordinates_names(self.markers_df.columns.tolist(), value)
            angle = []
            for frame in range(len(self.markers_df)):
                coordinates_3d = []
                for name in names:
                    values = kinematics_processing.get_marker_coordinate_values(self.markers_df, name, frame)
                    coordinates_3d.append(values)
                coordinates_3d = np.asarray(coordinates_3d)
                angle.append(kinematics_processing.compute_angle(coordinates_3d))
            parameter = pd.Series(angle)
            self.gait_param[key] = parameter
        return

    def gait_param_to_csv(self):
        """
        Writes the gait_param dataframe to a csv file with the name [INPUT_FILENAME]+_gait_param.csv
        :return:
        """
        self.gait_param.to_csv(self.path.split("/")[-1] + "_gait_param.csv")
        return