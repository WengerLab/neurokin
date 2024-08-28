from typing import Tuple
from neurokin.constants.gait_cycle_detection import STEP_FILTER_FREQ
import numpy as np
from neurokin.utils.kinematics import event_detection
import pytest, pickle


def get_key(input_value):
    if isinstance(input_value, list):
        return tuple(input_value)
    return input_value


@pytest.fixture
def custom_signal():
    with open('../../test_data/steps_test_data/steps_y.pkl', 'rb') as f:
        custom_signal = pickle.load(f).to_numpy()
    yield custom_signal


class TestGetToeLiftLanding: # no mock for lowpass_array because output using steps_y.pkl too large to hardcode. lowpass_array output mostly dependent on scipy.signal

    def median_distance(self, a: np.ndarray) -> np.ndarray:
        key = get_key(a.tolist())

        return_values = {
            get_key(np.array([2025, 4085, 4243, 4411, 4760, 4908]).tolist()): -168.0
        }
        result = return_values.get(key, "default_output")
        return result

    def get_peak_boundaries_scipy(self, y: np.ndarray, px: float, left_crop: int) -> Tuple[int, int]:
        key = get_key(px)

        return_values = {
            get_key(2025): (1991, 2060),
            get_key(4085): (4043, 4118),
            get_key(4243): (4170, 4275),
            get_key(4411): (4364, 4435),
            get_key(4760): (4718, 4789),
            get_key(4908): (4846, 4932)
        }

        result = return_values.get(key, "default_output")
        return result

    def test_get_toe_lift_landing_with_custom_signal(self, custom_signal, mocker):
        mocker.patch('neurokin.utils.kinematics.event_detection.get_peak_boundaries_scipy',
                     side_effect=self.get_peak_boundaries_scipy)

        mocker.patch('neurokin.utils.kinematics.event_detection.median_distance',
                     side_effect=self.median_distance)

        left_bounds, right_bounds, max_x = event_detection.get_toe_lift_landing(custom_signal, recording_fs=200)
        assert (np.array_equal(left_bounds, np.array([1991, 4043, 4170, 4364, 4718, 4846])) and
                np.array_equal(right_bounds, np.array([2060, 4118, 4275, 4435, 4789, 4932])) and
                np.array_equal(max_x, np.array([2025, 4085, 4243, 4411, 4760, 4908])))


class TestGetPeakBoundariesScipy:

    def median_distance(self, a: np.ndarray) -> np.ndarray:
        key = get_key(a.tolist())

        return_values = {
            get_key(np.array([2025, 4085, 4243, 4411, 4760, 4908]).tolist()): -168.0
        }

        result = return_values.get(key, "default_output")
        return result

    def test_get_peak_boundaries_scipy_with_custom_signal(self, custom_signal, mocker):
        mocker.patch('neurokin.utils.kinematics.event_detection.median_distance',
                         side_effect=self.median_distance)

        y = event_detection.lowpass_array(custom_signal, STEP_FILTER_FREQ, 200)
        avg_distance = abs(int(event_detection.median_distance(np.array([2025, 4085, 4243, 4411, 4760, 4908])) / 2))

        return_values = {
            get_key(2025): (1991, 2060),
            get_key(4085): (4043, 4118),
            get_key(4243): (4170, 4275),
            get_key(4411): (4364, 4435),
            get_key(4760): (4718, 4789),
            get_key(4908): (4846, 4932)
        }

        for p in [2025, 4085, 4243, 4411, 4760, 4908]:
            left = p - avg_distance if p - avg_distance > 0 else 0
            right = p + avg_distance if p + avg_distance < len(y) else len(y)
            key = get_key(p)
            assert event_detection.get_peak_boundaries_scipy(y=y[left:right], px=p, left_crop=left) == return_values.get(key, "default_output")


class TestLowpassArray:

    def test_lowpass_array_with_empty_array(self):
        with pytest.raises(ValueError):
            event_detection.lowpass_array(np.array([]), critical_freq=STEP_FILTER_FREQ, fs=200)

    def test_lowpass_array_with_random_elements(self):
        np.testing.assert_allclose(
            event_detection.lowpass_array(np.array([13, 53, 34, 46, 5, 62, 55, 63, 82, 65]),
                                          critical_freq=10, fs=30), np.array([13.00106708, 45.02504679, 47.27823826,
                                                                              28.49348883, 24.27018935, 47.35905187,
                                                                              60.27249539, 65.71560878, 77.55592426,
                                                                              64.99635261]))

    def test_lowpass_array_with_custom_array(
            self, custom_signal):
        assert sum(event_detection.lowpass_array(custom_signal, critical_freq=STEP_FILTER_FREQ, fs=200))*100 == pytest.approx(
            24730.50916683458*100)
        np.testing.assert_allclose(event_detection.lowpass_array(custom_signal, critical_freq=STEP_FILTER_FREQ, fs=200)[1:10],
                                   np.array([5.09284297, 5.093026, 5.09322321, 5.09343471, 5.09366009, 5.09389833,
                                             5.09414767, 5.09440584, 5.0946703])
                                   )
        assert event_detection.lowpass_array(custom_signal, critical_freq=STEP_FILTER_FREQ, fs=200)[400] == pytest.approx(5.09304201977675)


class TestMedianDistance:

    def test_median_distance_with_empty_array(self):
        assert np.isnan(event_detection.median_distance(np.array([])))

    def test_median_distance_with_1_element(self):
        assert np.isnan(event_detection.median_distance(np.array([1])))

    def test_median_distance_with_2_elements(self):
        assert event_detection.median_distance(np.array([1,2])) == pytest.approx(-1.0)

    def test_median_distance_with_random_array(self):
        assert event_detection.median_distance(np.array([1, 3, 5, 6, 9, 16, 19, 22, 27, 30, 38, 39, 70])) == pytest.approx(-3.0)


