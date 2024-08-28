import numpy as np
import pytest
from neurokin.utils.kinematics import gait_params_basics
import pickle


def get_key(input_value):
    if isinstance(input_value, list):
        return tuple(input_value)
    return input_value


@pytest.fixture
def custom_signal():
    with open('../../test_data/steps_test_data/steps_y.pkl', 'rb') as f:
        custom_signal = pickle.load(f).to_numpy()
    yield custom_signal

# class TestComparePhase():
#
#    def get_phase_at_max_amplitude(self, input_signal):
#        key = get_key(input_signal)
#
#        def raise_exception():
#            raise TypeError(f"Invalid input: {input_signal}")
#
#        return_values = {
#            get_key(np.sin(np.radians(np.arange(360))).tolist()): 270.0,
#            get_key([]): raise_exception,
#            get_key(["a"]): raise_exception,
#            get_key(["a", "b"]): raise_exception,
#            get_key([1]): 360.0,
#            get_key([1, 2]): -90.0
#        }
#        result = return_values.get(key, "default_output")
#
#        return result
#
#    def test_compare_phase(self, mocker):
#        mocker.patch('neurokin.utils.kinematics.gait_params_basics.get_phase_at_max_amplitude',
#                     side_effect=self.get_phase_at_max_amplitude)
#        compare_phase = gait_params_basics.compare_phase
#
#        with pytest.raises(TypeError):
#            compare_phase([], [1])
#        with pytest.raises(TypeError):
#            compare_phase([1], [])
#        with pytest.raises(TypeError):
#            compare_phase(['a'], [1])
#        with pytest.raises(TypeError):
#            compare_phase([1], ['a'])
#        with pytest.raises(TypeError):
#            compare_phase(['a', 'b'], [1])
#        with pytest.raises(TypeError):
#            compare_phase([1], ['a', 'b'])
#        assert compare_phase([1, 2], [1]) == pytest.approx(-450.0)
#        assert compare_phase(np.sin(np.radians(np.arange(360))).tolist(), [1, 2]) == pytest.approx(360.0)


class TestComparePhase:

    def get_phase_at_max_amplitude(self, input_signal):
        key = get_key(input_signal)

        def raise_exception():
            raise TypeError(f"Invalid input: {input_signal}")

        with open('../../test_data/steps_test_data/steps_y.pkl', 'rb') as f:
            custom_signal = pickle.load(f).to_numpy()
        mid = int(len(custom_signal)/2)

        return_values = {
            get_key(np.sin(np.radians(np.arange(360))).tolist()): 270.0,
            get_key(np.cos(np.radians(np.arange(360))).tolist()): 360.0,
            get_key([]): raise_exception,
            get_key(["a"]): raise_exception,
            get_key(["a", "b"]): raise_exception,
            get_key([1]): 360.0,
            get_key([1, 2]): 360.0,
            get_key(custom_signal[:mid].tolist()): 360.0,
            get_key(custom_signal[mid:].tolist()): 360.0
        }
        result = return_values.get(key, "default_output")

        return result

    def test_compare_phase_with_empty_first_argument(self, mocker):
        mocker.patch('neurokin.utils.kinematics.gait_params_basics.get_phase_at_max_amplitude',
                     side_effect=self.get_phase_at_max_amplitude)
        self.compare_phase = gait_params_basics.compare_phase
        with pytest.raises(TypeError):
            self.compare_phase([], [1])

    def test_compare_phase_with_empty_second_argument(self, mocker):
        mocker.patch('neurokin.utils.kinematics.gait_params_basics.get_phase_at_max_amplitude',
                     side_effect=self.get_phase_at_max_amplitude)
        self.compare_phase = gait_params_basics.compare_phase
        with pytest.raises(TypeError):
            self.compare_phase([1], [])

    def test_compare_phase_valid_input(self, mocker):
        mocker.patch('neurokin.utils.kinematics.gait_params_basics.get_phase_at_max_amplitude',
                     side_effect=self.get_phase_at_max_amplitude)
        self.compare_phase = gait_params_basics.compare_phase
        assert self.compare_phase([1, 2], [1]) == pytest.approx(0.0)

    def test_compare_phase_with_sin_and_cos_input(self, mocker):
        mocker.patch('neurokin.utils.kinematics.gait_params_basics.get_phase_at_max_amplitude',
                     side_effect=self.get_phase_at_max_amplitude)
        self.compare_phase = gait_params_basics.compare_phase
        assert self.compare_phase(np.sin(np.radians(np.arange(360))).tolist(), np.cos(np.radians(np.arange(360))).tolist()) == pytest.approx(-90.0)

    def test_compare_phase_with_custom_input(self, custom_signal, mocker):
        mocker.patch('neurokin.utils.kinematics.gait_params_basics.get_phase_at_max_amplitude',
                     side_effect=self.get_phase_at_max_amplitude)
        self.compare_phase = gait_params_basics.compare_phase
        mid = int(len(custom_signal)/2)
        assert self.compare_phase(custom_signal[:mid].tolist(), custom_signal[mid:].tolist()) == pytest.approx(0.0)


class TestGetAngle:
    def test_get_angle_with_empty_array(self):
        with pytest.raises(ValueError):
            gait_params_basics.get_angle(np.array([]))

    def test_get_angle_with_two_2d_points(self):
        with pytest.raises(ValueError):
            gait_params_basics.get_angle(np.array([[1, 2], [1, 3]]))

    def test_get_angle_with_3_1d_points_in_1d_array(self):
        with pytest.raises(ValueError):
            gait_params_basics.get_angle(np.array([1, 2, 3]))

    def test_get_angle_with_3_1d_points_in_2d_array(self):
        with pytest.raises(ValueError):
            gait_params_basics.get_angle(np.array([[1], [2], [3]]))

    def test_get_angle_with_3_different_2d_points(self):
        print(np.array([[1, 1], [0, 0], [0, 1]]).shape)
        assert gait_params_basics.get_angle(np.array([[1, 1], [0, 0], [0, 1]])) == pytest.approx(45.0)

    def test_get_angle_with_3_same_2d_points(self):
        assert np.isnan(gait_params_basics.get_angle(np.array([[0, 0], [0, 0], [0, 0]])))

    def test_get_angle_with_2_2d_points_1_repeated(self):
        assert gait_params_basics.get_angle(np.array([[0, 0], [0, 1], [0, 0]])) == pytest.approx(0.0)

    def test_get_angle_with_3_collinear_2d_points(self):
        assert gait_params_basics.get_angle(np.array([[0, 0], [0, 1], [0, 2]])) == pytest.approx(180.0)

    def test_get_angle_with_3_different_3d_points(self):
        assert gait_params_basics.get_angle(np.array([[0, 1, 1], [0, 0, 0], [0, 0, 1]])) == pytest.approx(45.0)

    def test_get_angle_with_3_different_4d_points(self):
        with pytest.raises(ValueError):
            gait_params_basics.get_angle(np.array([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0]]))

    def test_get_angle_with_3d_input(self):
        with pytest.raises(ValueError):
            gait_params_basics.get_angle(np.array([[[0, 1, 1], [0, 0, 0], [0, 0, 1]], [[0, 1, 1], [0, 0, 0], [0, 0, 1]],
                                                   [[0, 1, 1], [0, 0, 1], [0, 0, 1]]]))

        with pytest.raises(ValueError):
            gait_params_basics.get_angle(np.array([[[0, 1, 1], [0, 0, 0], [0, 0, 1]], [[0, 1, 1], [1, 0, 0], [0, 0, 1]],
                                                   [[0, 1, 1], [0, 0, 1], [0, 0, 1]]]))


class TestGetPhaseAtMaxAmplitude:
    def test_get_phase_at_max_amplitude_with_empty_array(self):
        with pytest.raises(ValueError):
            gait_params_basics.get_phase_at_max_amplitude(np.array([]))

    def test_get_phase_at_max_amplitude_with_vector_input(self):
        assert gait_params_basics.get_phase_at_max_amplitude(np.array(['j'])) == pytest.approx(90.0)

    def test_get_phase_at_max_amplitude_with_1_point(self):
        assert gait_params_basics.get_phase_at_max_amplitude(np.array([1])) == pytest.approx(360.0)

    def test_get_phase_at_max_amplitude_with_2_points(self):
        assert gait_params_basics.get_phase_at_max_amplitude(np.array([1, 2])) == pytest.approx(360.0)

    def test_get_phase_at_max_amplitude_with_sin_input(self):
        assert gait_params_basics.get_phase_at_max_amplitude(np.sin(np.radians(np.arange(360)))) == pytest.approx(270.0)

    def test_get_phase_at_max_amplitude_with_custom_input(self, custom_signal):
        assert gait_params_basics.get_phase_at_max_amplitude(custom_signal) == pytest.approx(360.0)


class TestGetPhase:
    def test_get_phase_with_empty_array(self):
        with pytest.raises(ValueError):
            gait_params_basics.get_phase(np.array([]))

    def test_get_phase_with_vector_input(self):
        np.testing.assert_allclose(gait_params_basics.get_phase(np.array(['j'])), np.array([1.57079633]))

    def test_get_phase_with_1_point(self):
        np.testing.assert_allclose(gait_params_basics.get_phase(np.array([1])), np.array([0]))

    def test_get_phase_with_2_points(self):
        np.testing.assert_allclose(gait_params_basics.get_phase(np.array([1, 2])), np.array([0., 3.14159265]))

    def test_get_phase_with_random_signal(self):
        np.testing.assert_allclose(gait_params_basics.get_phase(np.array([39, 5, 16, 22, 3, 38, 30, 19, 1, 6, 70, 9, 27])), np.array(
            [0., 1.14260506, 1.56542311, -1.88276653, 0.2841719, 0.706717, 2.81263504, -2.81263504, -0.706717, -0.2841719,
             1.88276653, -1.56542311, -1.14260506]))

    def test_get_phase_with_custom_signal(self, custom_signal):
        assert sum(gait_params_basics.get_phase(custom_signal))*1e15 == pytest.approx(-1208.3667400020204)
        print(gait_params_basics.get_phase(custom_signal)[1:10])
