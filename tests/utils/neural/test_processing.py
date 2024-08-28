import numpy as np
import pytest

from neurokin.neural_data import NeuralData
from neurokin.utils.neural import processing


def get_key(input_value):
    if isinstance(input_value, list):
        return tuple(input_value)
    return input_value


class TestSimplyMeanDataBinarize:

    def test_simply_mean_data_binarize_with_empty_array(self):
        assert np.array_equal(processing.simply_mean_data_binarize(np.array([])), np.array([]))

    def test_simply_mean_data_binarize_with_1_element_array(self):
        assert np.array_equal(processing.simply_mean_data_binarize(np.array([1.0])), np.array([1]))

    def test_simply_mean_data_binarize_with_random_array(self):
        assert np.array_equal(processing.simply_mean_data_binarize(np.array([1.0, 2.0, 3.0, 4.0])),
                              np.array([0, 0, 1, 1]))

    @pytest.mark.skip(reason="fails due to nan comparisons")
    def test_simply_mean_data_binarize_with_nan_array(self):
        # assert np.array_equal(processing.simply_mean_data_binarize(np.array([1, np.nan])), np.array([1, 1]))
        with pytest.raises(ValueError):
            processing.simply_mean_data_binarize(np.array([1, np.nan]))  # currently returns np.array([1, 1])

    @pytest.mark.skip(reason="fails due to lack of support for 2D arrays")
    def test_simply_mean_data_binarize_with_2d_array(self):
        # with pytest.raises(ValueError):
        #    processing.simply_mean_data_binarize(np.array([[1, 1, 2], [1, 2, 3]]))  # should be np.array([[0, 0, 1],[0, 1, 1]])
        assert np.array_equal(processing.simply_mean_data_binarize(np.array([[1, 1, 2], [1, 2, 3]])),
                              np.array([[0, 0, 1], [0, 1, 1]]))


class TestGetStimTimestamps:

    def test_get_stim_timestamps_with_empty_array(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([])), np.array([]))

    def test_get_stim_timestamps_with_nan_array(self):  # should this return Warning or ValueError?
        assert np.array_equal(processing.get_stim_timestamps(np.array([np.nan])), np.array([]))
        assert np.array_equal(processing.get_stim_timestamps(np.array([np.nan, 1])), np.array([1]))

    @pytest.mark.skip(reason="fails due to lack of check for dim")
    def test_get_stim_timestamps_with_2d_array(self):
        with pytest.raises(ValueError):  # currently returns np.array([0])
            processing.get_stim_timestamps(np.array([[1, 2, 3], [0, 2, 3]]))

    def test_get_stim_timestamps_with_random_array_without_expected_pulses(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4])),
                              np.array([1, 8]))

    def test_get_stim_timestamps_with_random_array_with_greater_expected_pulses(self):
        assert np.array_equal(
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=3),
            np.array([1, 8]))  # should there be a Warning message here

    def test_get_stim_timestamps_with_random_array_with_exact_expected_pulses(self):
        assert np.array_equal(
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=2),
            np.array([1, 8]))

    def test_get_stim_timestamps_with_random_array_with_lower_expected_pulses(self):
        assert np.array_equal(
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=1),
            np.array([1]))

    def test_get_stim_timestamps_with_random_array_with_0_expected_pulses(self):
        assert np.array_equal(
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=0),
            np.array([1, 8]))  # should this return empty array np.array([])?

    @pytest.mark.skip(reason="fails due to no handling of negative expected_pulses")
    def test_get_stim_timestamps_with_random_array_with_negative_expected_pulses(self):
        # assert np.array_equal(
        #    processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=-1),
        #    np.array([1]))
        # assert np.array_equal(
        #    processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=-2),
        #    np.array([]))  # might want to return ValueError?
        # assert np.array_equal(
        #    processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=-3),
        #    np.array([]))  # might want to return ValueError?
        with pytest.raises(ValueError):
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=-1)
        with pytest.raises(ValueError):
            processing.get_stim_timestamps(np.array([0, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4]), expected_pulses=-2)

    def test_get_stim_timestamps_with_random_array_mid_pulse(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([1, 1, 2, 1, 2, 3, 0, 0, 1, 2, 3, 4])),
                              np.array([0, 8]))

    def test_get_stim_timestamps_with_0_array(self):
        assert np.array_equal(processing.get_stim_timestamps(np.array([0, 0, 0, 0, 0, 0])), np.array([]))


class TestGetTTimestampsStimBlocks:

    @pytest.fixture
    def neural_data(self, mocker):
        mocker.patch('neurokin.utils.neural.processing.get_stim_timestamps', side_effect=self.get_stim_timestamps)
        mocker.patch('neurokin.utils.neural.importing.time_to_sample', side_effect=self.time_to_sample)
        data = NeuralData("")
        data.fs = 1
        data.sync_data = np.array([0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 0])
        yield data

    def get_stim_timestamps(self, sync_ch: np.ndarray, expected_pulses: int = None) -> np.ndarray:
        key = get_key(str(sync_ch))
        return_values = {get_key(str(np.array([]))): {str(1): np.array([])},
                         get_key(str(np.array([0, 0, 0, 0, 0, 0]))): {str(1): np.array([])},
                         get_key(str(np.array([0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 0]))): {
                             str(6): np.array([1, 3, 6, 8, 11, 13]),
                             str(4): np.array([1, 3, 6, 8]),
                             str(8): np.array([1, 3, 6, 8, 11, 13]),
                             str(0): np.array([1, 3, 6, 8, 11, 13])},
                         get_key(str(np.array([0, 1, 1, 0, 1, 1, 0, 0, 2, 2, 0, 2, 2, 0, 0, 3, 3, 0, 3]))): {
                             str(6): np.array([1, 4, 8, 11, 15, 18])}
                         }
        return return_values.get(key, "default_output")[str(expected_pulses)]

    def time_to_sample(self, timestamp: float, fs: float, is_t1: bool = False, is_t2: bool = False) -> int:
        return_values = {str(3): 2,
                         str(1): 0,
                         str(5): 4}
        return return_values[str(timestamp)]

    def test_get_timestamps_stim_blocks_with_empty_array(self, neural_data):
        neural_data.sync_data = np.array([])
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=1, pulses=1, time_stim=1) == []

    def test_get_timestamps_stim_blocks_with_0_array(self, neural_data):
        neural_data.sync_data = np.array([0, 0, 0, 0, 0, 0])
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=1, pulses=1, time_stim=1) == []

    def test_get_timestamps_stim_blocks_with_random_array_and_exact_n_amp_tested(self, neural_data):
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=3, pulses=2, time_stim=3) == [(1, 3),
                                                                                                             (6, 8),
                                                                                                             (11, 13)]

    def test_get_timestamps_stim_blocks_with_random_array_and_lower_n_amp_tested(self, neural_data):
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=2, pulses=2, time_stim=3) == [(1, 3),
                                                                                                             (6, 8)]

    def test_get_timestamps_stim_blocks_with_random_array_and_greater_n_amp_tested(self, neural_data):
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=4, pulses=2, time_stim=3) == [(1, 3),
                                                                                                             (6, 8),
                                                                                                             (11, 13)]

    def test_get_timestamps_stim_blocks_with_random_array_and_0_n_amp_tested(self, neural_data):
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=0, pulses=2, time_stim=3) == [(1, 3),
                                                                                                             (6, 8),
                                                                                                             (11, 13)]

    def test_get_timestamps_stim_blocks_with_random_array_with_1_pulse_per_block(self, neural_data):
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=6, pulses=1, time_stim=1) == [(1, 1),
                                                                                                             (3, 3),
                                                                                                             (6, 6),
                                                                                                             (8, 8),
                                                                                                             (11, 11),
                                                                                                             (13, 13)]

    def test_get_timestamps_stim_blocks_with_random_array_ending_mid_pulse(self,
                                                                           neural_data):  # should the end of the last block be adjusted to the length of the array?
        neural_data.sync_data = np.array([0, 1, 1, 0, 1, 1, 0, 0, 2, 2, 0, 2, 2, 0, 0, 3, 3, 0, 3])
        assert processing.get_timestamps_stim_blocks(neural_data, n_amp_tested=3, pulses=2, time_stim=5) == [(1, 5),
                                                                                                             (8, 12),
                                                                                                             (15, 19)]


class TestGetMedianDistance:

    def test_get_median_distance_with_empty_array(self):
        assert np.isnan(processing.get_median_distance(np.array([])))

    def test_get_median_distance_with_1_element(self):
        assert np.isnan(processing.get_median_distance(np.array([1])))

    def test_get_median_distance_with_2_elements(self):
        assert processing.get_median_distance(np.array([1, 2])) == pytest.approx(1.0)

    def test_get_median_distance_with_random_array(self):
        assert processing.get_median_distance(
            np.array([1, 3, 5, 6, 9, 16, 19, 22, 27, 30, 38, 39, 70])) == pytest.approx(3.0)

    @pytest.mark.skip(reason="fails due to lack of check for string input")
    def test_get_median_distance_with_1_string_element(
            self):  # currently returns nan, might be better to return TypeError
        with pytest.raises(TypeError):
            processing.get_median_distance(np.array(['a']))
        # assert np.isnan(processing.get_median_distance(np.array(['a'])))


class TestRunningMean:

    def test_running_mean_with_empty_array(self):
        assert np.array_equal(processing.running_mean(x=np.array([]), n=1), np.array([]))

    def test_running_mean_with_random_array(self):
        np.testing.assert_allclose(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=1),
                                   np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_running_mean_with_random_list(self):
        np.testing.assert_allclose(processing.running_mean(x=[1, 2, 3, 4, 5], n=1), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_running_mean_with_random_tuple(self):
        np.testing.assert_allclose(processing.running_mean(x=(1, 2, 3, 4, 5), n=1), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_running_mean_with_n_smaller_than_array_length(self):
        np.testing.assert_allclose(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=2),
                                   np.array([1.5, 2.5, 3.5, 4.5]))
        np.testing.assert_allclose(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=3), np.array([2.0, 3.0, 4.0]))
        np.testing.assert_allclose(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=4), np.array([2.5, 3.5]))

    def test_running_mean_with_n_equal_to_array_length(self):
        np.testing.assert_allclose(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=5), np.array([3.0]))

    @pytest.mark.skip(reason="fails due to lack of check for value of 'n'")
    def test_running_mean_with_invalid_n(self):
        # assert np.array_equal(processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=6), np.array([]))
        with pytest.raises(ValueError):  # currently returns np.array([])
            processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=6)

        with pytest.raises(ValueError):
            processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=0)

        with pytest.raises(ValueError):  # currently returns np.array([-15.])
            processing.running_mean(x=np.array([1, 2, 3, 4, 5]), n=-1)

    @pytest.mark.skip(reason="fails due to lack of check for dim")
    def test_running_mean_with_2d_array(self):
        with pytest.raises(ValueError):  # currently returns np.array([1., 2., 3., 4., 5., 1., 2., 3., 4., 5.])
            processing.running_mean(x=np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]), n=1)


class TestTrimEqualLen:

    def test_trim_equal_len_with_empty_list(self):
        raw = []
        assert all([np.array_equal(a, b) for a, b in zip(processing.trim_equal_len(raw), raw)])

    def test_trim_equal_len_with_list_of_at_least_1_empty_arrays(self):
        raw = [np.array([]), np.array([])]
        assert all([np.array_equal(a, b) for a, b in zip(processing.trim_equal_len(raw), raw)])

        raw = [np.array([]), np.array([1, 2, 3, 4, 5])]
        assert all([np.array_equal(a, b) for a, b in zip(processing.trim_equal_len(raw), [np.array([]), np.array([])])])

    def test_trim_equal_len_with_list_of_equal_length_arrays(self):
        raw = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])]
        assert all([np.array_equal(a, b) for a, b in zip(processing.trim_equal_len(raw), raw)])

    def test_trim_equal_len_with_list_of_unequal_length_arrays_in_any_order(self):
        raw = [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4, 5])]
        assert all(
            [np.array_equal(a, b) for a, b in
             zip(processing.trim_equal_len(raw), [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])])])

        raw = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4])]
        assert all(
            [np.array_equal(a, b) for a, b in
             zip(processing.trim_equal_len(raw), [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])])])

    @pytest.mark.skip(reason="fails due to lack of check for dim of list elements")
    def test_trim_equal_len_with_list_with_at_least_1_2d_array(self):
        raw = [np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]), np.array([1, 2, 3, 4, 5])]
        assert all([np.array_equal(a, b) for a, b in
                    zip(processing.trim_equal_len(raw), [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])])])
        # should shape of individual arrays be checked before trimming? this could add computational time if array too long.
        # could be unnecessary if input usually is list of arrays of same dimension.
        # OR: using 1 for loop instead of list comprehension to trim arrays as well as check dimensions simultaneously


class TestParseRaw:

    @pytest.fixture
    def input_data(self, mocker):
        mocker.patch('neurokin.utils.neural.processing.trim_equal_len', side_effect=self.trim_equal_len)
        raw = np.array(
            [0.4, 1.2, 2.4, 3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4, 2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])
        stimulation_idxs = np.array([3, 10, 14, 16, 19])
        samples_before_stim = 0
        skip_one = False
        min_len_chunk = 1
        yield raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk

    def trim_equal_len(self, raw):
        key = get_key(str(raw))
        return_values = {
            get_key(str([np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4]),
                         np.array([2.2, 3.6, 4.4, 3.5]),
                         np.array([2.6, 1.2]),
                         np.array([0.7, 0.3, 0.2]),
                         np.array([1.7])])): [np.array([3.1]), np.array([2.2]), np.array([2.6]), np.array([0.7]),
                                              np.array([1.7])],
            get_key(str([np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4]),
                         np.array([2.2, 3.6, 4.4, 3.5]),
                         np.array([2.6, 1.2]),
                         np.array([0.7, 0.3, 0.2])])): [np.array([3.1, 4.6]), np.array([2.2, 3.6]),
                                                        np.array([2.6, 1.2]), np.array([0.7, 0.3])],
            get_key(str([np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4, 2.2, 3.6, 4.4, 3.5]),
                         np.array([2.6, 1.2, 0.7, 0.3, 0.2]),
                         np.array([1.7])])): [np.array([3.1]), np.array([2.6]), np.array([1.7])],
            get_key(str([np.array([0.4, 1.2, 2.4, 3.1, 4.6, 3.3]),
                         np.array([2.7, 1.8, 0.2, 1.4]),
                         np.array([2.2, 3.6]),
                         np.array([4.4, 3.5, 2.6]),
                         np.array([1.2, 0.7, 0.3, 0.2, 1.7])])): [np.array([0.4, 1.2]), np.array([2.7, 1.8]),
                                                                  np.array([2.2, 3.6]), np.array([4.4, 3.5]),
                                                                  np.array([1.2, 0.7])],
            get_key(str([np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4])])): [
                np.array([3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4])],
            get_key(str([])): []}

        return return_values.get(key, "default_output")

    def test_parse_raw_with_min_len_chunk_lower_than_longest_chunk(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data

        np.testing.assert_allclose(
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
            np.array([[3.1], [2.2], [2.6], [0.7], [1.7]]))

        min_len_chunk = 2
        np.testing.assert_allclose(
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
            np.array([[3.1, 4.6], [2.2, 3.6], [2.6, 1.2], [0.7, 0.3]]))

    def test_parse_raw_with_min_len_chunk_equals_length_of_longest_chunk(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        min_len_chunk = 7

        np.testing.assert_allclose(
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
            [[3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4]])

    def test_parse_raw_with_min_len_chunk_greater_than_length_of_longest_chunk(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        min_len_chunk = 8

        with pytest.raises(ValueError):  # should this return an empty list instead?
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk)

    def test_parse_raw_with_skip_one_equals_true(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        skip_one = True

        assert np.array_equal(processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
                              np.array([[3.1], [2.6], [1.7]]))

    @pytest.mark.skip(reason="fails due to lack of support for stimulation_idxs of type 'list'")
    def test_parse_raw_with_stimulation_idxs_as_list(self,
                                                     input_data):  # could add support for this data type since raw can work as a list
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        stimulation_idxs = [3, 10, 14, 16, 19]

        # with pytest.raises(TypeError):
        #    processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk)
        np.testing.assert_allclose(
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
            np.array([[3.1], [2.2], [2.6], [0.7], [1.7]]))

    @pytest.mark.skip(reason="fails due to lack of constraint on length of stimulation_idxs")
    def test_parse_raw_with_empty_stimulation_idxs(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        stimulation_idxs = np.array([])

        with pytest.raises(ValueError):
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk)

    def test_parse_raw_with_empty_array(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        raw = np.array([])

        with pytest.raises(ValueError):  # message not very informative
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk)

    def test_parse_raw_with_samples_before_stim_greater_than_first_stim_id(self, input_data):
        raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk = input_data
        samples_before_stim = 4

        np.testing.assert_allclose(
            processing.parse_raw(raw, stimulation_idxs, samples_before_stim, skip_one, min_len_chunk),
            np.array([[0.4, 1.2], [2.7, 1.8], [2.2, 3.6], [4.4, 3.5], [1.2, 0.7]]))


class TestGetAverageAmplitudes:

    @pytest.fixture
    def input_data(self, mocker):
        mocker.patch('neurokin.utils.neural.processing.average_block', side_effect=self.average_block)
        parsed_raw = np.array([[0.4, 1.2], [2.7, 1.8], [2.2, 3.6], [4.4, 3.5], [1.2, 0.7]])
        tested_amplitudes = [1]
        yield parsed_raw, tested_amplitudes

    def average_block(self, array: np.typing.ArrayLike, start: int, stop: int) -> np.ndarray:
        block = array[start:stop]
        key = get_key(str(block))

        return_values = {get_key(str(np.array([[0.4, 1.2],
                                               [2.7, 1.8],
                                               [2.2, 3.6],
                                               [4.4, 3.5],
                                               [1.2, 0.7]]))): [2.18, 2.16],
                         get_key(str(np.array([[0.4, 1.2],
                                               [2.7, 1.8]]))): [1.55, 1.5],
                         get_key(str(np.array([[2.2, 3.6],
                                               [4.4, 3.5]]))): [3.3, 3.55],
                         get_key(str(np.array([[0.4, 1.2]]))): [0.4, 1.2],
                         get_key(str(np.array([[2.7, 1.8]]))): [2.7, 1.8]
                         }

        return return_values.get(key, "default_output")

    def test_get_average_amplitudes_with_1_tested_amplitude(self, input_data):
        parsed_raw, tested_amplitudes = input_data
        np.testing.assert_allclose(processing.get_average_amplitudes(parsed_raw, tested_amplitudes), [[2.18, 2.16]])

    def test_get_average_amplitudes_with_multiple_tested_amplitudes_and_default_pulses_number(self, input_data):
        parsed_raw, tested_amplitudes = input_data
        tested_amplitudes = [1, 2]

        np.testing.assert_allclose(processing.get_average_amplitudes(parsed_raw, tested_amplitudes),
                                   [np.array([1.55, 1.5]), np.array([3.3, 3.55])])

    def test_get_average_amplitudes_with_multiple_tested_amplitudes_and_custom_pulses_number(self, input_data):
        parsed_raw, tested_amplitudes = input_data
        tested_amplitudes = [1, 2]
        pulses_number = 1

        np.testing.assert_allclose(processing.get_average_amplitudes(parsed_raw, tested_amplitudes, pulses_number),
                                   [np.array([0.4, 1.2]), np.array([2.7, 1.8])])


class TestAverageBlock:

    @pytest.fixture
    def input_data(self):
        array = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        start = 0
        stop = 8
        yield array, start, stop

    def test_average_block_with_random_1d_input(self, input_data):
        array, start, stop = input_data

        assert processing.average_block(array, start, stop) == pytest.approx(3.5)

    def test_average_block_with_random_2d_input(self, input_data):
        array, start, stop = input_data
        array = np.array([[0.4, 1.2], [2.7, 1.8], [2.2, 3.6], [4.4, 3.5], [1.2, 0.7]])

        np.testing.assert_allclose(processing.average_block(array, start, stop), np.array([2.18, 2.16]))

    def test_average_block_with_stop_greater_than_array_length(self, input_data):
        array, start, stop = input_data
        stop = 9

        assert processing.average_block(array, start, stop) == pytest.approx(3.5)

    def test_average_block_with_start_greater_than_array_length(self, input_data):
        array, start, stop = input_data
        start = 9

        assert np.isnan(processing.average_block(array, start, stop))  # could be better to return IndexError
        # with pytest.raises(IndexError):
        #    processing.average_block(array, start, stop)

    def test_average_block_with_empty_array(self, input_data):
        array, start, stop = input_data
        array = np.array([])

        assert np.isnan(processing.average_block(array, start, stop))  # could be better to return ValueError
        # with pytest.raises(ValueError):
        #    processing.average_block(array, start, stop)


class TestFindClosestIndex:

    def test_find_closest_index_with_empty_input(self):
        data = np.array([])
        with pytest.raises(ValueError):
            processing.find_closest_index(data, datapoint=3)

    def test_find_closest_index_with_nan_input(self):
        data = np.array([np.nan, 1])
        with pytest.raises(ValueError,
                           match=r"The input array contains nan which will always return as the nearest to datapoint"):
            processing.find_closest_index(data, datapoint=3)

    def test_find_closest_index_with_input_having_single_match(self):
        data = np.array([0.4, 1.2, 2.4, 3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4,
                         2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])
        assert processing.find_closest_index(data, datapoint=3) == 3

    def test_find_closest_index_with_input_having_two_matches(self):
        data = np.array([0.4, 1.2, 2.4, 3.1, 4.6, 2.9, 2.7, 1.8, 0.2, 1.4,
                         2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])
        assert processing.find_closest_index(data, datapoint=3) == 3

    @pytest.mark.skip(reason="fails due to lack of support for 2D arrays")
    def test_find_closest_index_(self):
        data = np.array([[0.4, 1.2, 2.4, 3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4],
                         [2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7]])
        assert processing.find_closest_index(data, datapoint=3) == (0, 3)  # currently returns 3


class TestFindClosestSmallerIndex:

    @pytest.mark.skip(reason="fails due to lack of check for empty input")
    def test_find_closest_smaller_index_with_empty_input(self):
        data = np.array([])

        # assert processing.find_closest_smaller_index(data, datapoint=3) == 0
        with pytest.raises(ValueError):
            processing.find_closest_index(data, datapoint=3)

    def test_find_closest_smaller_index_with_nan_input(self):  # should this case be handled differently?
        data = np.array([np.nan, 1])

        assert processing.find_closest_smaller_index(data, datapoint=3) == 1

    def test_find_closest_smaller_index_with_input_having_single_match(self):
        data = np.array([0.4, 1.2, 2.4, 3.1, 4.6, 3.3, 2.7, 1.8, 0.2, 1.4,
                         2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])

        assert processing.find_closest_smaller_index(data, datapoint=3) == 6

    def test_find_closest_smaller_index_with_input_having_two_matches(self):
        data = np.array([0.4, 1.2, 2.4, 3.1, 4.6, 2.9, 2.7, 1.8, 0.2, 1.4,
                         2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7])
        print(processing.find_closest_smaller_index(data, datapoint=3))
        assert processing.find_closest_smaller_index(data, datapoint=3) == 5

    @pytest.mark.skip(reason="fails due to lack of support for 2D arrays")
    def test_find_closest_index_(self):
        data = np.array([[0.4, 1.2, 2.4, 3.1, 4.6, 2.9, 2.7, 1.8, 0.2, 1.4],
                         [2.2, 3.6, 4.4, 3.5, 2.6, 1.2, 0.7, 0.3, 0.2, 1.7]])

        assert processing.find_closest_index(data, datapoint=3) == (0, 5)  # currently returns 5


class TestGetSpectrogramData:

    @pytest.fixture
    def input_data(self):
        path = '../../test_data/TDT_test_data'

        baseline = NeuralData(path=path)
        baseline.load_tdt_data(stream_name='NPr1', sync_present=True, t2=10)

        fs = baseline.fs
        raw = baseline.raw
        yield fs, raw

    def test_get_spectrogram_data(self,
                                  input_data):  # result too big to hardcode so checked with the first segment only
        fs, raw = input_data
        sxx_actual, _, _ = processing.get_spectrogram_data(fs, raw)
        sxx_expected = np.array([1.8551391e-16, 7.8996974e-12, 3.1156334e-12, 2.1707767e-13, 3.4549365e-13,
                                 1.4605963e-12, 8.0570811e-14, 1.2705112e-13, 5.3634533e-14, 3.7372462e-13,
                                 1.4671948e-13, 4.1749980e-14, 2.0093047e-13, 6.4747084e-15, 2.1308730e-14,
                                 9.2793815e-15, 2.6254560e-14, 9.4176630e-15, 4.8945430e-15, 5.0443518e-15,
                                 1.5850226e-14, 6.6846681e-14, 9.4427551e-14, 7.2213989e-15, 4.9174959e-15,
                                 3.6758786e-15, 2.5547733e-15, 1.8223889e-16, 1.3239343e-16, 3.5653942e-15,
                                 4.4903304e-15, 1.6791256e-15, 1.9090689e-15, 6.5368688e-16, 2.1796196e-17,
                                 2.8413204e-15, 2.8762929e-15, 2.6574212e-15, 1.2653715e-15, 6.5820267e-16,
                                 1.5681276e-15, 3.8306700e-16])

        np.testing.assert_allclose(sxx_actual[0, :42, 0] * 1e15, sxx_expected * 1e15)


class TestCalculatePowerSpectralDensity:

    @pytest.fixture
    def input_data(self):
        path = '../../test_data/TDT_test_data'

        baseline = NeuralData(path=path)
        baseline.load_tdt_data(stream_name='NPr1', sync_present=True, t2=10)

        fs = baseline.fs
        raw = baseline.raw
        yield fs, raw

    def test_calculate_power_spectral_density(self, input_data):
        fs, data = input_data
        _, pxx_actual = processing.calculate_power_spectral_density(data, fs)
        pxx_expected = np.array([3.94866323e-13, 2.21475616e-12, 1.18454378e-12, 6.85829865e-13,
                                 4.84005089e-13, 3.29589817e-13, 2.30785172e-13, 1.64235569e-13,
                                 1.12752594e-13, 8.17098464e-14, 6.74502229e-14, 6.73876441e-14,
                                 4.18533751e-14, 2.63994166e-14, 1.96033358e-14, 1.43593735e-14,
                                 1.24931380e-14, 1.25488516e-14, 1.08167903e-14, 7.93379085e-15,
                                 6.65554148e-15, 2.30616799e-14, 8.14694381e-14, 3.47835571e-14,
                                 6.78727501e-15, 5.48199766e-15, 4.33956917e-15, 4.22389539e-15,
                                 5.55817175e-15, 5.96957185e-15, 5.61319374e-15, 4.65065049e-15,
                                 4.12314336e-15, 4.62462456e-15, 5.14880835e-15, 5.29464583e-15,
                                 4.81274676e-15, 4.00256060e-15, 3.44040685e-15, 3.72561936e-15,
                                 4.55908157e-15, 4.99562583e-15])
        np.testing.assert_allclose(pxx_actual[0, :42] * 1e15, pxx_expected * 1e15)
