import numpy as np
import pytest

from neurokin.utils.neural import importing


def get_key(input_value):
    if isinstance(input_value, list):
        return tuple(input_value)
    return input_value


class TestTimeToSample:

    def test_time_to_sample_normal(self):
        assert importing.time_to_sample(timestamp=1, fs=40, is_t1=False, is_t2=False) == 40
        assert importing.time_to_sample(timestamp=1.5, fs=40, is_t1=False, is_t2=False) == 60
        assert importing.time_to_sample(timestamp=1.75, fs=40.2, is_t1=False, is_t2=False) == 70

    def test_time_to_sample_with_is_t1_equal_true(self):
        assert importing.time_to_sample(timestamp=1.5, fs=40, is_t1=True, is_t2=False) == 60
        assert importing.time_to_sample(timestamp=1.75, fs=40.2, is_t1=True, is_t2=False) == 71

    def test_time_to_sample_with_is_t2_equal_true(self):
        assert importing.time_to_sample(timestamp=1.5, fs=40, is_t1=False, is_t2=True) == 59
        assert importing.time_to_sample(timestamp=1.75, fs=40.2, is_t1=False, is_t2=True) == 70
        assert importing.time_to_sample(timestamp=28.72, fs=24414.0625, is_t2=True) == 701171

    def test_time_to_sample_with_timestamp_equal_zero(self):
        assert importing.time_to_sample(timestamp=0, fs=40.2, is_t1=False, is_t2=False) == 0
        assert importing.time_to_sample(timestamp=0, fs=40.2, is_t1=True, is_t2=False) == 0
        assert importing.time_to_sample(timestamp=0, fs=24414.0625, is_t1=True, is_t2=False) == 0

    @pytest.mark.skip(reason="fails due to lack of check for value of 'timestamp' parameter when 'is_t2' is 'True'")
    def test_time_to_sample_with_timestamp_equal_zero_and_is_t2_equal_true(self):
        with pytest.raises(ValueError):
            importing.time_to_sample(timestamp=0, fs=40.2, is_t1=False, is_t2=True)
        # assert importing.time_to_sample(timestamp=0, fs=40.2, is_t1=False, is_t2=True) == -1  # maybe should process timestamp == 0 case differently so that output stays positive? should correct output be 0 whenever timestamp == 0 no matter t1 or t2?

    @pytest.mark.skip(reason="fails due to lack of constraints on values of 'timestamp' parameter")
    def test_time_to_sample_with_negative_timestamp(self):
        # assert importing.time_to_sample(timestamp=-1, fs=40.2, is_t1=False, is_t2=False) == -40  # this should probably throw ValueError, unless time is allowed to be in the negative direction
        with pytest.raises(ValueError):
            importing.time_to_sample(timestamp=-1, fs=40.2, is_t1=False, is_t2=False)

    @pytest.mark.skip(reason="fails due to lack of constraints on values of 'fs' parameter")
    def test_time_to_sample_with_fs_equal_0():
        # assert importing.time_to_sample(timestamp=1.75, fs=0, is_t1=False, is_t2=False) == 0  # maybe should throw ValueError?
        with pytest.raises(ValueError):
            importing.time_to_sample(timestamp=1.75, fs=0, is_t1=False, is_t2=False)

    @pytest.mark.skip(reason="fails due to lack of constraints on values of 'fs' parameter")
    def test_time_to_sample_with_negative_fs(self):
        # assert importing.time_to_sample(timestamp=1, fs=-40.2, is_t1=False, is_t2=False) == -40  # this should probably throw ValueError
        with pytest.raises(ValueError):
            importing.time_to_sample(timestamp=1, fs=-40.2, is_t1=False, is_t2=False)


class TestImportTDTChannelData:

    def time_to_sample(self, timestamp: float, fs: float, is_t1: bool = False, is_t2: bool = False) -> int:
        key = get_key(timestamp)

        return_values = {get_key(0): 0,
                         get_key(28.72): 701171}

        return return_values.get(key, 'default_output')

    @pytest.fixture
    def folderpath(self):
        yield '../../test_data/TDT_test_data'

    @pytest.fixture
    def all_channels_output(self, folderpath, mocker):
        mocker.patch('neurokin.utils.neural.importing.time_to_sample', side_effect=self.time_to_sample)
        fs, raw, stim_data, fs_sync = importing.import_tdt_channel_data(folderpath, stream_name="NPr1",
                                                                        stim_name="Wav1", sync_present=True)
        yield fs, raw, stim_data, fs_sync

    def test_import_tdt_channel_data_all_channels(self, all_channels_output):
        fs, raw, stim_data, fs_sync = all_channels_output

        assert fs == pytest.approx(24414.0625)
        assert fs_sync == pytest.approx(24414.0625)
        assert np.sum(raw) == pytest.approx(-1.0475945)
        assert raw[3][238923] == pytest.approx(-3.9779516e-06)
        np.testing.assert_allclose(raw[0][:42],
                                   np.array([1.47801649e-04, 1.27564883e-04, 1.64924393e-04, 1.48525534e-04,
                                             1.35455994e-04, 1.66159371e-04, 1.43361045e-04, 1.33363705e-04,
                                             1.63555334e-04, 1.43828947e-04, 1.34855480e-04, 1.65302947e-04,
                                             1.35081427e-04, 1.35835842e-04, 1.50924738e-04, 1.10977657e-04,
                                             1.12502399e-04, 1.29641427e-04, 8.84166002e-05, 1.10677531e-04,
                                             1.15529976e-04, 7.78901813e-05, 1.06039581e-04, 1.01933379e-04,
                                             7.12062683e-05, 9.83324280e-05, 8.50119322e-05, 6.22217049e-05,
                                             9.13965632e-05, 8.03805233e-05, 8.01165152e-05, 1.15176997e-04,
                                             1.03646627e-04, 1.02100421e-04, 1.38182586e-04, 1.15387054e-04,
                                             1.16143441e-04, 1.45824844e-04, 1.12533606e-04, 1.16105992e-04,
                                             1.43227662e-04, 1.03025362e-04]))

        def test_import_tdt_channel_data_single_channel(self, all_channels_output, folderpath):
            fs, raw, stim_data, fs_sync = all_channels_output

            np.testing.assert_allclose(raw[2], importing.import_tdt_channel_data(folderpath, ch=3, t2=28.72)[1])
            np.testing.assert_allclose(raw[0], importing.import_tdt_channel_data(folderpath, ch=1, t2=28.72)[1])

        def test_import_tdt_channel_data_multiple_channels(self, all_channels_output, folderpath):
            fs, raw, stim_data, fs_sync = all_channels_output

            np.testing.assert_allclose(raw[2:5],
                                       importing.import_tdt_channel_data(folderpath, ch=[3, 4, 5], t2=28.72)[1])

    class TestImportOpenEphysChannelData:

        @pytest.fixture
        def input_data(self):
            folderpath = '../../test_data/OE_test_data/Record Node 102'
            experiment = 'experiment1'
            recording = 'recording1'
            source_processor = 'Rhythm_FPGA-100.0'
            yield folderpath, experiment, recording, source_processor

        @pytest.fixture
        def all_channels_output(self, input_data):
            folderpath, experiment, recording, source_processor = input_data
            fs, neural_data_au, sync_data = importing.import_open_ephys_channel_data(folderpath, experiment, recording,
                                                                                     source_processor=source_processor)
            yield fs, neural_data_au, sync_data

        def test_import_open_ephys_channel_data_all_channels(self, all_channels_output):
            fs, neural_data_au, sync_data = all_channels_output
            assert fs == pytest.approx(30000.0) and sync_data is None
            assert np.sum(neural_data_au) == 1018188679
            assert neural_data_au[18][2837:2958].tolist() == np.array(
                [-1990, -2076, -2097, -2078, -2097, -2108, -2119, -2097, -2118, -2118, -2171, -2192, -2192, -2203,
                 -2159,
                 -2118, -2126, -2136, -2118, -2118, -2118, -2118, -2121, -2117, -2117, -2117, -2117, -2117, -2117,
                 -2117,
                 -2117, -2126, -2138, -2133, -2122, -2133, -2140, -2138, -2138, -2118, -2116, -2116, -2116, -2116,
                 -2116,
                 -2106, -2094, -2094, -2073, -2060, -2062, -1987, -1977, -1950, -1976, -1965, -1932, -1917, -1912,
                 -1901,
                 -1821, -1805, -1776, -1784, -1794, -1800, -1771, -1730, -1720, -1602, -1602, -1601, -1601, -1584,
                 -1558,
                 -1527, -1516, -1473, -1420, -1391, -1309, -1302, -1308, -1279, -1223, -1197, -1175, -1164, -1159,
                 -1143,
                 -1132, -1100, -1090, -1090, -1084, -1090, -1051, -1027, -1015, -994, -950, -955, -950, -918, -901,
                 -887,
                 -885, -811, -768, -750, -737, -731, -726, -707, -678, -659, -651, -630, -614, -577, -577]).tolist()

        def test_import_open_ephys_channel_data_single_channel(self, all_channels_output, input_data):
            fs, neural_data_au, sync_data = all_channels_output
            folderpath, experiment, recording, source_processor = input_data

            assert neural_data_au[5].tolist() == \
                   importing.import_open_ephys_channel_data(folderpath, experiment, recording, channels=[5],
                                                            source_processor=source_processor)[1].tolist()
            assert neural_data_au[0].tolist() == \
                   importing.import_open_ephys_channel_data(folderpath, experiment, recording, channels=[0],
                                                            source_processor=source_processor)[
                       1].tolist()  # using channels=0 does not work, has to be a list

        def test_import_open_ephys_channel_data_multiple_channels(self, all_channels_output, input_data):
            fs, neural_data_au, sync_data = all_channels_output
            folderpath, experiment, recording, source_processor = input_data

            assert neural_data_au[21:26].tolist() == \
                   importing.import_open_ephys_channel_data(folderpath, experiment, recording,
                                                            channels=[21, 22, 23, 24, 25],
                                                            source_processor=source_processor)[1].tolist()
