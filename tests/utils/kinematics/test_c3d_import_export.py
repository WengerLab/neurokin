import pytest
import numpy as np
import pandas as pd
from neurokin.utils.kinematics import c3d_import_export

C3D_PATH_1 = '../../test_data/neural_correlates_test_data/230428/NWE00159/15/runway15.c3d'
C3D_PATH_2 = '../../test_data/neural_correlates_test_data/230619/NWE00163/13/runway_13.c3d'


def get_key(input_value):
    if isinstance(input_value, list):
        return tuple(input_value)
    return input_value


class TestImportC3D:

    def get_c3d_labels(self, handle):
        key = get_key(str(handle))

        with open(C3D_PATH_1, "rb") as handle_1:
            return_values = {get_key(str(handle_1)): ['rshoulder', 'rcrest', 'rhip', 'rknee', 'rankle', 'rmtp',
                                                      'lshoulder', 'lcrest', 'lhip', 'lknee', 'lankle', 'lmtp']}

        with open(C3D_PATH_2, "rb") as handle_2:
            return_values[get_key(str(handle_2))] = ['rshoulder', 'rcrest', 'rhip', 'rknee', 'rankle', 'rmtp',
                                                     'lshoulder', 'lcrest', 'lhip', 'lknee', 'lankle', 'lmtp',
                                                     '*12', '*13', '*14', '*15', '*16']
        result = return_values.get(key, "default_output")
        return result

    def create_empty_df(self, scorer, bodyparts, frames_no):
        df = {}
        coords = ['x', 'y', 'z']
        for bodypart in bodyparts:
            for coord in coords:
                df[(scorer, bodypart, coord)] = np.empty((frames_no))
        return pd.DataFrame(df)

    def test_import_c3d_with_path_1(self, mocker):
        mocker.patch('neurokin.utils.kinematics.c3d_import_export.create_empty_df',
                     side_effect=self.create_empty_df)
        mocker.patch('neurokin.utils.kinematics.c3d_import_export.get_c3d_labels',
                     side_effect=self.get_c3d_labels)
        result = c3d_import_export.import_c3d(C3D_PATH_1)
        assert result[0] == 823 and result[1] == 1316 and result[2] == pytest.approx(200.0)
        assert result[3].to_numpy().sum() == pytest.approx(1524390.6)
        np.testing.assert_allclose(result[3].to_numpy()[0],
                                   np.array([103.01258, -411.10077, 68.74401, 90.08401, -504.80438, 76.760345, 74.72459,
                                             -531.19867, 60.693214,
                                             104.548, -521.9521, 34.099796, 82.267, -558.17737, 25.414665, 95.49268,
                                             -557.6948, 8.591636,
                                             88.909584, -412.69595, 68.28553, 57.04074, -490.7331, 76.41941, 39.24682,
                                             -512.6552, 54.764286,
                                             44.36522, -480.46832, 44.55431, 41.618965, -501.12137, 13.386004, 51.77887,
                                             -482.21115, 8.363125]))
        assert result[3].to_numpy()[10][25] == pytest.approx(-501.5426)

    def test_import_c3d_with_path_2(self, mocker):
        mocker.patch('neurokin.utils.kinematics.c3d_import_export.create_empty_df',
                     side_effect=self.create_empty_df)
        mocker.patch('neurokin.utils.kinematics.c3d_import_export.get_c3d_labels',
                     side_effect=self.get_c3d_labels)
        result = c3d_import_export.import_c3d(C3D_PATH_2)
        assert result[0] == 3903 and result[1] == 5771 and result[2] == pytest.approx(200.0)
        assert result[3].to_numpy().sum() == pytest.approx(-2189393.8)
        np.testing.assert_allclose(result[3].to_numpy()[0], np.array([68.69498, -501.1288, 57.40453, 86.684166,
                                                                      -592.5322, 81.13452, 88.05857, -621.1553,
                                                                      56.49255, 113.13604, -589.824, 44.444004,
                                                                      114.26332, -629.649, 15.165457, 134.51573,
                                                                      -623.157, 6.4416075, 32.920746, -518.92676,
                                                                      57.184517, 52.78712, -588.1357, 85.892296,
                                                                      47.983295, -617.79614, 61.004707, 28.009104,
                                                                      -575.7024, 44.99936, 31.88251, -594.2461,
                                                                      12.577683, 32.621742, -573.4188, 9.076784,
                                                                      -2.1502872, -657.41956, -9.539355, 0.,
                                                                      0., 0., 0., 0.,
                                                                      0., 0., 0., 0.,
                                                                      0., 0., 0.]))
        assert result[3].to_numpy()[10][25] == pytest.approx(-615.9607)


def test_create_empty_df():
    frames_no = 2
    bodyparts = ['lmtp', 'lankle', 'rmtp', 'rankle']

    df_expected_dict = {}
    for bodypart in bodyparts:
        for coord in ['x', 'y', 'z']:
            df_expected_dict[('scorer', bodypart, coord)] = np.full(frames_no, np.nan)
    df_expected = pd.DataFrame(df_expected_dict)

    df_actual = c3d_import_export.create_empty_df('scorer', bodyparts, frames_no)

    pd.testing.assert_frame_equal(df_actual, df_expected, check_names=False, check_like=True)


class TestGetC3DLabels:

    def test_get_c3d_labels_with_path_1(self):
        with open(C3D_PATH_1, "rb") as handle_1:
            assert c3d_import_export.get_c3d_labels(handle_1) == ['rshoulder', 'rcrest', 'rhip', 'rknee', 'rankle',
                                                                  'rmtp', 'lshoulder', 'lcrest', 'lhip', 'lknee',
                                                                  'lankle', 'lmtp']

    def test_get_c3d_labels_with_path_2(self):
        with open(C3D_PATH_2, "rb") as handle_2:
            assert c3d_import_export.get_c3d_labels(handle_2) == ['rshoulder', 'rcrest', 'rhip', 'rknee', 'rankle',
                                                                  'rmtp', 'lshoulder', 'lcrest', 'lhip', 'lknee',
                                                                  'lankle', 'lmtp', '*12', '*13', '*14', '*15', '*16']
