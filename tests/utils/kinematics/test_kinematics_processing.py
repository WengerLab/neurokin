from neurokin.utils.kinematics import kinematics_processing
import numpy as np
import pandas as pd
import pytest


class TestGetMarkerCoordinatesNames:

    @pytest.fixture
    def mock_df_columns_names(self):
        yield [('lmtp', 'x'), ('lmtp', 'y'), ('lmtp', 'z'), ('lankle', 'x'),
               ('lankle', 'y'), ('lankle', 'z'), ('rmtp', 'x'), ('rmtp', 'y')]

    def test_get_marker_coordinates_names_with_mismatch(self, mock_df_columns_names):
        markers = ['lmtp', 'rankle']
        actual = kinematics_processing.get_marker_coordinates_names(mock_df_columns_names, markers)
        expected = ([], [('lmtp', 'x'), ('lmtp', 'y'), ('lmtp', 'z')])
        assert actual == expected

    def test_get_marker_coordinates_names_with_match(self, mock_df_columns_names):
        markers = ['lmtp', 'lankle']
        actual = kinematics_processing.get_marker_coordinates_names(mock_df_columns_names, markers)
        expected = ([('lankle', 'x'), ('lankle', 'y'), ('lankle', 'z')], [('lmtp', 'x'), ('lmtp', 'y'), ('lmtp', 'z')])
        assert actual == expected


def test_get_marker_coordinate_values():
    df = pd.DataFrame({'lmtp': [1.0,2.0,3.0],
                       'rankle': [10.0,11.0,12.0],
                       'rmtp': [16.0,17.0,18.0]})
    actual = kinematics_processing.get_marker_coordinate_values(df, ['lmtp', 'rankle'], 1)
    expected = np.array([2.0, 11.0])
    np.testing.assert_allclose(np.array(actual), expected)


class TestTiltCorrect:

    @pytest.fixture
    def mock_df(self):
        return pd.DataFrame({('lmtp', 'x'): [1.4, 2.4],
                             ('lmtp', 'y'): [4.5, 0.2],
                             ('lmtp', 'z'): [4.8, 9.2],
                             ('lankle', 'x'): [0.1, 1.1],
                             ('lankle', 'y'): [2.3, 2.3],
                             ('lankle', 'z'): [9.2, 5.2],
                             ('rmtp', 'x'): [8.9, 9.9],
                             ('rmtp', 'y'): [0.2, 9.2],
                             ('rmtp', 'z'): [2.3, 8.2],
                             ('rankle', 'x'): [8.8, 9.8],
                             ('rankle', 'y'): [3.4, 7.8],
                             ('rankle', 'z'): [3.4, 3.5]})

    def test_shift_correct(self, mock_df):
        actual = kinematics_processing.tilt_correct(mock_df, reference_marker=("lmtp", "x"), columns_to_correct=[("rankle", "x"), ("rmtp", "x"), ("lankle", "x"), ("lmtp", "x")])
        expected = pd.DataFrame({('lmtp', 'x'): [0.0, 0.0],
                                 ('lmtp', 'y'): [4.5, 0.2],
                                 ('lmtp', 'z'): [4.8, 9.2],
                                 ('lankle', 'x'): [-1.3, -1.3],
                                 ('lankle', 'y'): [2.3, 2.3],
                                 ('lankle', 'z'): [9.2, 5.2],
                                 ('rmtp', 'x'): [7.5, 7.5],
                                 ('rmtp', 'y'): [0.2, 9.2],
                                 ('rmtp', 'z'): [2.3, 8.2],
                                 ('rankle', 'x'): [7.4, 7.4],
                                 ('rankle', 'y'): [3.4, 7.8],
                                 ('rankle', 'z'): [3.4, 3.5]})

        pd.testing.assert_frame_equal(actual, expected)


class TestShiftCorrect:

    @pytest.fixture
    def mock_df(self):
        return pd.DataFrame({('lmtp', 'x'): [-1.4, -0.9],
                             ('lmtp', 'y'): [4.5, 0.2],
                             ('lmtp', 'z'): [4.8, 9.2],
                             ('lankle', 'x'): [-0.1, -0.3],
                             ('lankle', 'y'): [2.3, 2.3],
                             ('lankle', 'z'): [9.2, 5.2],
                             ('rmtp', 'x'): [-8.9, -9.8],
                             ('rmtp', 'y'): [0.2, 9.2],
                             ('rmtp', 'z'): [2.3, 8.2],
                             ('rankle', 'x'): [-9.8, -9.9],
                             ('rankle', 'y'): [3.4, 7.8],
                             ('rankle', 'z'): [3.4, 3.5]})

    def test_shift_correct(self, mock_df):
        actual = kinematics_processing.shift_correct(mock_df, reference_marker=("rankle", "x"), columns_to_correct=[("rankle", "x"), ("rmtp", "x"), ("lankle", "x"), ("lmtp", "x")])
        expected = pd.DataFrame({('lmtp', 'x'): [8.5, 9.0],
                                 ('lmtp', 'y'): [4.5, 0.2],
                                 ('lmtp', 'z'): [4.8, 9.2],
                                 ('lankle', 'x'): [9.8, 9.6],
                                 ('lankle', 'y'): [2.3, 2.3],
                                 ('lankle', 'z'): [9.2, 5.2],
                                 ('rmtp', 'x'): [1.0, 0.1],
                                 ('rmtp', 'y'): [0.2, 9.2],
                                 ('rmtp', 'z'): [2.3, 8.2],
                                 ('rankle', 'x'): [0.1, 0.0],
                                 ('rankle', 'y'): [3.4, 7.8],
                                 ('rankle', 'z'): [3.4, 3.5]})

        pd.testing.assert_frame_equal(actual, expected)


class TestGetUnilateralDF:

    @pytest.fixture
    def mock_df(self):
        yield pd.DataFrame({'lmtp': [1.3, 6.4],
                             'lankle': [5.0, 2.8],
                             'rmtp': [8.3, 2.8],
                             'rankle': [4.8, 9.8]})

    def test_get_unilateral_df_using_name_starts_with(self, mock_df):
        actual = kinematics_processing.get_unilateral_df(mock_df, side="l", name_starts_with=True)
        expected = pd.DataFrame({'lmtp': [1.3, 6.4], 'lankle': [5.0, 2.8]})
        assert actual.equals(expected)

    def test_get_unilateral_df_using_name_ends_with(self, mock_df):
        actual = kinematics_processing.get_unilateral_df(mock_df, side="ankle", name_ends_with=True)
        expected = pd.DataFrame({'lankle': [5.0, 2.8], 'rankle': [4.8, 9.8]})
        assert actual.equals(expected)

    def test_get_unilateral_df_using_column_names(self, mock_df):
        actual = kinematics_processing.get_unilateral_df(mock_df, column_names=["lmtp", "rankle"])
        expected = pd.DataFrame({'lmtp': [1.3, 6.4], 'rankle': [4.8, 9.8]})
        assert actual.equals(expected)
