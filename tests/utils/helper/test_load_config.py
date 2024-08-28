import pytest
import os
from neurokin.utils.helper import load_config


@pytest.fixture
def empty_config_file():
    config_file_path = "empty_config.yaml"
    with open(config_file_path, "w") as f:
        pass
    yield config_file_path
    os.remove(config_file_path)


class TestReadConfig:

    def test_read_config_with_missing_file(self):
        fake_path = ""
        with pytest.raises(FileNotFoundError) as message:
            load_config.read_config(fake_path)
        assert message.match("Could not find the config file at " + fake_path + " \n Please make sure the path is correct and the file exists")

    def test_read_config_with_empty_file(self, empty_config_file):
        empty_config_file_path = empty_config_file
        config = load_config.read_config(empty_config_file_path)
        assert config is None

    def test_read_config_with_custom_file(self):
        config_file_path = "../../test_data/config_test/config.yaml"
        config = load_config.read_config(config_file_path)
        assert config == {'skeleton':
                              {'angles':
                                   {'joints':
                                        {'left_crest': ['lshoulder', 'lcrest', 'lhip'],
                                         'left_hip': ['lcrest', 'lhip', 'lknee'],
                                         'left_knee': ['lhip', 'lknee', 'lankle'],
                                         'left_ankle': ['lknee', 'lankle', 'lmtp'],
                                         'right_crest': ['rshoulder', 'rcrest', 'rhip'],
                                         'right_hip': ['rcrest', 'rhip', 'rknee'],
                                         'right_knee': ['rhip', 'rknee', 'rankle'],
                                         'right_ankle': ['rknee', 'rankle', 'rmtp'],
                                         'trunk_cross_l': ['lshoulder', 'rcrest', 'lhip'],
                                         'trunk_cross_r': ['rshoulder', 'lcrest', 'rhip']},
                                    'references':
                                        {'full_body_tilt': ['lshoulder', 'lmtp', 'mcorner'],
                                         'maze_corner_open_right': ['mcorner_ol', 'mcorner_or', 'mcorner_cr']}},
                               'distances':
                                   {'lef_shoulder_to_toe': ['lshoulder', 'lmtp']},
                               'elevations':
                                   {'left_hip': 'lhip',
                                    'right_hip': 'rhip'}},
                          'features': {'joint_angles_dlc2kin.JointAnglesDLC': None,
                                       'joint_angles_dlc2kin.AngularVelocityDLC': None,
                                       'momentum_dlc2kin.AccelerationDLC': None,
                                       'momentum_dlc2kin.SpeedDLC': None},
                          'events':
                              {'freezing_of_gait':
                                   {'get_frames_where_feature_of_marker_id_is_below_threshold':
                                        {'marker_id': 'rknee',
                                         'feature': 'speed',
                                         'threshold': 5},
                                    'get_freezing_events': ['(marker_id', 'center_of_gravity)', '(min_duration', '0.5)'],
                                    'get_predictions': ['(model_path', 'path)', '(hyperparameters', 'dictionary_of_hyperparams)']}}}
