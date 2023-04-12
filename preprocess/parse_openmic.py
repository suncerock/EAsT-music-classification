import json
import os
import warnings

import numpy as np
import pandas as pd


def parse_openmic_data(data_root, feature_root, output_dir):
    """
    Parse the openmic data and generate json line data
    Raise error if missing audio file
    Raise warning if missing feature file

    Parameters
    ----------
    data_root : str
        root of the openmic dataset
    feature_root : str
        root of the extracted feature for openmic
    output_dir : str
        path to write the summary json line file
    """
    openmic_data = np.load(os.path.join(data_root, 'openmic-2018.npz'), allow_pickle=True)
    data_y, data_sample_key = openmic_data['Y_true'], openmic_data['sample_key']
    
    data_y[data_y > 0.5] = 1
    data_y[data_y < 0.5] = -1
    data_y[data_y == 0.5] = 0
    data_y = data_y.astype(np.int32)
    
    split_train = pd.read_csv(os.path.join(data_root, 'partitions/split01_train.csv'), header=None).squeeze("columns")
    split_test = pd.read_csv(os.path.join(data_root, 'partitions/split01_test.csv'), header=None).squeeze("columns")
    split_train = set(split_train)
    split_test = set(split_test)

    train_output = os.path.join(output_dir, 'openmic_train.json')
    test_output = os.path.join(output_dir, 'openmic_test.json')
    
    fout_train = open(train_output, 'w')
    fout_test = open(test_output, 'w')

    for idx, key in enumerate(data_sample_key):

        audio_path = os.path.join(data_root, 'audio', key[:3], key + '.ogg')
        if not os.path.exists(audio_path):
            raise RuntimeError("Audio file not found! {}".format(audio_path))

        vggish = os.path.join(feature_root, 'vggish', key + '.npy')
        if not os.path.exists(vggish):
            warnings.warn("Missing VGGish feature: {}".format(vggish))

        openl3 = os.path.join(feature_root, 'openl3', key + '.npy')
        if not os.path.exists(openl3):
            warnings.warn("Missing Open-L3 feature: {}".format(openl3))

        passt = os.path.join(feature_root, 'passt', key + '.npy')
        if not os.path.exists(passt):
            warnings.warn("Missing PaSST feature: {}".format(passt))

        pann = os.path.join(feature_root, 'pann', key + '.npy')
        if not os.path.exists(pann):
            warnings.warn("Missing Pann feature: {}".format(pann))

        data = dict(
            sample_key=key,
            audio_path=audio_path,
            label=data_y[idx].tolist(),
            vggish=vggish,
            openl3=openl3,
            passt=passt,
            pann=pann
        )

        if key in split_train:
            json.dump(data, fout_train)
            fout_train.write('\n')
            fout_train.flush()
        elif key in split_test:
            json.dump(data, fout_test)
            fout_test.write('\n')
            fout_test.flush()
        else:
            raise RuntimeError('Unknown sample key={}! Abort!'.format(key))
    
    return


if __name__ == '__main__':
    import fire

    fire.Fire(parse_openmic_data)