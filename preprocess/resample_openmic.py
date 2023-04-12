import argparse
import os
import soundfile as sf
import shutil
from tqdm import tqdm

import librosa

def resample_openmic(data_root, target_root, target_sr=44100):
    """
    Resample the OpenMIC data

    Parameters
    ----------
    data_root : str
        root of the openmic dataset
    target_root : str
        root to write the output data
    target_sr : int
        target sampling rate
    """
    data_root_audio = os.path.join(data_root, 'audio')
    target_root_audio = os.path.join(target_root, 'audio')
    if not os.path.exists(target_root_audio):
        os.mkdir(target_root_audio)

    for subdir in os.listdir(data_root_audio):
        target_subdir = os.path.join(target_root_audio, subdir)
        if not os.path.exists(target_subdir):
            os.mkdir(target_subdir)
        for filename in tqdm(os.listdir(os.path.join(data_root_audio, subdir))):
            input_file = os.path.join(data_root_audio, subdir, filename)
            output_file = os.path.join(target_root_audio, subdir, filename)
            if os.path.exists(output_file):
                os.remove(output_file)

            audio, sr = sf.read(input_file)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if not sr == target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sf.write(output_file, audio, samplerate=target_sr)
    
    shutil.copyfile(
        os.path.join(data_root, 'openmic-2018.npz'),
        os.path.join(target_root, 'openmic-2018.npz')
    )
    shutil.copytree(
        os.path.join(data_root, 'partitions'),
        os.path.join(target_root, 'partitions')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--data_root', type=str, required=True,
                        help="Root of openmic data")
    parser.add_argument('-o', '--target_root', type=str, required=True,
                        help="Root to write the output data")
    parser.add_argument('-sr', '--target_sr', type=int, required=True,
                        help="Target sampling rate")

    args = parser.parse_args()

    resample_openmic(
        data_root=args.data_root,
        target_root=args.target_root,
        target_sr=args.target_sr
    )
