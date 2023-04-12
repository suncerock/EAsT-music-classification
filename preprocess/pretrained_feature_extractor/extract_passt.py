import os
import soundfile as sf
import sys
sys.path.append("../../../src/hear21passt")
from tqdm import tqdm

import librosa
import numpy as np
import torch

from hear21passt.base import load_model, get_timestamp_embeddings


def extract_passt_feature(input_path, output_path, model):
    """
    Extract passt feature for one wavefile and save as an .npy file
    
    Parameters
    ----------
    input_path : str
        path to the audio file to extract the feature
    output_path : str
        path to save the output feature
    input_str : int
        sampling rate of the input file
    """

    audio, _ = librosa.load(input_path, sr=32000)
    audio = torch.from_numpy(audio[np.newaxis]).float()
    output, _ = get_timestamp_embeddings(audio, model)
    output = output.cpu().detach().squeeze(dim=0).numpy()
    np.save(output_path, output)

def extract_multiple_passt_feature(input_dir, output_dir, device):
    """
    Extract vggish passt for multiple wav files and save as .npy files
    
    Parameters
    ----------
    input_dir : str
        directory of the input wav files
    output_dir : str
        directory of the output feature files
    """
    input_file_list = os.listdir(input_dir)

    model = load_model(mode="embed_only", timestamp_window=960, timestamp_hop=960)
    model = model.to(device)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for input_file in tqdm(input_file_list):
        input_path = os.path.join(input_dir, input_file)

        output_file = input_file.replace('.wav', '.npy')
        output_path = os.path.join(output_dir, output_file)

        extract_passt_feature(input_path, output_path, model)


if __name__ == '__main__':
    import fire

    fire.Fire()