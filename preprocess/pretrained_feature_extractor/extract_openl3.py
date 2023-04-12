import os
import soundfile as sf
from tqdm import tqdm

import numpy as np
import openl3

def extract_openl3_feature(input_path, output_path, model):
    """
    Extract vggish file for one wavefile and save as an .npy file
    
    Parameters
    ----------
    input_path : str
        path to the audio file to extract the feature
    output_path : str
        path to save the output feature
    input_str : int
        sampling rate of the input file
    """
    audio, sr = sf.read(input_path)
    output, _ = openl3.get_audio_embedding(audio, sr, hop_size=0.96, model=model)
    np.save(output_path, output)

def extract_multiple_vggish_feature(input_dir, output_dir, dataset="openmic"):
    """
    Extract openl3 file for multiple wav files and save as .npy files
    
    Parameters
    ----------
    input_dir : str
        directory of the input wav files
    output_dir : str
        directory of the output feature files
    input_str : int
        sampling rate of the input file
    """
    input_file_list = os.listdir(input_dir)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = openl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music", embedding_size=512)

    if dataset == "magna":
        for input_file in tqdm(input_file_list):
            input_path = os.path.join(input_dir, input_file)

            output_file = input_file.replace('.wav', '.npy')
            output_path = os.path.join(output_dir, output_file)

            extract_openl3_feature(input_path, output_path, model)
    elif dataset == "openmic":
        for input_subdir in tqdm(input_file_list):
            input_subdir_path = os.path.join(input_dir, input_subdir)
            for input_file in tqdm(os.listdir(input_subdir_path)):
                input_path = os.path.join(input_subdir_path, input_file)

                output_file = input_file.replace('.ogg', '.npy')
                output_path = os.path.join(output_dir, output_file)

                extract_openl3_feature(input_path, output_path, model)
    
if __name__ == '__main__':
    import fire

    fire.Fire()