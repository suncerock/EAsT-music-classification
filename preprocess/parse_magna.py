import json
import os
from tqdm import tqdm
import soundfile as sf
import warnings
warnings.filterwarnings('ignore', message='PySoundFile failed')

import librosa
import numpy as np
import pandas as pd

def parse_magna(data_root, feature_root, output_dir):
    label_file = os.path.join(data_root, "annotations_final.csv")
    df_label = pd.read_csv(label_file, sep="\t")

    col_names = df_label.columns.to_list()[1:-1]
    label_count = df_label.to_numpy()[:, 1:-1].astype(np.int32).sum(axis=0)

    top_50_col_index = np.argsort(-label_count)[:50]
    top_50_col_names = [col_names[i] for i in top_50_col_index]

    df_label = df_label[["clip_id"] + top_50_col_names + ["mp3_path"]]
    
    if not os.path.exists(os.path.join(data_root, "wav")):
        os.mkdir(os.path.join(data_root, "wav"))
    
    data_list = []
    for _, line in tqdm(df_label.iterrows()):
        clip_id = line["clip_id"]
        mp3_path = os.path.join(data_root, line["mp3_path"])
        label = line.values[1:-1].astype(np.int32)
        label[label == 0] = -1
        wav_path = os.path.join(data_root, "wav/{}.wav".format(clip_id))
        
        try:
            y, _ = librosa.load(mp3_path, sr=16000)
            sf.write(wav_path, y, samplerate=16000)
        except Exception:
            print(mp3_path)
            continue

        vggish = os.path.join(feature_root, 'vggish', '{}.npy'.format(clip_id))
        # if not os.path.exists(vggish):
        #     warnings.warn("Missing VGGish feature: {}".format(vggish))
        
        openl3 = os.path.join(feature_root, 'openl3', '{}.npy'.format(clip_id))
        # if not os.path.exists(openl3):
        #     warnings.warn("Missing Open-L3 feature: {}".format(openl3))

        passt = os.path.join(feature_root, 'passt', '{}.npy'.format(clip_id))
        # if not os.path.exists(passt):
        #     warnings.warn("Missing PaSST feature: {}".format(passt))

        pann = os.path.join(feature_root, 'pann', '{}.npy'.format(clip_id))
        # if not os.path.exists(pann):
        #     warnings.warn("Missing pann feature: {}".format(pann))

        data = dict(
            clip_id=clip_id,
            audio_path=wav_path,
            label=label.tolist(),
            vggish=vggish,
            openl3=openl3,
            passt=passt,
            pann=pann
        )

        data_list.append(data)
    
    with open(os.path.join(output_dir, "magna.json"), "w") as f:
        for data in data_list:
            json.dump(data, f)
            f.write('\n')
            f.flush()

if __name__ == '__main__':
    import fire

    fire.Fire(parse_magna)