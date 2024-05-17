import io
from pathlib import Path
import random

import PIL
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import ToTensor

STEP = 0


def plot_spectrogram_to_buf(spectrogram_tensor, name=None):
    plt.figure(figsize=(20, 5))
    plt.imshow(spectrogram_tensor)
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def log_predictions(
        wb,
        gen_audio,
        wave,
        audio_path,
        logs,
        examples_to_log=10,
        *args,
        **kwargs,
):
    # if wb.writer is None:
    #     return

    tuples = list(zip(gen_audio, wave, audio_path, logs['ground_truth'], logs['d_step']))
    print(len(gen_audio), len(wave), len(audio_path))
    print(len(tuples))
    rows = {}
    for pred, target, audio_path, truth, d_step in tuples[:examples_to_log]:
        print('here')
        rows[Path(audio_path).name] = {
            "predicted_audio": wb.Audio(pred.squeeze(0).detach().cpu().numpy(),
                                        sample_rate=22050),
            "target_audio": wb.Audio(target.squeeze(0).detach().cpu().numpy(),
                                     sample_rate=22050),
            "ground_truth": wb.Audio(truth.squeeze(0).detach().cpu().numpy(),
                                     sample_rate=22050),
            "d_step": d_step
        }
    add_table(wb, "predictions", pd.DataFrame.from_dict(rows, orient="index"), step=logs['step'])


def log_spectrogram(wb, spectrogram_batch):
    spectrogram = random.choice(spectrogram_batch.cpu())
    image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
    wb.add_image("spectrogram", ToTensor()(image))


def add_table(wb, table_name, table: pd.DataFrame, step=None):
    wb.log({f"{table_name}_train_distillation": wb.Table(dataframe=table)}, step=step)
