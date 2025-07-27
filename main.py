import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import librosa
import hydra
import numpy as np
from inference import InferenceHandler

import argparse
import sys

# Capture CLI args before Hydra gets involved
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Path to input audio file")
parser.add_argument("-o", "--output", required=True, help="Path to output MIDI file")
cli_args, hydra_args = parser.parse_known_args()

# Remove known args so Hydra doesn't try to parse them
sys.argv = [sys.argv[0]] + hydra_args


def get_predictions(
    model,
    eval_audio_file,
    outpath,
    mel_norm=True,
    eval_dataset="Slakh",
    verbose=True,
    contiguous_inference=False,
    use_tf_spectral_ops=False,
    batch_size=8,
    max_length=1024
):  
    handler = InferenceHandler(
        model=model,
        device=torch.device('cuda'),
        mel_norm=mel_norm,
        contiguous_inference=contiguous_inference,
        use_tf_spectral_ops=use_tf_spectral_ops
    )

    def load_audio(fname):
        audio, _ = librosa.load(fname, sr=16000)
        if eval_dataset == "NSynth":
            audio = np.pad(audio, (int(0.05 * 16000), 0), "constant", constant_values=0)
        return audio

    if verbose:
        print("Running inference on:", eval_audio_file)

    audio = load_audio(eval_audio_file)

    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    handler.inference(
        audio=audio,
        audio_path=eval_audio_file,
        outpath=outpath,
        batch_size=batch_size,
        max_length=max_length,
        verbose=verbose
    )

    if verbose:
        print("Saved prediction to:", outpath)

@hydra.main(config_path="config", config_name="config_slakh_segmem", version_base="1.1")
def main(cfg):
    input_audio_file = cli_args.input
    output_midi_path = cli_args.output
    assert os.path.isfile(input_audio_file), f"Audio file not found: {input_audio_file}"
    
    assert cfg.path, "Missing model checkpoint path"

    print(f"Loading weights from: {cfg.path}")
    pl = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)

    if cfg.path.endswith(".ckpt"):
        model_cls = hydra.utils.get_class(cfg.model._target_)
        print("torch.cuda.device_count():", torch.cuda.device_count())
        pl = model_cls.load_from_checkpoint(
            cfg.path,
            config=cfg.model.config,
            optim_cfg=cfg.optim,
        )
        model = pl.model
    else:
        model = pl.model
        strict_mode = cfg.eval.get("load_weights_strict", False)
        model.load_state_dict(torch.load(cfg.path), strict=strict_mode)

    model.eval()

    mel_norm = "pretrained/mt3.pth" not in cfg.path

    get_predictions(
        model=model,
        eval_audio_file=input_audio_file,
        outpath=output_midi_path,
        mel_norm=mel_norm,
        eval_dataset=cfg.eval.eval_dataset,
        contiguous_inference=cfg.eval.contiguous_inference,
        use_tf_spectral_ops=cfg.eval.use_tf_spectral_ops,
        batch_size=cfg.eval.batch_size,
    )
    print("Current working dir:", os.getcwd())


if __name__ == "__main__":
    main()
