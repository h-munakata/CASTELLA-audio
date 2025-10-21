import argparse
from pathlib import Path

import numpy as np
import torch
from msclap import CLAP
from tqdm import tqdm


def save_audio_feat(list_wav, output_dir, extractor):
    for path_wav in tqdm(list_wav):
        path_feats = output_dir / f"{path_wav.stem}.npz"
        if path_feats.exists():
            continue
        feat, proj_feat = extractor.extract_audio_feats(str(path_wav))

        np.savez(path_feats, features=feat)


class ClapExtractor:
    def __init__(self, win_sec, hop_sec):
        # if gpu is available, use it
        self.use_cuda = torch.cuda.is_available()
        self.wrapper = CLAP(use_cuda=self.use_cuda, version="2023")
        if self.use_cuda:
            self.wrapper.clap.caption_encoder = self.wrapper.clap.caption_encoder.cuda()
            print("Inference on GPU")
        else:
            print("Inference on CPU")

        self.sl_win = SlidingWindos(win_sec, hop_sec)

    @torch.no_grad()
    def extract_audio_feats(self, path_wav):
        audio, sr = self.wrapper.read_audio(path_wav, resample=True)
        if audio.shape[1] > 300 * sr:
            audio = audio[:, : int(300 * sr)]
        frames = self.sl_win(audio[0], sr)
        frames = frames.cuda() if self.use_cuda else frames

        feats = self.wrapper.clap.audio_encoder.base(frames)["embedding"]
        proj_feats = self.wrapper.clap.audio_encoder.projection(feats)

        feats = feats.cpu().numpy()
        proj_feats = proj_feats.cpu().numpy()

        return feats, proj_feats


class SlidingWindos:
    def __init__(self, win_sec, hop_sec):
        self.win_sec = win_sec
        self.hop_sec = hop_sec

    def __call__(self, audio, sr):
        """
        Perform sliding window processing on a 1D tensor with center-based cutting.

        Parameters:
        audio (torch.tensor): 1D tensor.
        win_sec (float): Length of each window.
        hop_sec (float): Number of elements to move the window at each step.
        sr (int): Sampling rate.

        Returns:
        torch.tensor: 2D tensor with shape (num_frames, win_length).
        """
        if audio.ndim != 1:
            raise ValueError("Input audio must be 1D tensor.")

        win_length = int(self.win_sec * sr)
        hop_length = int(self.hop_sec * sr)

        frames = audio.unfold(0, win_length, hop_length)

        return torch.tensor(frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", type=Path, help="audio data directory")
    parser.add_argument("--win_sec", type=float, default=1.0, help="window length")
    parser.add_argument("--hop_sec", type=float, default=1.0, help="hop length")
    args = parser.parse_args()

    extractor = ClapExtractor(args.win_sec, args.hop_sec)

    list_wav = sorted(args.audio_dir.glob("*.wav"))
    output_dir = Path("features") / "castella" / "clap"
    output_dir.mkdir(exist_ok=True, parents=True)

    save_audio_feat(list_wav, output_dir, extractor)
