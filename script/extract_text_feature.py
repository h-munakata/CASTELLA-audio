import argparse
import json
from pathlib import Path

import numpy as np
import torch
from msclap import CLAP
from tqdm import tqdm


def save_text_feat(json_path, output_dir, extractor):
    if not json_path.exists():
        print(f"{json_path} does not exist.")
        return

    with open(json_path) as f:
        data = json.load(f)

    for _d in tqdm(data):
        for idx, _m in enumerate(_d["moments"]):
            feat, proj_feat = extractor.extract_text_feats(_m["local_caption"])

            path_feats = output_dir / f"qid{_d['yid']}_{idx + 1}.npz"
            np.savez(path_feats, last_hidden_state=feat)


class ClapExtractor:
    def __init__(self):
        # if gpu is available, use it
        self.use_cuda = torch.cuda.is_available()
        self.wrapper = CLAP(use_cuda=self.use_cuda, version="2023")
        self.text_enc = self.wrapper.clap.caption_encoder
        if self.use_cuda:
            print("Inference on GPU")
            self.text_enc = self.text_enc.cuda()
        else:
            print("Inference on CPU")

    @torch.no_grad()
    def extract_text_feats(self, text):
        x = self.wrapper.preprocess_text([text])
        mask = x["attention_mask"]
        len_output = torch.sum(mask, dim=-1, keepdims=True)
        out = self.text_enc.base(**x)
        hidden_states = out[0]
        pooled_output = out[1]

        if "clip" in self.text_enc.text_model:
            out = self.clip_text_projection(pooled_output)  # get CLS token output
        elif "gpt" in self.text_enc.text_model:
            batch_size = x["input_ids"].shape[0]
            sequence_lengths = (
                torch.ne(x["input_ids"], 0).sum(-1) - 1
            )  # tensor([13, 14, 18, 17])
            out = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths,
            ]  # [batch_size, 768] = [4, 768]
        else:
            out = hidden_states[:, 0, :]  # get CLS token output

        projected_feat = self.text_enc.projection(out)

        feat = hidden_states[0, :len_output].cpu().numpy()
        proj_feat = projected_feat.cpu().numpy()

        return feat, proj_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path, default=None, help="text data path")
    args = parser.parse_args()

    output_dir = Path("features") / "castella" / "clap_text"
    output_dir.mkdir(exist_ok=True, parents=True)

    extractor = ClapExtractor()
    save_text_feat(args.json_path, output_dir, extractor)
