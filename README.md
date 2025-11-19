# What is this?
This repository contains scripts related to audio data of CASTELLA.
- Downloader
- Feature extractor using MS-CLAP 2023

Due to changes in the specifications of the download source, the downloader may not function correctly. If this happens, please open an issue

# How to use?
### Download audio
```bash
uv sync
uv run script/download.py {JSON FILE OF CASTELLA}  # e.g., `CASTELLA/json/en/train.json`
uv run script/download.py {JSON FILE OF CASTELLA} -v  # If you want to download video
```
- Download script requires `FFmpeg`.
- Downloaded audio/video is in `./download`.
- `yt-dlp` may require a newer version than the one specified in pyploject.toml.

### Feature extractor using CLAP
```bash
uv run script/extract_audio_feature.py {JSON FILE OF CASTELLA}  # e.g., `CASTELLA/json/en/train.json`
uv run script/extract_text_feature.py {AUDIO FOLDER}  # e.g., ./download/audio
```
