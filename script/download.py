import argparse
import json
import logging
import time
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler("download.log", mode="a", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

MAX_DURATION = 300  # 5 minutes in seconds


def get_option(output_dir, is_video):
    logger.info(f"Output directory: {str(Path(output_dir))}")
    basic_opt = {
        "outtmpl": str(Path(output_dir) / "%(id)s.%(ext)s"),
        "verbose": True,
        "noplaylist": True,
        "download_ranges": get_download_range,
    }

    if not is_video:
        add_opt = {
            "format": "bestaudio/best",
            "extractaudio": True,
            "audioformat": "wav",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "postprocessor_args": [
                "-ar",
                "32000",  # Set sample rate to 32000Hz
                "-ac",
                "1",  # Set number of channels to 1 (mono)
            ],
        }

    else:
        add_opt = {
            "format": "worstvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",  # Convert to mp4 format
                }
            ],
        }

    return {**basic_opt, **add_opt}


def get_download_range(info_dict, ydl):
    duration = info_dict.get("duration", 0)
    if duration and duration > MAX_DURATION:
        logger.info(
            f"This video is longer than {MAX_DURATION} seconds ({int(duration)} seconds). Downloading the first {MAX_DURATION} seconds."
        )
        end_time = MAX_DURATION
    else:
        end_time = duration
    return [{"start_time": 0, "end_time": end_time}]


def download(yids, output_dir, is_video=False):
    ext = "wav" if not is_video else "mp4"
    yids = [yid for yid in yids if not (output_dir / f"{yid}.{ext}").exists()]
    logger.info(f"Downloading {len(yids)} videos...")
    options = get_option(output_dir, is_video)
    dlp = yt_dlp.YoutubeDL(options)
    for yid in yids:
        try:
            logger.info(f"Starting download for video {yid}")
            dlp.download([yid])
            logger.info(f"Successfully downloaded video {yid}")
        except Exception as e:
            logger.error(f"Error downloading video {yid}: {e}")
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download audio or video")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument(
        "--video", action="store_true", help="Download video instead of audio"
    )
    args = parser.parse_args()
    is_video = args.video

    # Load data from input JSON file
    with open(args.input_file, "r") as infile:
        data = json.load(infile)

    yids = [item["yid"] for item in data]
    output_dir = (
        Path("download") / "audio" if not is_video else Path("download") / "video"
    )
    download(yids, output_dir=output_dir, is_video=is_video)
