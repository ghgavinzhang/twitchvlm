import argparse
import json
from pathlib import Path
from tqdm import tqdm
import shutil


def build_dataset(frames_dir, chat_json_path, output_dir):
    # Load VLM chat dataset
    with open(chat_json_path) as f:
        chat_data = json.load(f)
    chat_map = {entry["id"]: entry["conversations"][1]["value"] for entry in chat_data}

    # Sort frames and filter by available chat
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    frames = [f for f in frames if f.stem in chat_map]

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(frames) - 1), desc="Processing frame pairs"):
        frame_0 = frames[i]
        frame_1 = frames[i + 1]

        frame_id = frame_0.stem
        if frame_id not in chat_map:
            continue

        chat_text = chat_map[frame_id]
        out_folder = output_dir / f"{i:07d}"
        out_folder.mkdir(parents=True, exist_ok=True)

        seed = int(frame_0.stem[-6:])

        # Copy frame images
        shutil.copy(frame_0, out_folder / f"{seed:06d}_0.jpg")
        shutil.copy(frame_1, out_folder / f"{seed:06d}_1.jpg")

        # Save prompt.json without caption
        prompt = {
            "output": f"Generate the next frame that follows this Twitch chat reaction: {chat_text}"
        }
        with open(out_folder / "prompt.json", "w") as f:
            json.dump(prompt, f, indent=2)

        # Write dummy metadata
        metadata = {
            "seed": seed,
            "p2p_threshold": 0.5,
            "cfg_scale": 7.5,
            "clip_sim_0": 1.0,
            "clip_sim_1": 1.0,
            "clip_sim_dir": 1.0,
            "clip_sim_image": 1.0,
        }
        with open(out_folder / "metadata.jsonl", "w") as f:
            f.write(json.dumps(metadata) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build fake prompt2prompt dataset from frame pairs.")
    parser.add_argument("--frames-dir", type=str, required=True, help="Directory of input video frames.")
    parser.add_argument("--chat-json", type=str, required=True, help="Path to vlm_chat_dataset.json.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write fake dataset.")
    args = parser.parse_args()

    build_dataset(
        frames_dir=Path(args.frames_dir),
        chat_json_path=Path(args.chat_json),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    # Example command: python3 generate_pix2pix_dataset.py --frames-dir ./frames --chat-json ./vlm_chat_dataset.json --output-dir ./instruct-pix2pix/data/instruct-pix2pix-dataset-000
    main()