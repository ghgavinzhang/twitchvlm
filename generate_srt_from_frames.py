import os
import torch
from PIL import Image
from datetime import timedelta
from transformers import AutoProcessor, AutoModelForVision2Seq

import argparse

def to_srt_time(seconds):
    return str(timedelta(seconds=seconds)).replace(".", ",").zfill(8)

def main(frame_dir, output_srt, fps_interval, model_path):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path).to("cuda").eval()

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    captions = []

    for i, fname in enumerate(frame_files):
        img_path = os.path.join(frame_dir, fname)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)

        caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        timestamp = i * fps_interval
        captions.append((timestamp, caption))

    with open(output_srt, "w", encoding="utf-8") as f:
        for i, (start, text) in enumerate(captions, 1):
            end = start + fps_interval
            f.write(f"{i}\n{to_srt_time(start)} --> {to_srt_time(end)}\n{text}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", type=str, required=True)
    parser.add_argument("--output_srt", type=str, required=True)
    parser.add_argument("--fps_interval", type=int, default=5)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
