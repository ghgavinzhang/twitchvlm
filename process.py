import json
from glob import glob
from pathlib import Path

FRAME_INTERVAL = 10  # seconds
WINDOW = 10  # seconds of chat per frame

# Load chat
with open("chat.json", "r") as f:
    comments = json.load(f)["comments"]

# Sort by timestamp
comments = sorted(comments, key=lambda x: x["content_offset_seconds"])

# Track used messages to avoid overlaps
used_chat_ids = set()

dataset = []
frame_paths = sorted(glob("frames/frame_*.jpg"))

for idx, frame_path in enumerate(frame_paths):
    start_time = idx * FRAME_INTERVAL
    end_time = start_time + WINDOW

    image_filename = Path(frame_path).name
    frame_id = Path(frame_path).stem

    # Collect messages in this time window
    window_msgs = []
    for c in comments:
        if c["_id"] in used_chat_ids:
            continue
        t = c["content_offset_seconds"]
        if start_time <= t < end_time:
            window_msgs.append(c["message"]["body"])
            used_chat_ids.add(c["_id"])

    if window_msgs:
        joined_text = "\n".join(window_msgs)
        sample = {
            "id": frame_id,
            "image": image_filename,
            "conversations": [
                {"from": "human", "value": "<image>"},
                {"from": "gpt", "value": joined_text}
            ]
        }
        dataset.append(sample)

# ✅ Save as proper .json (entire dataset is one list)
with open("vlm_chat_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)
