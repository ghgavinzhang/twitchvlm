import os
import subprocess
from PIL import Image
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import MllamaForConditionalGeneration, AutoProcessor

import torch
from tqdm import tqdm
from IPython import embed
import json
#cp /home/gavinzhang/rlpd_vlm/checkpoint_add_files/* /nfs/kun2/users/gavin/vlm_checkpoints/model/checkpoint-3318/

# ==== CONFIGURATION ====
video_path = "test.mp4"  # ← input video
frame_dir = "frames"
output_video_path = "twitch_overlay_output.mp4"
model_path = "/nfs/kun2/users/gavin/vlm_checkpoints/model3"
image_processor_name = "openai/clip-vit-large-patch14-336"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
chat_save_path = "chat_messages_new_6epoch_selfhistory.json"


sample_interval_sec = 10        # sample frame every N seconds
chat_display_duration = 3       # how long each chat message stays visible
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # or update this
font_size = 24

# # ==== STEP 1: Extract frames ====
# os.makedirs(frame_dir, exist_ok=True)
# print("⏳ Extracting frames from video...")
# subprocess.run([
#     "ffmpeg", "-i", video_path,
#     "-vf", f"fps=1/{sample_interval_sec}",
#     f"{frame_dir}/frame_%04d.jpg",
#     "-hide_banner", "-loglevel", "error"
# ])

# Load prior chat data
with open("vlm_chat_dataset_deduped.json") as f:
    prior_chat_data = json.load(f)

# Map frame ID to previous chat string
frame_to_prev_chat = {}
for entry in prior_chat_data:
    frame_id = entry["id"]  # e.g., frame_001079
    human_msg = next((m["value"] for m in entry["conversations"] if m["from"] == "human"), None)
    if human_msg:
        # Extract just the previous chat lines (between "Previous chat:\n" and "\n\nInstruction:")
        try:
            prev = human_msg.split("Previous chat:\n", 1)[1].split("\n\nInstruction:")[0].strip()
        except IndexError:
            prev = ""
        frame_to_prev_chat[frame_id] = prev

# ==== STEP 2: Load model and processors ====
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)

# ==== STEP 3: Generate chat messages ====
frame_files = frame_files = [f"frame_{i:06d}.jpg" for i in range(1141, 1207)]  #sorted(os.listdir(frame_dir))
chat_messages = []
previous_text = "none yet"
for idx, frame_file in tqdm(enumerate(frame_files), total=len(frame_files), desc="Generating chat"):
    image_path = os.path.join(frame_dir, frame_file)
    image = Image.open(image_path).convert("RGB")

    # frame_id = frame_file[:-4]  # strip ".jpg"
    # previous_text = frame_to_prev_chat.get(frame_id, "(none yet)")

    qs = (
        "<image>\n"
        f"Previous chat:\n{previous_text}\n\n"
        "Instruction:\nYou are watching a live Twitch stream of Teamfight Tactics. "
        "Your job is to respond like a real Twitch chatter reacting to the current game state."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": qs}
            ]
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=64)

    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    output_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    timestamp = idx * sample_interval_sec
    chat_messages.append((timestamp, output_text))
    previous_text = output_text



chat_messages_json = [{"timestamp": ts, "message": msg} for ts, msg in chat_messages]
with open(chat_save_path, "w") as f:
    json.dump(chat_messages_json, f, indent=2)

print(f"💾 Saved chat messages to {chat_save_path}")

