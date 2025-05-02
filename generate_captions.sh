#!/bin/bash

# --- CONFIG ---
VIDEO_INPUT="input_video.mp4"
FRAME_DIR="frames"
OUTPUT_SRT="output.srt"
OUTPUT_VIDEO="captioned_video.mp4"
MODEL_PATH="/path/to/llama-3.2-vision"
FPS_INTERVAL=5

# Step 1: Extract frames
echo "🖼️ Extracting frames every $FPS_INTERVAL seconds..."
mkdir -p "$FRAME_DIR"
ffmpeg -y -i "$VIDEO_INPUT" -vf "fps=1/$FPS_INTERVAL" "$FRAME_DIR/frame_%06d.jpg"

# Step 2: Generate SRT using Python
echo "🧠 Generating subtitles using LLaMA 3.2 Vision..."
python3 generate_srt_from_frames.py \
    --frame_dir "$FRAME_DIR" \
    --output_srt "$OUTPUT_SRT" \
    --fps_interval "$FPS_INTERVAL" \
    --model_path "$MODEL_PATH"

# Step 3: Burn subtitles into the video
echo "🎬 Overlaying subtitles onto video..."
ffmpeg -y -i "$VIDEO_INPUT" -vf subtitles="$OUTPUT_SRT" -c:a copy "$OUTPUT_VIDEO"

echo "✅ Done! Output saved to $OUTPUT_VIDEO"
