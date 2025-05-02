mkdir frames
ffmpeg -i input_video.mp4 -vf "fps=1/5" frames/frame_%06d.jpg
