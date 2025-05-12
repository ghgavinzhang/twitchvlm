# Scripts to run

For data collection, download the Twitch downloader GUI at https://github.com/lay295/TwitchDownloader/releases.
Use the GUI to download the video and the chat. 
Run extract_frames.sh to process the video into frames, and process.py to create the dataset.json for VLM finetuning. 

For VLM finetuning, clone the [https://github.com/ghgavinzhang/twitchvlm/edit/master/README.md](https://github.com/2U1/Llama3.2-Vision-Finetune) repo and run the finetune_twitch.sh script. Run generate_captions.py on the finetuned model to get the captions as text output, and then use the TwitchDownloader GUI to render the final output.
