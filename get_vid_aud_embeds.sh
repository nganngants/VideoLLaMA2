#!/bin/bash

python3 videollama2/eval/inference_audio_video.py \
            --model-path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
            --video-folder MESC/video_data \
            --question-file mesc_train.json \
            --output-dir embeddings_train/ \
            --dataset MESC