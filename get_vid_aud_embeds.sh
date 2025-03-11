#!/bin/bash

python3 videollama2/eval/inference_audio_video.py \
            --model-path DAMO-NLP-SG/VideoLLaMA2.1-7B-AV \
            --video-folder MESC/mesc_test \
            --question-file MESC/mesc_test.json \
            --output-dir embeddings/ \
            --dataset MESC