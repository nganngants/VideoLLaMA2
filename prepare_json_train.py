def extract_k_vid_user_prev(jsonl_file, output_file):
  import json
  import os
  import shutil
  import re

  k_vid_user_prev = []
  with open(jsonl_file, 'r') as f:
    for line in f:
      data = json.loads(line)
      k_vid_user_prev += data['k_vid_user_prev']
  k_vid_user_prev = list(set(k_vid_user_prev))
  k_vid_user_prev = [{'video': vid} for vid in k_vid_user_prev]
  with open(output_file, 'w') as f:
    json.dump(k_vid_user_prev, f)

extract_k_vid_user_prev('train.jsonl', 'mesc_train.json')