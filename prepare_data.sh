#!/bin/bash

pip install gdown
sudo apt install unzip

gdown --folder https://drive.google.com/drive/folders/1d-vCvHlGHQNZNzfXqwTwVOQhJ6yzJUh9

cd MESC

cat MESC_Video.zip.* > MESC_Video.zip
unzip MESC_Video.zip
rm -rf MESC_Video.zip.*

unzip mesc_test.zip

cd ..

python3 create_mesc_json.py --video-folder MESC/MESC_Video --output-path MESC/mesc.json
