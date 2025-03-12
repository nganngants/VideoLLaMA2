import os
import json
import math
import argparse
import warnings
import traceback
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import gc
import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer, get_video_audio_embeddings
from videollama2.utils import disable_torch_init
import h5py

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class MESCDataset(Dataset):
    def __init__(self, questions, processor, folder_path):
        self.questions = questions
        self.processor = processor
        self.folder_path = folder_path
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]
        
        video_path = sample['video']

        if not video_path.startswith(self.folder_path):
            video_path = os.path.join(self.folder_path, video_path)
        
        try:
            audio_video_dict = self.processor(video_path, va=True)
        except:
            print("video read error")
            audio_video_dict = None
        
        return {
            'audio_video': audio_video_dict,
            'video_name': video_path.split("/")[-1].split(".")[0],
        }

def collate_fn(batch):
    aud_vid  = [x['audio_video'] for x in batch]
    v_id = [x['video_name'] for x in batch]
    return aud_vid, v_id


def run_inference(args):
    disable_torch_init()

    # Initialize the model
    model, processor, tokenizer = model_init(args.model_path)

    gt_questions = json.load(open(args.question_file, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    assert args.batch_size == 1, "Batch size must be 1 for inference"
    if args.dataset == "MESC":
        dataset = MESCDataset(gt_questions, processor['video'], args.video_folder)
    else:
        raise NotImplementedError
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate over each sample in the ground truth file
    for i, (aud_vid_tensors, video_names) in enumerate(tqdm(dataloader)):
        audio_video_tensor = aud_vid_tensors[0]
        video_name = video_names[0]

        try:
            embeddings = get_video_audio_embeddings(audio_video_tensor, model, tokenizer)

            audio_np = embeddings["audio"].detach().cpu().numpy().astype("float16")
            video_np = embeddings["video"].detach().cpu().numpy().astype("float16")

            output_path = os.path.join(args.output_dir, f"{video_name}.h5")
        
            with h5py.File(output_path, 'w') as f:
                # Use gzip compression with maximum compression level (9)
                f.create_dataset('audio', data=audio_np, compression='gzip', compression_opts=1)
                f.create_dataset('video', data=video_np, compression='gzip', compression_opts=1)
            
            # Save embeddings
            # output_path = os.path.join(args.output_dir, f"{video_name}.pt")
            # torch.save(embeddings, output_path)

            # clean up
            del audio_video_tensor
            del embeddings
            del audio_np
            del video_np

            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing video: {video_name}, Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=False)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=False)
    parser.add_argument('--output-dir', help='Directory to save the video and audio embeddings.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    run_inference(args)

zip -r -v - embeddings_train | split -b 4G - embeddings/embeddings_train.zip.
