import os
import json
import math
import argparse
import warnings
import traceback
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import re
import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import torch

from threadpoolctl import threadpool_limits


# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class AVH_val(Dataset):

    def __init__(self, questions, processor, processor2):
        self.questions = questions
        self.processor = processor
        self.processor2 = processor2


    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        sample = self.questions[idx]


        text = sample['text']+" Answer yes or no." 

        video_path = "/mnt/lynx1/datasets/AVHBench/data/AVHBench_v0/video/" +sample["video_id"]+".mp4"
        question = text 
        answer = sample["label"]
        print("question", question)
        audio_video_dict = self.processor(video_path, va=True)


        return {
            'audio_video':  audio_video_dict,
            'video_name':  video_path.split("/")[-1],
            'question':    question,
            'question_id': video_path.split("/")[-1], #question_id,
            'answer':      answer,
        }

def collate_fn(batch):
    aud_vid  = [x['audio_video'] for x in batch]
    v_id = [x['video_name'] for x in batch]
    qus  = [x['question'] for x in batch]
    qid  = [x['question_id'] for x in batch]
    ans  = [x['answer'] for x in batch]
  
    return aud_vid, v_id, qus, qid, ans


def run_inference(args):
    disable_torch_init()

    # Initialize the model
    model, processor, tokenizer = model_init(args.model_path)

    gt_questions = json.load(open(args.question_file, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    assert args.batch_size == 1, "Batch size must be 1 for inference"
  
    dataset = AVH_val(gt_questions, processor['video'], processor["audio"])
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    answer_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    # Iterate over each sample in the ground truth file
    for i, (aud_vid_tensors, video_names, questions, question_ids, answers) in enumerate(tqdm(dataloader)):
        audio_video_tensor = aud_vid_tensors[0]
        question     = questions[0]
        question_id  = video_names[0] #question_ids[0]
        answer       = answers[0]
      

        # try:
        output = mm_infer(
                audio_video_tensor,
                question,
                model=model,
                tokenizer=tokenizer,
                modal='video',
                do_sample=False,  
                use_AVCD=args.use_AVCD
            )

        sample_set = {'id': question_id, 'question': question, 'answer': answer, 'pred': output}
        ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='',default="/mnt/bear3/users/cyong/VideoLLaMA2.1-7B-AV")
    parser.add_argument('--video-folder', help='Directory containing video files.', default="/mnt/bear3/users/cyong/AVHBench/data/AVHBench_v0/video")
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default="./json/AVH_val.json")
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', default="./AVH_validation.json") #last2
    parser.add_argument("--use-AVCD", type=str, default=False)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=4)
    parser.add_argument("--dataset", type=str, default="AVH_val")
    args = parser.parse_args()

    with threadpool_limits(limits=4):
        run_inference(args)
