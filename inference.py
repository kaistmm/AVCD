import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import argparse
import torch
def inference(args):

    model_path = "/mnt/bear3/users/cyong/VideoLLaMA2.1-7B-AV"
    model, processor, tokenizer = model_init(model_path)

    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError
    

    audio_video_path = "/mnt/bear3/users/cyong/AVHBench/data/AVHBench_v0/video/00159.mp4"
    question =  "Did it snow outside? It looks warm outside."
  
    preprocess = processor['audio' if args.modal_type == "a" else "video"]
    audio_video_tensor = preprocess(audio_video_path, va=True) 
  
    output = mm_infer(
        audio_video_tensor,
        question,
        model=model,
        tokenizer=tokenizer,
        modal='audio' if args.modal_type == "a" else "video",
        do_sample=False,
    )

    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',default="av",  help='')
    parser.add_argument('--modal-type',default="av", choices=["a", "v", "av"], help='')
    args = parser.parse_args()

    inference(args)

