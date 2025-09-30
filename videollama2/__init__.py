import os
import copy
import warnings
import shutil
from functools import partial

import torch
from torch import nn
from .model import load_pretrained_model
from .mm_utils import process_image, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria, process_audio_file
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, DEFAULT_AUDIO_TOKEN

import torch.nn.functional as F
import numpy as np


def model_init(model_path=None, **kwargs):
    model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    processor = {
        'image': partial(process_image, processor=processor, aspect_ratio=None),
        'video': partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames),
        'audio': process_audio_file,
    }

    return model, processor, tokenizer


def mm_infer(image_or_video, instruct, model, tokenizer, modal='video', generate_long=False,use_AVCD=False, **kwargs):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """

    # 1. text preprocess (tag process & generate prompt).
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'text':
        modal_token = ''
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    else:
        raise ValueError(f"Unsupported modal: {modal}")



    # 1. vision preprocess (load & transform image or video).
    if modal == 'text':
        tensor = None
    else:
        if isinstance(image_or_video, dict):
            tensor = {k: v.half().cuda() for k, v in image_or_video.items()}
        else:
            tensor = image_or_video.half().cuda()  
       
        tensor = [(tensor, modal)]


    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct[0]}]
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    if not generate_long:
        #This is for transformers/generation/utils.py
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_masks,
                images=tensor,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
                use_AVCD=use_AVCD,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print('outputs', outputs)
        return outputs
    
    else:
        # This is for simple forward to generate answers (Captioning will use the below code)
        with torch.no_grad():
            generated_tokens = input_ids.clone()
            ques_leng = input_ids.shape[1]

            count=0

            for series in range(max_new_tokens):  # Maximum response length
                count+=1
            
                attention_masks = generated_tokens.ne(tokenizer.pad_token_id).long().cuda()
                generated_tokens_tensor = generated_tokens

                if not use_AVCD:
                    orig, _, _ = model(
                            input_ids=generated_tokens_tensor,
                            attention_mask=attention_masks,
                            images=tensor,
                            return_dict=True,
                        )
                    next_token_logits = orig.logits[:, -1, :]
                    del orig

                else:
                   
                    orig, avg_dominance, threshold = model(
                                input_ids=generated_tokens_tensor,
                                attention_mask=attention_masks,
                                images=tensor,
                                return_dict=True,
                            )
                    orig_logits = orig.logits[:, -1, :]
                
                    beta = 0.2
                    cutoff = torch.log(torch.tensor(beta)) + orig_logits.max(dim=-1, keepdim=True).values
                    
                    probs_ent = F.softmax(orig_logits, dim=-1)  # Shape: [batch_size, num_candidates]
                    log_probs_ent = F.log_softmax(orig_logits, dim=-1)  # Shape: [batch_size, num_candidates]
                    entropy = -(probs_ent * log_probs_ent).sum(dim=-1)  # Shape: [batch_size]
                    
                    if entropy.item() <0.6:
                        next_token_logits = orig_logits
                        print("pass")
                        del orig, threshold, orig_logits
                        torch.cuda.empty_cache()
                    else:
                       
                        if avg_dominance[0][0] =="language":
                            modality1 = "VA"
                            modality2 = "A"
                            modality3 = "V"
                        elif avg_dominance[0][0] =="video":
                            modality1 = "LA"
                            modality2 = "A"
                            modality3="L"
                        else:
                            modality1 = "LV"
                            modality2 = "V"
                            modality3 = "L"
                

                        outputs1, _, _ = model(
                            input_ids=generated_tokens_tensor,
                            attention_mask=attention_masks,
                            images=tensor,
                            return_dict=True,
                            modality = modality1,
                            threshold = threshold
                        
                        )
                        outputs1_logit = outputs1.logits[:, -1, :]
                        
                        outputs2, _, _ = model(
                            input_ids=generated_tokens_tensor,
                            attention_mask=attention_masks,
                            images=tensor,
                            return_dict=True,
                            modality = modality2,
                            threshold = threshold
                        )
                        outputs2_logit = outputs2.logits[:, -1, :]

                        outputs3, _, _ = model(
                            input_ids=generated_tokens_tensor,
                            attention_mask=attention_masks,
                            images=tensor,
                            return_dict=True,
                            modality = modality3,
                            threshold = threshold
                        )
                        outputs3_logit = outputs3.logits[:, -1, :]

                        alpha = 0.5

                        #This is AVCD formula
                        contrastive_logits = (2+2*alpha)*orig_logits-2*alpha*outputs1_logit+outputs2_logit+outputs3_logit
                        #This is single-instance CD
                        # contrastive_logits = (1+alpha)*orig_logits-alpha*outputs1_logit
                        
                        #Apply adaptive plausibility constraint
                        next_token_logits = contrastive_logits.masked_fill(orig_logits < cutoff, -float(1e-4))

                        #To reduce memory usage, you can delete below.
                        del orig, outputs1, outputs2, outputs3, outputs1_logit, outputs2_logit, outputs3_logit, threshold
                        torch.cuda.empty_cache()

            
                # Select the next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1).item()

                if next_token == tokenizer.eos_token_id:
                    break
            
                next_token_tensor = torch.tensor([[next_token]], device=generated_tokens.device, dtype=torch.long)
                generated_tokens = torch.cat([generated_tokens, next_token_tensor], dim=1)  # Concatenate along sequence dimension
                
        #delete question
        generated_tokens = generated_tokens[:,ques_leng:]
        
        outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        print("answer", outputs)
    
        return outputs
