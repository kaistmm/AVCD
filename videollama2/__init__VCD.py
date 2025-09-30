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

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd


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


def mm_infer(image_or_video, instruct, model, tokenizer, modal='video', number=None,layer_number = None, **kwargs):
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
        
        tensor_vcd = {}
        tensor_vcd_aud={}
        tensor_both = {}

        tensor_vcd['audio'] =tensor['audio']
        tensor_vcd_aud['video'] = tensor['video']
        video_ten = tensor['video']
        audio_ten = tensor['audio']
        tensor_vcd['video'] = add_diffusion_noise(video_ten, 500)
        tensor_vcd_aud['audio'] = add_diffusion_noise(audio_ten, 500)
        tensor_both['video'] = add_diffusion_noise(video_ten, 500)
        tensor_both['audio'] = add_diffusion_noise(audio_ten, 500)

       
        tensor = [(tensor, modal)]    
        tensor_vcd = [(tensor_vcd, modal)]
        tensor_vcd_aud =[(tensor_vcd_aud, modal)]
        tensor_both = [(tensor_both, modal)]

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

    # This is for transformers/generation/utils.py
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=[tensor, tensor_both, tensor_vcd_aud, tensor_vcd],
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            use_cd=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print('outputs', outputs)

    return outputs

 
 
    # with torch.no_grad():
    #     generated_tokens = input_ids.clone()
    #     ques_leng = input_ids.shape[1]

    #     count=0

    #     for series in range(max_new_tokens):  # Maximum response length
    #         count+=1
           
    #         attention_masks = generated_tokens.ne(tokenizer.pad_token_id).long().cuda()
    #         generated_tokens_tensor = generated_tokens

    #         orig, _, threshold = model(
    #                     input_ids=generated_tokens_tensor,
    #                     attention_mask=attention_masks,
    #                     images=tensor,
    #                     return_dict=True,
    #                 )
            
    #         orig_logits = orig.logits[:, -1, :]
           
    #         beta = 0.2
    #         cutoff = torch.log(torch.tensor(beta)) + orig_logits.max(dim=-1, keepdim=True).values
            
    #         probs_ent = F.softmax(orig_logits, dim=-1)  # Shape: [batch_size, num_candidates]
    #         log_probs_ent = F.log_softmax(orig_logits, dim=-1)  # Shape: [batch_size, num_candidates]
    #         entropy = -(probs_ent * log_probs_ent).sum(dim=-1)  # Shape: [batch_size]
            
    #         if entropy.item() <0.6:
    #             contrastive_logits = orig_logits
    #             print("pass")
    #             del orig, threshold, orig_logits
    #             torch.cuda.empty_cache()
    #         else:
    #             #Assume language is the most dominant.

    #             outputs1, _, _ = model(
    #                 input_ids=generated_tokens_tensor,
    #                 attention_mask=attention_masks,
    #                 images=tensor_both,
    #                 return_dict=True,                
    #             )
    #             outputs1_logit = outputs1.logits[:, -1, :]
                   
    #             outputs2, _, _ = model(
    #                 input_ids=generated_tokens_tensor,
    #                 attention_mask=attention_masks,
    #                 images=tensor_vcd,
    #                 return_dict=True,
    #             )
    #             outputs2_logit = outputs2.logits[:, -1, :]

    #             outputs3, _, _ = model(
    #                 input_ids=generated_tokens_tensor,
    #                 attention_mask=attention_masks,
    #                 images=tensor_vcd_aud,
    #                 return_dict=True,
    #             )
    #             outputs3_logit = outputs3.logits[:, -1, :]

               

    #             alpha = 1
    #             contrastive_logits =(2+2*alpha) *orig_logits - 2*alpha*outputs1_logit +outputs2_logit+outputs3_logit
    #             contrastive_logits = contrastive_logits.masked_fill(orig_logits < cutoff, -float(1e-4))

               
    #             del orig, outputs1, outputs2, outputs3, outputs1_logit, outputs2_logit, outputs3_logit, threshold
    #             torch.cuda.empty_cache()

         
    #         # Select the next token
    #         probs = F.softmax(contrastive_logits, dim=-1)
    #         next_token = torch.argmax(probs, dim=-1).item()

    #         if next_token == tokenizer.eos_token_id:
    #             break
            
    #         next_token_tensor = torch.tensor([[next_token]], device=generated_tokens.device, dtype=torch.long)
    #         generated_tokens = torch.cat([generated_tokens, next_token_tensor], dim=1)  # Concatenate along sequence dimension
            
    # #delete question
    # generated_tokens = generated_tokens[:,ques_leng:]
    
    # outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    # print("answer", outputs)
  
    # return outputs
