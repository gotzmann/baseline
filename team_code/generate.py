import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os.path
import os

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from .utils import get_query_from_input, get_text_emb

# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
# from llava.eval.run_llava import eval_model

# model_path = "liuhaotian/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

# -- PandaGPT

### from transformers import AutoModel, AutoTokenizer
### from copy import deepcopy
### import os
### import ipdb
import gradio as gr
### import mdtex2html
from model.openllama import OpenLLAMAPEFTModel
### import torch
### import json

# -- Baseline

DEVICE = torch.device("cuda:0")
DIALOGUE_DICT = {}

# bad_words_ids = tokenizer(["\nUser: ", "\n Bot:",], add_special_tokens=False).input_ids
bad_words_ids = [
    [29871, 13, 2659, 29901, 29871],
    [29871, 13, 11273, 29901],
]

gen_params = {
    "do_sample": False,
    "max_new_tokens": 80,
    "early_stopping": True,
    "num_beams": 1,
    "remove_invalid_values": True,
    "eos_token_id": 29889,
    "pad_token_id": 29889,
    "forced_eos_token_id": 29889,
    "use_cache": True,
    "bad_words_ids": bad_words_ids,
    "num_return_sequences": 1,
}


@torch.no_grad()
def gen_answer(model, tokenizer, query, history=None):
    query = torch.cat([history, query], dim=1)

    out = model.generate(
        inputs_embeds=query,
        **gen_params,
    )
    out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)
    return generated_texts[0]


def imagebind_huge(pretrained=False):
    model = imagebind_model.ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )

    if pretrained:
        model.load_state_dict(torch.load("/app/.checkpoints/imagebind_huge.pth"))

    return model


# Function that returns model and tokenizer that will be used during the generation
def setup_model_and_tokenizer():

    # -- init the model
    # https://huggingface.co/openllmplayground/pandagpt_13b_max_len_400
    # https://huggingface.co/openllmplayground/pandagpt_13b_max_len_400/resolve/main/pytorch_model.pt

    panda_args = {
        'model':               'openllama_peft',
        'imagebind_path': '/app/.checkpoints', # /imagebind_huge.pth # '../pretrained_ckpt/imagebind_ckpt',
        'vicuna_path':    '/app/vicuna-13b-v1.3', # '../pretrained_ckpt/vicuna_ckpt/13b_v0',
        'delta_path':     '/app/panda/pytorch_model.pt', # '../pretrained_ckpt/pandagpt_ckpt/13b/pytorch_model.pt',
        'stage': 2,
        'max_tgt_len': 128,
        'lora_r': 32,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
    }

    panda_model = OpenLLAMAPEFTModel()
    delta_ckpt = torch.load(panda_args['delta_path'], map_location=torch.device('cpu'))
    panda_model.load_state_dict(delta_ckpt, strict=False)
    panda_model = panda_model.eval().half().cuda()
    print("Init the Panda 13B model...")


    tokenizer = AutoTokenizer.from_pretrained("/app/Llama-2-7B-fp16", padding_side="left", use_fast=False)
    # tokenizer = AutoTokenizer.from_pretrained("/app/llava-v1.5-13b", padding_side="left", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("/app/Llama-2-7B-fp16", torch_dtype=torch.float16).eval().to(device=DEVICE)
    # model = AutoModelForCausalLM.from_pretrained("/app/llava-v1.5-13b", torch_dtype=torch.float16).eval().to(device=DEVICE)

    # Instantiate model for image and audio embeddings
    model_imagebind = imagebind_huge(pretrained=True).eval().to(device=DEVICE)
    model_imagebind.query_dict = {}
        
    EMB_DIM = 4096
    N_MODALITY_EMBS = 32
    ENC_DIM = model_imagebind.modality_heads[ModalityType.VISION][-1].out_features

    projection = nn.Linear(ENC_DIM, N_MODALITY_EMBS * EMB_DIM).to(device=model.device, dtype=model.dtype).eval()
    workdir = os.getcwd()

    img_tokens_emb = torch.load(
        f"{workdir}/team_code/ckpts/IMG_EMB_LLaMa-7b-EN-Linear-ImageBind",
        map_location=model.device,
    )
    audio_tokens_emb = torch.load(
        f"{workdir}/team_code/ckpts/AUDIO_EMB_LLaMa-7b-EN-Linear-ImageBind",
        map_location=model.device,
    )
    projection = torch.load(
        f"{workdir}/team_code/ckpts/projection_LLaMa-7b-EN-Linear-ImageBind",
        map_location=model.device,
    )

    return [
        model,
        model_imagebind,
        img_tokens_emb,
        audio_tokens_emb,
        projection,
        panda_model
    ], tokenizer


# Function that generates the responses for dialodues queries w.r.t. history.
def generate_text(model, tokenizer, cur_query_list, history_tensor=None):

    panda_model = model[5]

    prompt = cur_query_list[0]["content"]
    image_path = ""
    audio_path = ""
    video_path = ""
    thermal_path = ""
    top_p = 0.96
    temperature = 1.0
    max_length = 400 # 256
    modality_cache = None

    response = panda_model.generate({
        'prompt': prompt,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        'thermal_paths': [thermal_path] if thermal_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })

    return ([], response)

    # -- baseline



    if history_tensor is not None:
        history_tensor = torch.concat(
            [history_tensor[0], get_text_emb(model[0], tokenizer, history_tensor[1])],
            dim=1,
        )
    else:
        # If the current history is empty
        # it is assigned to the system prompt
        PROMPT = "This is a dialog with AI assistant.\n"
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        prompt_embeddings = model[0].model.embed_tokens(prompt_ids)
        history_tensor = prompt_embeddings

    prompt = get_query_from_input(model, tokenizer, cur_query_list).to(DEVICE)
    response = gen_answer(model[0], tokenizer, prompt, history=history_tensor)

    history_tensor = torch.concat([history_tensor, prompt], dim=1)

    return response, history_tensor

def get_ppl(model, tokenizer, cur_query_tuple, history_tensor=None):
    if history_tensor is not None:
        history_tensor = torch.concat([history_tensor[0], get_text_emb(model[0], tokenizer, history_tensor[1])], dim=1)
    else:
        # If the current history is empty
        # it is assigned to the system prompt
        PROMPT = "This is a dialog with AI assistant.\n"
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        prompt_embeddings = model[0].model.embed_tokens(prompt_ids)
        history_tensor = prompt_embeddings

    current_query = get_query_from_input(model, tokenizer, cur_query_tuple[0])
    current_answer = get_text_emb(model[0], tokenizer, cur_query_tuple[1])

    # Input dialogue query with history
    dialogue_emb = torch.concat([history_tensor, current_query], dim=1).to(DEVICE)
    inputs_embeds=torch.concat([dialogue_emb, current_answer], dim=1)
    
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        out_logits = model[0](inputs_embeds=inputs_embeds).logits

    shift_logits = out_logits[..., : -1, :].contiguous()
    labels = tokenizer.encode(cur_query_tuple[1], add_special_tokens=False, return_tensors="pt")
    context_before_labels = torch.LongTensor([-100] * dialogue_emb.shape[1]).unsqueeze(0)
    labels = torch.concat([context_before_labels, labels], dim=1).to(DEVICE)
    shift_labels = labels[..., 1:].contiguous()
    
    neg_log_likelihood = loss(shift_logits.transpose(1, 2), shift_labels)
    ppl = torch.exp2(neg_log_likelihood)
    
    return ppl.item(), dialogue_emb
