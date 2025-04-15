import json
import argparse
import os
import logging
from tqdm import tqdm
import numpy as np
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='model path',required=True)
parser.add_argument("--file_path", required=True, type=str, help="file path")
parser.add_argument("--model_name", required=True, type=str, help="generation model name")
parser.add_argument("--version", default=1, type=int, help="version")
parser.add_argument("--seed", default=42, type=int, help="random seed")
parser.add_argument("--cuda", default="3", type=str, help="cuda")
parser.add_argument('--num_beams', type=int, default=5)
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument("--number_of_generations", default=5, type=int, help="sample number of generations")
parser.add_argument("--temperature", default=1.0, type=float, help="temperature when generate")
parser.add_argument("--output_path", required=True, type=str, help="generated sequences output path")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
seed_value = args.seed
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
np.random.seed(seed_value)
import torch
torch.manual_seed(seed_value)
from transformers import AutoTokenizer,AutoModelForCausalLM,LlamaTokenizer,LlamaForCausalLM

model_path = args.model_path
model_name = args.model_name

if "mistral" in model_name:
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(model_path,trust_remote_code=True).cuda()
    model.eval()
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path,trust_remote_code=True,torch_dtype=torch.bfloat16).cuda()
    model.eval()    
device = "cuda"

with open(args.file_path,'r',encoding="utf-8") as file:
    data = json.load(file)
    file.close()
pbar = tqdm(total=len(data))

index = 0
current_progressing = 0
result_list = []
with open(args.file_path,'r',encoding="utf-8") as file:
    for question in data:
        current_progressing = current_progressing + 1
        input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request. Don't correct the name in the Instruction.\r\n\r\n"
            f"### Instruction:\r\n{question} Don't correct the name in the question.\r\n\r\n### Response:\r\n"
        )    
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_len = len(inputs.input_ids[0])
        output_text_list = []
        logit_list = []
        loss_list = []
            
        for i in range(args.number_of_generations):
            logits = []
            # do sampling of generations
            output = model.generate(
                inputs=inputs.input_ids, 
                max_new_tokens=100,
                do_sample=True,
                output_attentions=True,
                output_scores=True,
                num_return_sequences=1,
                temperature=args.temperature,
                num_beams=args.num_beams,
                top_p=args.top_p
            )
            output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            output_text = output_text[len(input_text):]
            output_text_list.append(output_text)
                
            # get logit of output tokens
            output_ids = output.clone()
            logit = model(output_ids,output_hidden_states=True).logits[0].clone().detach().cpu()
            logit = torch.softmax(logit[input_len-1:][:],dim=-1)
            logit = [logit[i][j]  for i,j in enumerate(output[0][input_len:])]
        
            for i in range(len(output[0][input_len:])):
                logits.append({str(tokenizer.decode(output[0][input_len+i])):round(float(logit[i]),4)})
            logit_list.append(logits)
            logit_list_str = json.dumps(logit_list)
                
            # get loss of generated answer
            input_ = torch.tensor([tokenizer.encode(input_text)+tokenizer.encode(output_text)]).to(model.device)
            labels =torch.tensor(len(tokenizer.encode(input_text))*[-100]+tokenizer.encode(output_text)).to(model.device)
            output_for_loss = model(input_ids=input_,labels=labels)
            loss = round(float(output_for_loss.loss.detach().cpu()),4)
            loss_list.append(loss)
                
        result_dict = dict()
        result_dict["question"] = question
        result_dict["generation_model"] = args.model_name
        result_dict["generated_answer"] = output_text_list
        result_dict["logits"] = logit_list
        result_dict["loss"] = loss_list
        if current_progressing == 1:
            logger.info(result_dict["generated_answer"])
        result_list.append(result_dict)
        pbar.update(1)

with open(f"{args.output_path}/{args.model_name}_sample{args.number_of_generations}_temp{args.temperature}_result_v{args.version}.json","w",encoding='utf-8') as file:
    json.dump(result_list, file, ensure_ascii=False, indent=4)
    file.close()

    
        


