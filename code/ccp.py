import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel,LlamaTokenizer,LlamaForCausalLM
import torch
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import MaximumClaimProbability, ClaimConditionedProbabilityClaim
from lm_polygraph.estimators.claim.claim_conditioned_probability_new import ClaimConditionedProbabilityClaimNew
from lm_polygraph.stat_calculators import *
from lm_polygraph.stat_calculators import extract_claims_new
# from lm_polygraph.utils.openai_chat import OpenAIChat
from lm_polygraph.utils.openai_chat_new import OpenAIChat
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.generation_metrics import *
from lm_polygraph.generation_metrics.openai_fact_check_new import OpenAIFactCheckNew
import numpy as np
import json
from lm_polygraph.ue_metrics import roc_auc

with open("/data/zhangyuhao/robust_uncertainty/data/llama3-8b-instruct_new/llama3-8b-instruct_new_sample5_temp1.0_result_v1_fact_v1_with_label_map_v1.json", 'r', encoding='utf-8') as file1:
    data = json.load(file1)
    file1.close()

questions = []
for idx,dt in enumerate(data):
    question = dt["question"]
    questions.append(question)
    
# model_path = "/data/zhangyuhao/model/mistral-7b"
model_path = "/data/zhangyuhao/model/chatglm3_6b_32k/models--THUDM--chatglm3-6b-32k"
if "chatglm3" in model_path:
    base_model = AutoModel.from_pretrained(model_path,trust_remote_code=True)
elif "mistral" in model_path:
    base_model = LlamaForCausalLM.from_pretrained(model_path,trust_remote_code=True)
else:
    base_model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
if "mistral" in model_path:
    tokenizer = LlamaTokenizer.from_pretrained(model_path,trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
if "chatglm3" not in model_path:
    tokenizer.pad_token = tokenizer.eos_token
model = WhiteboxModel(base_model, tokenizer, model_path=model_path)

# texts = ["Tell me a bio of Albert Einstein."]
texts = questions
if type(texts) != list:
    texts = [texts]
stat = {}
# print(texts)

os.environ["OPENAI_KEY"] = "YOUR_API_KEY"
os.environ["OPENAI_BASE"] = "https://api.lingyiwanwu.com/v1"

for calculator in [
    GreedyProbsCalculator(),
    extract_claims_new.ClaimsExtractorNew(OpenAIChat("yi-lightning")),
]:
    stat.update(calculator(stat, texts, model))
   
# print("Output:", stat["greedy_texts"][0])
# print()
# for claim in stat["claims"][0]:
#     print("claim:", claim.claim_text)
#     print("aligned tokens:", claim.aligned_token_ids)
#     print()
max_prob = MaximumClaimProbability()
# print(max_prob(stat))  # Uncertainty for each claim, the higher, the less certain
for calculator in [
    GreedyAlternativesNLICalculator(Deberta())
]:
    stat.update(calculator(stat, texts, model))

ccp = ClaimConditionedProbabilityClaimNew()

uncertainty_list = []
for i in range(len(texts)):
    ccp_uncertainty_dict = dict()
    ccp_uncertainty_dict["question"] = texts[i]
    ccp_uncertainty_dict["generation"] = stat["greedy_texts"][i]
    ccp_uncertainty_dict["ccp_uncertainty"] = ccp(stat)[i]
    ccp_uncertainty_dict["max_prob_uncertainty"] = [float(un) for un in max_prob(stat)[i]]
    uncertainty_list.append(ccp_uncertainty_dict)

output_path = "/data/zhangyuhao/poly_graph/lm_polygraph/result/ccp_in_all_name_chatglm3-6b-32k_new_result.json"
with open(output_path,"w",encoding="utf-8") as file:
    json.dump(uncertainty_list, file, ensure_ascii=False, indent=4)

