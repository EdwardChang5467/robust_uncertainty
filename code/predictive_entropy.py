import logging
import time
from tqdm import tqdm
import sys
import numpy as np
import re
import json
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--generation_path", required=True, type=str, help="generation file path")
parser.add_argument("--output_path", required=True, type=str, help="result file path")
args = parser.parse_args()

def get_predictive_entropy(generations,logits):
    generations_predictive_entropy = []
    for i,generation in enumerate(generations):
        generation_logits = logits[i]
        generation_predictive_entropy = 0.0
        for j in range(len(generation_logits)):
            token = list(generation_logits[j].keys())[0]
            if token in generation:
                token_prob = generation_logits[j][token]
                generation_predictive_entropy = generation_predictive_entropy - token_prob * np.log(token_prob)
        generations_predictive_entropy.append(generation_predictive_entropy)
    return generations_predictive_entropy
        
                

def get_length_normed_predictive_entropy(generations,logits):
    generations_ln_predictive_entropy = []
    for i,generation in enumerate(generations):
        generation_logits = logits[i]
        generation_predictive_entropy = 0.0
        for j in range(len(generation_logits)):
            token = list(generation_logits[j].keys())[0]
            if token in generation:
                token_prob = generation_logits[j][token]
                generation_predictive_entropy = generation_predictive_entropy - token_prob * np.log(token_prob)
        generation_ln_predictive_entropy = generation_predictive_entropy / len(generation_logits)
        generations_ln_predictive_entropy.append(generation_ln_predictive_entropy)
    return generations_ln_predictive_entropy

generation_path = args.generation_path
output_path = args.output_path

with open(generation_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    file.close()

new_data = []
for idx,dt in enumerate(data):
    generations = dt["generated_answer"]
    logits = dt["logits"]
    question = dt["question"]
    predictive_entropy = get_predictive_entropy(generations,logits)
    length_normed_predictive_entropy = get_length_normed_predictive_entropy(generations,logits)
    
    new_dict = dict()
    new_dict["question"] = question
    new_dict["generated_answer"] = generations
    new_dict["predictive_entropy"] = predictive_entropy
    new_dict["ln_predictive_entropy"] = length_normed_predictive_entropy
    new_data.append(new_dict)

with open(output_path,"w",encoding="utf-8") as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)

