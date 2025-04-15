import json
import logging
import time
from tqdm import tqdm
import argparse
import logging
import numpy as np
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", required=True, type=str, help="fact file path")
parser.add_argument("--check_file_path", required=True, type=str, help="check file path")
parser.add_argument("--output_path", required=True, type=str, help="result file path")
args = parser.parse_args()

file_path = args.file_path
check_file_path = args.check_file_path
output_path = args.output_path
log = logging.getLogger()

def get_factual_right_identified_gen_uncertainty(gen, gen_probs):
    probs_list = []
    for logit_dict in gen_probs:
        token = list(logit_dict.keys())[0]
        if token in gen:
            prob = logit_dict[token]
            probs_list.append(prob)
            
    generation_log_prob = 0.0
    for tok_prob in probs_list:
        generation_log_prob += math.log(tok_prob)
    
    # get length-normed log probability 
    generation_normed_log_prob = generation_log_prob / len(probs_list)
        
    generation_prob = math.exp(generation_normed_log_prob)
    return 1 - generation_prob
    
    
def get_factual_correct_true_and_false_gen_uncertainty(gen, gen_probs, aligned_generation_token_ids, label):
    prob_list = []
    for aligned_sent_token_ids in aligned_generation_token_ids:
        for aligned_fact_token_ids in aligned_sent_token_ids:
            for i in range(len(gen_probs)):
                token = list(gen_probs[i].keys())[0]
                prob = gen_probs[i][token]
                if i in aligned_fact_token_ids:
                    prob_list.append(prob)          

    generation_log_prob = 0.0
    for tok_prob in prob_list:
        generation_log_prob += math.log(tok_prob)
    # get length-normed log probability 
    generation_normed_log_prob = generation_log_prob / len(prob_list)
    generation_prob = math.exp(generation_normed_log_prob)
    if label == "FT":  
        return 1- generation_prob                
    else:
        return generation_prob

def get_factual_correct_true_and_false_fact_uncertainty(gen, gen_probs, aligned_generation_token_ids, label):
    label_list = []
    fact_uncertainty = []
    for aligned_sent_token_ids in aligned_generation_token_ids:
        for aligned_fact_token_ids in aligned_sent_token_ids:
            fact_prob_list = []
            for i in range(len(gen_probs)):
                token = list(gen_probs[i].keys())[0]
                prob = gen_probs[i][token]
                if i in aligned_fact_token_ids:
                    fact_prob_list.append(prob) 
            fact_log_prob = 0.0
            if len(aligned_fact_token_ids) > 0:
                for tok_prob in fact_prob_list:
                    fact_log_prob += math.log(tok_prob)   
                fact_normed_log_prob = fact_log_prob / len(fact_prob_list) 
                fact_normed_prob = math.exp(fact_normed_log_prob)
                if label == "FT":
                    fact_uncertainty.append(1-fact_normed_prob)
                else:
                    fact_uncertainty.append(fact_normed_prob)   
    return fact_uncertainty

def compute_uncertainty():
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        file.close()
    with open(check_file_path, 'r', encoding='utf-8') as file:
        check_data = json.load(file)
        file.close()
    
    err_cnt = 0
    err_flag = 0
    uncertainty_list = []
    pbar = tqdm(total=len(data))
    for idx,dt in enumerate(data):
        err_flag = 0
        question = dt["question"]
        labels = dt["generation_labels"]
        generations = dt["generated_answer"]
        probs = dt["logits"]
        check_labels = check_data[idx]["check_labels"]
        loss = dt["loss"]
        aligned_generations_token_ids = dt["aligned_token_ids"]
        uncertainty = []
        fact_uncertainty_list = []
        fact_labels = []
        fact_check_labels = []
        if len(check_labels) != len(generations):
            err_cnt += 1
            uncertainty_dict = dict()
            uncertainty_dict["question"] = question
            uncertainty_dict["uncertainty"] = []
            uncertainty_dict["mean_uncertainty_in_gen"] = 0
            uncertainty_dict["fact_uncertainty"] = []
            uncertainty_dict["fact_labels"] = []
            uncertainty_dict["fact_check_labels"] = []
            uncertainty_list.append(uncertainty_dict)
            pbar.update(1)
            continue
        for i,generation in enumerate(generations):
            label = labels[i]
            aligned_generation_token_ids = aligned_generations_token_ids[i]
            generation_tokens_probs = probs[i]
            gen_check_label = check_labels[i]
            if label == "FR":
                gen_uncertainty = get_factual_right_identified_gen_uncertainty(generation, generation_tokens_probs)
                fact_uncertainty = [gen_uncertainty]
            else:
                try:
                    gen_uncertainty = get_factual_correct_true_and_false_gen_uncertainty(generation, generation_tokens_probs, aligned_generation_token_ids, label)
                    fact_uncertainty = get_factual_correct_true_and_false_fact_uncertainty(generation, generation_tokens_probs, aligned_generation_token_ids, label)
                except:
                    err_flag = 1
                    break
            uncertainty.append(gen_uncertainty)
            for un in fact_uncertainty:
                fact_uncertainty_list.append(un)
            for j in range(len(fact_uncertainty)):
                fact_labels.append(label)
            for k in range(len(fact_uncertainty)):
                fact_check_labels.append(gen_check_label)
            
        
        if err_flag == 1:
            uncertainty_dict = dict()
            uncertainty_dict["question"] = question
            uncertainty_dict["uncertainty"] = []
            uncertainty_dict["mean_uncertainty_in_gen"] = 0
            uncertainty_dict["fact_uncertainty"] = []
            uncertainty_dict["fact_labels"] = []
            uncertainty_dict["fact_check_labels"] = []
            uncertainty_list.append(uncertainty_dict)
            pbar.update(1)
            continue   
        mean_uncertainty = np.mean(uncertainty)
        
        uncertainty_dict = dict()
        uncertainty_dict["question"] = question
        uncertainty_dict["uncertainty"] = uncertainty
        uncertainty_dict["mean_uncertainty_in_gen"] = mean_uncertainty
        uncertainty_dict["fact_uncertainty"] = fact_uncertainty_list
        uncertainty_dict["fact_labels"] = fact_labels
        uncertainty_dict["fact_check_labels"] = fact_check_labels
        uncertainty_list.append(uncertainty_dict)
        pbar.update(1)
    print("err_cnt = "+str(err_cnt))    
    with open(output_path,"w",encoding="utf-8") as file:
        json.dump(uncertainty_list, file, ensure_ascii=False, indent=4)    
            
compute_uncertainty()         
            