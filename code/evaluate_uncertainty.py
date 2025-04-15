import json
import openai
import logging
import time
from tqdm import tqdm
import sys
import re
import numpy as np
import argparse
from typing import Optional
import logging
import evaluate
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--label_path", required=True, type=str, help="fact with label file path")
parser.add_argument("--uncertainty_path", required=True, type=str, help="uncertainty file path")
parser.add_argument("--output_path", required=True, type=str, help="result file path")
args = parser.parse_args()

label_path = args.label_path
output_path = args.output_path
# check_label_path = args.check_label_path
uncertainty_path = args.uncertainty_path

def get_rocauc(uncertainty_scores, label_scores):
    roc_auc = evaluate.load("roc_auc")
    roc_auc_result = roc_auc.compute(references=label_scores, prediction_scores=uncertainty_scores)
    return roc_auc_result["roc_auc"]

def get_uncertainty_corr_with_fabrication_rate():
    with open(label_path, 'r', encoding='utf-8') as file1:
        label_data = json.load(file1)
        file1.close()
    with open(uncertainty_path, 'r', encoding='utf-8') as file2:
        uncertainty_data = json.load(file2)
        file2.close()
    uncertainty_scores = []
    label_scores = []
    print(len(uncertainty_data))
    print(len(label_data))
    for idx,data in enumerate(label_data):
        labels = data["check_labels"]
        if len(labels) == len(uncertainty_data[idx]["uncertainty"]) and len(labels) > 0:
            for jdx,label in enumerate(labels):
                uncertainty_scores.append(uncertainty_data[idx]["uncertainty"][jdx])
                if label == 1:
                    label_scores.append(1)
                else:
                    label_scores.append(0)

    corr_pearson, _ = pearsonr(uncertainty_scores, label_scores)
    corr_spearman, _ = spearmanr(uncertainty_scores, label_scores)
    roc_auc = get_rocauc(uncertainty_scores,label_scores)
    return (corr_pearson,corr_spearman,roc_auc)

def get_fact_uncertainty_corr_with_fabrication_rate():
    with open(uncertainty_path, 'r', encoding='utf-8') as file2:
        uncertainty_data = json.load(file2)
        file2.close()
    uncertainty_scores = []
    label_scores = []
    for idx,data in enumerate(uncertainty_data):
        labels = data["fact_check_labels"]
        if len(labels) == len(uncertainty_data[idx]["fact_uncertainty"]):
            for jdx,label in enumerate(labels):
                uncertainty_scores.append(uncertainty_data[idx]["fact_uncertainty"][jdx])
                if label == 1:
                    label_scores.append(1)
                else:
                    label_scores.append(0)

    corr_pearson, _ = pearsonr(uncertainty_scores, label_scores)
    corr_spearman, _ = spearmanr(uncertainty_scores, label_scores)
    roc_auc = get_rocauc(uncertainty_scores,label_scores)
    return (corr_pearson,corr_spearman,roc_auc)

corr_pearson,corr_spearman,roc_auc = get_uncertainty_corr_with_fabrication_rate()
print("pearson_corr:" + str(corr_pearson))
print("spearman_corr:" + str(corr_spearman))
print("rocauc:"+str(roc_auc))

corr_pearson,corr_spearman,roc_auc = get_fact_uncertainty_corr_with_fabrication_rate()
print("fact_pearson_corr:" + str(corr_pearson))
print("fact_spearman_corr:" + str(corr_spearman))
print("fact_rocauc:"+str(roc_auc))

        
    