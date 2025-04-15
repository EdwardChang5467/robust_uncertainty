import json
import evaluate
from scipy.stats import pearsonr, spearmanr

def get_rocauc(uncertainty_scores, label_scores):
    roc_auc = evaluate.load("roc_auc")
    roc_auc_result = roc_auc.compute(references=label_scores, prediction_scores=uncertainty_scores)
    return roc_auc_result["roc_auc"]

with open("/data/zhangyuhao/robust_uncertainty/data/chatglm3-6b-32k_new/chatglm3-6b-32k_new_sample5_temp1.0_pe_result_v1.json", 'r', encoding='utf-8') as file1:
    data = json.load(file1)
    file1.close()

with open("/data/zhangyuhao/robust_uncertainty/data/chatglm3-6b-32k_new/chatglm3-6b-32k_new_sample5_temp1.0_result_v1_fact_v1_with_label_map_v1_check_lable_v1.json", 'r', encoding='utf-8') as file2:
    labels_data = json.load(file2)
    file2.close()

all_labels = []
all_pe = []
all_lnpe = []

for idx,dt in enumerate(data):
    predictive_entropy = dt["predictive_entropy"]
    ln_predictive_entropy = dt["ln_predictive_entropy"]
    generations_labels = labels_data[idx]["check_labels"]
    if len(generations_labels)>0 and len(generations_labels) == len(predictive_entropy) and len(ln_predictive_entropy)>0:
        for label in generations_labels:
            if label == 1:         
                all_labels.append(1)
            else:
                all_labels.append(0)
        for en in predictive_entropy:
            all_pe.append(en)
        for ln_en in ln_predictive_entropy:
            all_lnpe.append(ln_en)

pe_rocauc = get_rocauc(all_pe,all_labels)
pe_corr_pearson, _ = pearsonr(all_pe,all_labels)
pe_corr_spearman, _ = spearmanr(all_pe,all_labels)
lnpe_rocauc = get_rocauc(all_lnpe,all_labels)
lnpe_corr_pearson, _ = pearsonr(all_lnpe,all_labels)
lnpe_corr_spearman, _ = spearmanr(all_lnpe,all_labels)

print("pe_pearson_corr: "+str(pe_corr_pearson))
print("pe_spearman_corr: "+str(pe_corr_spearman))
print("pe_rocauc: "+str(pe_rocauc))
print("lnpe_pearson_corr: "+str(lnpe_corr_pearson))
print("lnpe_spearman_corr: "+str(lnpe_corr_spearman))
print("lnpe_rocauc: "+str(lnpe_rocauc))
        