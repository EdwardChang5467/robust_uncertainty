import json
import evaluate
from scipy.stats import pearsonr, spearmanr

def get_rocauc(uncertainty_scores, label_scores):
    roc_auc = evaluate.load("roc_auc")
    roc_auc_result = roc_auc.compute(references=label_scores, prediction_scores=uncertainty_scores)
    return roc_auc_result["roc_auc"]

with open("/data/zhangyuhao/poly_graph/lm_polygraph/result/ccp_in_all_name_with_label_mistral-7b_new_result.json", 'r', encoding='utf-8') as file:
    data = json.load(file)
    file.close()

mean_ccp = []
max_ccp = []
labels = []
mean_max_prob = []
max_max_prob = []
# print(data[54])
for idx,dt in enumerate(data):
    if len(dt["ccp_uncertainty"])>0 and len(dt["check_label"])>0:
        if dt["check_label"][0] == 1:
            labels.append(1)
        else:
            labels.append(0)
        
        ccp_uncertainty = dt["ccp_uncertainty"]
        max_prob_uncertainty = dt["max_prob_uncertainty"]
        max_ccp.append(max(ccp_uncertainty))
        max_max_prob.append(max(max_prob_uncertainty))
        mean_ccp.append(sum(ccp_uncertainty)/len(ccp_uncertainty))
        mean_max_prob.append(sum(max_prob_uncertainty)/len(max_prob_uncertainty))

mean_ccp_roc_auc = get_rocauc(mean_ccp,labels)
max_ccp_roc_auc = get_rocauc(max_ccp,labels)
max_max_prob_roc_auc = get_rocauc(max_max_prob,labels)
mean_max_prob_roc_auc = get_rocauc(mean_max_prob,labels)
print("mean_ccp_roc_auc:"+str(mean_ccp_roc_auc))
print("max_ccp_roc_auc:"+str(max_ccp_roc_auc))
print("max_max_prob_roc_auc:"+str(max_max_prob_roc_auc))
print("mean_max_prob_roc_auc:"+str(mean_max_prob_roc_auc))

mean_ccp_corr_pearson, _ = pearsonr(mean_ccp,labels)
mean_ccp_corr_spearman, _ = spearmanr(mean_ccp,labels)
max_ccp_corr_pearson, _ = pearsonr(max_ccp,labels)
max_ccp_corr_spearman, _ = spearmanr(max_ccp,labels)
mean_max_prob_corr_pearson, _ = pearsonr(mean_max_prob,labels)
mean_max_prob_corr_spearman, _ = spearmanr(mean_max_prob,labels)
max_max_prob_corr_pearson, _ = pearsonr(max_max_prob,labels)
max_max_prob_corr_spearman, _ = spearmanr(max_max_prob,labels)
print("mean_ccp_corr_pearson: " + str(mean_ccp_corr_pearson))
print("mean_ccp_corr_spearman: " + str(mean_ccp_corr_spearman))
print("max_ccp_corr_pearson: " + str(max_ccp_corr_pearson))
print("max_ccp_corr_spearman: " + str(max_ccp_corr_spearman))
print("mean_max_prob_corr_pearson: " + str(mean_max_prob_corr_pearson))
print("mean_max_prob_corr_spearman: " + str(mean_max_prob_corr_spearman))
print("max_max_prob_corr_pearson: " + str(max_max_prob_corr_pearson))
print("max_max_prob_corr_spearman: " + str(max_max_prob_corr_spearman))
