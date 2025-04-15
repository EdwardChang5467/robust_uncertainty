import json
import csv_to_json
import evaluate
import numpy as np
from sklearn.metrics import cohen_kappa_score

label_path = "/data/zhangyuhao/robust_uncertainty/data/yi-lightning_fake_name_label.json"
yi_label_path = "/data/zhangyuhao/robust_uncertainty/data/yi-lightning_fake_name_yi-lightning_label_2.json"
multi_sample_yi_answer_path = "/data/zhangyuhao/robust_uncertainty/data/yi-lightning_cleaned_fake_names_sample7.json"

def get_metrics(TP,TN,FP,FN):
    # get accuracy,recall,f1 score
    return (float(TP+TN)/float(TP+TN+FP+FN), float(TP)/float(TP+FN), float(2*TP)/float(2*TP+FP+FN))

def compute_kappa_with_ground(model_answer_list, ground_answer_list):
    # compute Cohen's Kappa
    # consider Random consistency
    return cohen_kappa_score(model_answer_list, ground_answer_list)
    

def get_rocauc(model_answer_list, ground_answer_list):
    model_label_list = []
    ground_label_list = []
    for i in range(len(ground_answer_list)):
        if model_answer_list[i] == "No":
            model_label_list.append(1)
        elif model_answer_list[i] == "Yes":
            model_label_list.append(0)
        else:
            model_label_list.append(1)
            
        if ground_answer_list[i] == "Yes":
            ground_label_list.append(0)
        else:
            ground_label_list.append(1)
    roc_auc = evaluate.load("roc_auc")
    roc_auc_result = roc_auc.compute(references=ground_label_list, prediction_scores=model_label_list)
    return roc_auc_result["roc_auc"]

def evaluate_fake_name_label(label_path,yi_label_path,multi_sample_yi_answer_path):
    labels_dict = csv_to_json.read_json(label_path)
    yi_label_dict = csv_to_json.read_json(yi_label_path)
    multi_sample_yi_answer = csv_to_json.read_json(multi_sample_yi_answer_path)
    ground_label_list = []
    gpt_4o_label_list = []
    kimi_label_list = []
    yi_label_list = []
    name_list = []
    for name,labels in labels_dict.items():
        name_list.append(name)
        ground_label_list.append(labels["manual"])
        gpt_4o_label_list.append(labels["GPT-4o"])
        kimi_label_list.append(labels["kimi"])
    for name,yi_label in yi_label_dict.items():
        yi_label_list.append(yi_label)
    
    yi_TP = 0
    yi_TN = 0
    yi_FP = 0
    yi_FN = 0
    gpt_4o_TP = 0 
    gpt_4o_TN = 0
    gpt_4o_FP = 0
    gpt_4o_FN = 0
    kimi_TP = 0
    kimi_TN = 0
    kimi_FP = 0
    kimi_FN = 0   
    multi_yi_TP = 0
    multi_yi_FP = 0
    multi_yi_TN = 0
    multi_yi_FN = 0
    multi_yi_labels = []
    # calculate precision and recall 
    for i in range(len(ground_label_list)):
        if ground_label_list[i] == "No" and yi_label_list[i] == "No":
            yi_TP = yi_TP + 1
        elif ground_label_list[i] == "No" and yi_label_list[i] == "Yes":
            yi_FN = yi_FN + 1
        elif ground_label_list[i] == "Yes" and yi_label_list[i] == "Yes":
            yi_TN = yi_TN + 1
        elif ground_label_list[i] == "Yes" and yi_label_list[i] == "No":
            yi_FP = yi_FP + 1
            
        if ground_label_list[i] == "No" and gpt_4o_label_list[i] == "No":
            gpt_4o_TP = gpt_4o_TP + 1
        elif ground_label_list[i] == "No" and gpt_4o_label_list[i] == "Yes":
            gpt_4o_FN = gpt_4o_FN + 1
        elif ground_label_list[i] == "Yes" and gpt_4o_label_list[i] == "Yes":
            gpt_4o_TN = gpt_4o_TN + 1
        elif ground_label_list[i] == "Yes" and gpt_4o_label_list[i] == "No":
            gpt_4o_FP = gpt_4o_FP + 1
        
        if ground_label_list[i] == "No" and kimi_label_list[i] == "No":
            kimi_TP = kimi_TP + 1
        elif ground_label_list[i] == "No" and kimi_label_list[i] == "Yes":
            kimi_FN = kimi_FN + 1
        elif ground_label_list[i] == "Yes" and kimi_label_list[i] == "Yes":
            kimi_TN = kimi_TN + 1
        elif ground_label_list[i] == "Yes" and kimi_label_list[i] == "No":
            kimi_FP = kimi_FP + 1
        
        if ground_label_list[i] == "No" and name_list[i] in multi_sample_yi_answer:
            multi_yi_TP = multi_yi_TP + 1
        elif ground_label_list[i] == "No" and name_list[i] not in multi_sample_yi_answer:
            multi_yi_FN = multi_yi_FN + 1
        elif ground_label_list[i] == "Yes" and name_list[i] in multi_sample_yi_answer:
            multi_yi_FP = multi_yi_FP + 1
        elif ground_label_list[i] == "Yes" and name_list[i] not in multi_sample_yi_answer:
            multi_yi_TN = multi_yi_TN + 1
            
        if name_list[i] in multi_sample_yi_answer:
            multi_yi_labels.append("No")
        else:
            multi_yi_labels.append("Yes")
    
    cnt = 0
    for j in range(len(yi_label_list)):
        if gpt_4o_label_list[j] != ground_label_list[j]:
            cnt = cnt + 1
    print(cnt)
    
    yi_metrics = get_metrics(yi_TP,yi_TN,yi_FP,yi_FN)
    gpt_4o_metrics = get_metrics(gpt_4o_TP,gpt_4o_TN,gpt_4o_FP,gpt_4o_FN)
    kimi_metrics = get_metrics(kimi_TP,kimi_TN,kimi_FP,kimi_FN)
    multi_yi_metrics = get_metrics(multi_yi_TP,multi_yi_TN,multi_yi_FP,multi_yi_FN)
    yi_rocauc = get_rocauc(yi_label_list,ground_label_list)
    gpt_4o_rocauc = get_rocauc(gpt_4o_label_list,ground_label_list)
    kimi_rocauc = get_rocauc(kimi_label_list,ground_label_list)
    multi_yi_rocauc = get_rocauc(multi_yi_labels,ground_label_list)
    yi_kappa = compute_kappa_with_ground(yi_label_list,ground_label_list)
    gpt_4o_kappa = compute_kappa_with_ground(gpt_4o_label_list,ground_label_list)
    kimi_kappa = compute_kappa_with_ground(kimi_label_list,ground_label_list)
    multi_yi_kappa = compute_kappa_with_ground(multi_yi_labels,ground_label_list)
    
    print("-------------------------")
    print("GPT-4o metircs:")
    print("Accuracy:"+str(gpt_4o_metrics[0]))
    print("Recall:"+str(gpt_4o_metrics[1]))
    print("F1 score:"+str(gpt_4o_metrics[2]))
    print("ROCAUC:"+str(gpt_4o_rocauc))
    print("Cohen's Kappa:"+str(gpt_4o_kappa))
    print("-------------------------")
    print("-------------------------")
    print("kimi metircs:")
    print("Accuracy:"+str(kimi_metrics[0]))
    print("Recall:"+str(kimi_metrics[1]))
    print("F1 score:"+str(kimi_metrics[2]))
    print("ROCAUC:"+str(kimi_rocauc))
    print("Cohen's Kappa:"+str(kimi_kappa))
    print("-------------------------")
    print("-------------------------")
    print("yi-lightning metircs:")
    print("Accuracy:"+str(yi_metrics[0]))
    print("Recall:"+str(yi_metrics[1]))
    print("F1 score:"+str(yi_metrics[2]))
    print("ROCAUC:"+str(yi_rocauc))
    print("Cohen's Kappa:"+str(yi_kappa))
    print("-------------------------")
    print("-------------------------")
    print("yi-lightning multi-sample metircs:")
    print("Accuracy:"+str(multi_yi_metrics[0]))
    print("Recall:"+str(multi_yi_metrics[1]))
    print("F1 score:"+str(multi_yi_metrics[2]))
    print("ROCAUC:"+str(multi_yi_rocauc))
    print("Cohen's Kappa:"+str(multi_yi_kappa))
    print("-------------------------")

if __name__ == "__main__":
    evaluate_fake_name_label(label_path,yi_label_path,multi_sample_yi_answer_path)    