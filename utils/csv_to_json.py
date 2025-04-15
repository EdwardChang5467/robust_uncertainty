import pandas as pd
import sys
import json

def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def csv_with_one_key_2_list(csv_path):
    question_list = []
    df = pd.read_csv(csv_path)
    keys = list(df.columns)
    if len(keys) == 1:
        for i in range(len(df[keys[0]])):
            question = df[keys[0]][i]
            question_list.append(question)
            
    else:
        print("Please input a csv file with one column.")
        sys.exit()
    return question_list

def preprocess_data(csv_path, json_save_path):
    # csv_path = "/data/zhangyuhao/robust_uncertainty/data/test.csv"
    question_list = csv_with_one_key_2_list(csv_path)
    # json_save_path = "../data/test.json"
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(question_list, f, ensure_ascii=False, indent=4)


