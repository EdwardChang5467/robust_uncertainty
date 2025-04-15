import openai
import logging
import time
from tqdm import tqdm
import sys
import re
import json
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--generation_path", required=True, type=str, help="generation file path")
parser.add_argument("--true_name_path", required=True, type=str, help="true name file path")
parser.add_argument("--fake_name_path", required=True, type=str, help="fake name file path")
parser.add_argument("--output_path", required=True, type=str, help="result file path")
args = parser.parse_args()

generation_path = args.generation_path
output_path = args.output_path
true_name_path = args.true_name_path
fake_name_path = args.fake_name_path

def classify_generation(generation_path,output_path):
    with open(generation_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        file.close()
    with open(true_name_path, 'r', encoding='utf-8') as file:
        true_name_list = json.load(file)
        file.close()
    with open(fake_name_path, 'r', encoding='utf-8') as file:
        fake_name_ques_list = json.load(file)
        file.close()
    fake_name_list = []
    for jdx in range(len(fake_name_ques_list)):
        fake_ques = fake_name_ques_list[jdx]
        fake_name = (fake_ques.split("Tell me a bio of ")[1]).rstrip(".\n")
        fake_name_list.append(fake_name)
    
    for idx, dt in enumerate(data):
        generations = dt["generated_answer"]
        # generations = dt["generation"]
        if type(generations) != list:
            generations = [generations]
        question = dt["question"]
        fake_labels = []
        name = (question.split("Tell me a bio of ")[1]).rstrip(".\n")
        for generation in generations:
            if "couldn't find any information" in generation or "I am not sure" in generation or "misspell" in generation or "provide more" in generation or "mistake" in generation or "Unfortunately" in generation:
                fake_labels.append("FR") # fake right
            else:
                if name not in generation:
                    flag = 0
                    for true_name in true_name_list:
                        if true_name in generation:
                            if idx == 0:
                                print("111111")
                            fake_labels.append("FT") # fake correct true
                            flag = 1
                            break
                    if flag == 0:
                        if idx == 0:
                            print("2222222")
                        fake_labels.append("FF")
                        
                else:
                    if name not in fake_name_list:
                        fake_labels.append("FT")
                    else:
                        fake_labels.append("FF") # fake correct false
        data[idx]["generation_labels"] = fake_labels
    
    with open(output_path,"w",encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


classify_generation(generation_path,output_path)