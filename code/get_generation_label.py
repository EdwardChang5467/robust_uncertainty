import json
import openai
import time
import argparse
import logging
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--generation_path", required=True, type=str, help="fact with label file path")
parser.add_argument("--output_path", required=True, type=str, help="result file path")
args = parser.parse_args()

generation_path = args.generation_path
output_path = args.output_path

def send_request(openai_model,messages):
    sleep_time_values = (5, 10, 30, 60, 120)
    for i in range(len(sleep_time_values)):
        try:
            return openai.ChatCompletion.create(
                model=openai_model, messages=messages
            )
        except Exception as e:
            sleep_time = sleep_time_values[i]
        time.sleep(sleep_time)
        
    return openai.ChatCompletion.create(model=openai_model, messages=messages)

with open(generation_path,"r",encoding="utf-8") as file:
    data = json.load(file)
    file.close()

pbar = tqdm(total=len(data))
new_data = []
for idx,dt in enumerate(data):
    generations = dt["generated_answer"]
    labels = []
    for gen in generations:
        CHECKING_PROMPT = (
f'''Given the generation, please answer whether hallucinations occurred during the generation and provide the reason. If hallucinations occurred, answer with "Yes" or "No" and wrap with "[]".
Example:
[Generation]: Albert Einstan was a renowned German-born theoretical physicist who is widely regarded as one of the most influential scientists of the 20th century. He was born on March 14, 1879, in Ulm, Kingdom of W ürttemberg,  German Empire. Einstan is best known for his theory of relativity and the famous equation E=mc². He was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.
[Answer]: Albert Einstan is wrong. Albert Einstein is right. The answer is [Yes].

[Generation]: {gen}
[Answer]: 
'''
        )
        API_BASE = "https://api.lingyiwanwu.com/v1"
        API_KEY = "YOUR_API_KEY"
        openai.api_key = API_KEY
        openai.api_base = API_BASE
        model_name = "yi-lightning"

        messages = [
            {"role": "system", "content": "You are an intelligent assistant."},
            {"role": "user", "content": CHECKING_PROMPT},
        ]
        chat = send_request(model_name,messages)
        reply = chat.choices[0].message.content
        matches = re.findall(r'\[.*?\]', reply)
        if len(matches) > 0:
            if "[Yes]" in matches and "[No]" not in matches:
                labels.append(1)
            elif "[No]" in matches and "[Yes]" not in matches:
                labels.append(0)
            elif "[Yes]" in matches and "[No]" in matches:
                if matches[-1] == "[Yes]":
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                chat = send_request(model_name,messages)
                reply = chat.choices[0].message.content
                matches = re.findall(r'\[.*?\]', reply)
                if "[Yes]" in matches and "[No]" not in matches:
                    labels.append(1)
                elif "[No]" in matches and "[Yes]" not in matches:
                    labels.append(0)
                elif "[Yes]" in matches and "[No]" in matches:
                    if matches[-1] == "[Yes]":
                        labels.append(1)
                    else:
                        labels.append(0)
        else:
            chat = send_request(model_name,messages)
            reply = chat.choices[0].message.content
            matches = re.findall(r'\[.*?\]', reply)
            if "[Yes]" in matches and "[No]" not in matches:
                labels.append(1)
            elif "[No]" in matches and "[Yes]" not in matches:
                labels.append(0)
            elif "[Yes]" in matches and "[No]" in matches:
                if matches[-1] == "[Yes]":
                    labels.append(1)
                else:
                    labels.append(0)
        
    new_dict = dict()    
    new_dict["question"] = dt["question"]
    new_dict["check_labels"] = labels
    new_data.append(new_dict)
    pbar.update(1)
    
with open(output_path,"w",encoding="utf-8") as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)



