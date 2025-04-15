import json
import openai
import time
import argparse
import logging
from tqdm import tqdm
import re


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

with open("/data/zhangyuhao/poly_graph/lm_polygraph/result/ccp_in_all_name_mistral-7b_new_result.json","r",encoding="utf-8") as file:
    gen_data = json.load(file)
    file.close()


pbar = tqdm(total=len(gen_data))
for idx in range(len(gen_data)):
    fake_labels = []
    gdt = gen_data[idx]
    question = gdt["question"]
    generation = gdt["generation"]
    labels = []
    CHECKING_PROMPT = (
f'''Given the generation, please answer whether hallucinations occurred during the generation and provide the reason. If hallucinations occurred, answer with "Yes" or "No" and wrap with "[]".
Example:
[Generation]: Albert Einstan was a renowned German-born theoretical physicist who is widely regarded as one of the most influential scientists of the 20th century. He was born on March 14, 1879, in Ulm, Kingdom of W ürttemberg,  German Empire. Einstan is best known for his theory of relativity and the famous equation E=mc². He was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.
[Answer]: Albert Einstan is wrong. Albert Einstein is right. The answer is [Yes].

[Generation]: {generation}
[Answer]: 
'''
    )
    API_BASE = "https://api.lingyiwanwu.com/v1"
    API_KEY = "03415280c5ab4d9993b799e56443c2eb"
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
    gen_data[idx]["check_label"] = labels
    pbar.update(1)
    

with open("/data/zhangyuhao/poly_graph/lm_polygraph/result/ccp_in_all_name_with_label_mistral-7b_new_result.json","w",encoding="utf-8") as file:
    json.dump(gen_data, file, ensure_ascii=False, indent=4)