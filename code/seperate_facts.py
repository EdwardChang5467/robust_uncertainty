import json
import openai
import logging
import time
from tqdm import tqdm
import sys
import re
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", required=True, type=str, help="file path")
# parser.add_argument("--model_name", required=True, type=str, help="generation model name")
args = parser.parse_args()


file_path = args.file_path
model_name = "yi-lightning"
log = logging.getLogger()
API_BASE = "https://api.lingyiwanwu.com/v1"
API_KEY = "YOUR_API_KEY"

def send_request(openai_model,messages):
    sleep_time_values = (5, 10, 30, 60, 120)
    for i in range(len(sleep_time_values)):
        try:
            return openai.ChatCompletion.create(
                model=openai_model, messages=messages
            )
        except Exception as e:
            sleep_time = sleep_time_values[i]
            log.info(
                f"Request to OpenAI failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
            )
        time.sleep(sleep_time)

    return openai.ChatCompletion.create(model=openai_model, messages=messages)

def fact_seperator(model_name,file_path):
    if API_KEY is not None:
        openai.api_key = API_KEY
    if API_BASE is not None:
        openai.api_base = API_BASE
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        file.close()
    model_reply = dict()
    model_answer = dict()
    final_check_dict = dict()
    pbar = tqdm(total=len(data))
    for idx,data_dict in enumerate(data):
        question = data_dict["question"]
        answers = data_dict["generated_answer"]
        data_facts = []
        for generation in answers:
            if "couldn't find any information" in generation or "misspell" in generation or "provide more" in generation or "mistake" in generation or "Unfortunately" in generation:
                generation_facts = []
            else:
                # sentences = [(sent+".").lstrip(' ') for sent in generation.split('.')[:-1]]
                sentences = re.split(r'(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z]\.)(?<=\.|\?)\s', generation)
                # sentences = [sent for sent in sentences if len(sent) > 0 and (sent[-1] == '.' or sent[-1] == '!')]
                sentences = [sent for sent in sentences if sent.strip() and (sent.strip()[-1] == '.' or sent.strip()[-1] == '!')]
                replies = []
                for sentence in sentences:
                    CLAIM_EXTRACTION_PROMPT = f"""Please breakdown the sentence into independent claims. Do not modify the spelling of personal names.

    Example 1:
    Sentence: \"He was born in London and raised by his mother and father until 11 years old.\"
    Claims:
    - He was born in London.
    - He was raised by his mother and father.
    - He was raised by his mother and father until 11 years old.

    Example 2:
    Sentence:\"Albert Einstan was a renowned German-born theoretical physicist who is widely regarded as one of the most influential scientists of the 20th century.\"
    Claims:
    - Albert Einstan was a renowned theoretical physicist.
    - Albert Einstan was German-born.
    - Albert Einstan is widely regarded as a scientist.
    - Albert Einstan is widely regarded as one of the most influential scientists.
    - Albert Einstan is widely regarded as one of the most influential scientists of the 20th century.

Sentence: \"{sentence}\"
Claims:"""
                    messages = [
                        {"role": "system", "content": "You are an intelligent assistant."},
                        {"role": "user", "content": CLAIM_EXTRACTION_PROMPT},
                    ]
                    chat = send_request(model_name,messages)
                    reply = chat.choices[0].message.content
                    replies.append(reply)
                pattern = re.compile(r'\b\d+\..+\n')
                generation_facts = []
                for reply in replies:
                    matches1 = pattern.findall(reply)
                    facts = []
                    if len(matches1) > 0:
                        for match in matches1:
                            fact = match[match.find('.')+1:].lstrip(" ").rstrip("\n")
                            facts.append(fact)
                    else:
                        if reply.find("-") >= 0:
                            facts = reply.split('\n-')
                            facts = [fct.lstrip('-').lstrip(' ') for fct in facts if "independent claims" not in fct and "Claims" not in fct]
                        else:
                            facts = []
                    generation_facts.append(facts)
            data_facts.append(generation_facts)
        data[idx]["facts"] = data_facts
        # if idx == 1:
        #     with open("/data/zhangyuhao/robust_uncertainty/data/jais_13b/fact_test.json","w",encoding="utf-8") as file:
        #         json.dump(data, file, ensure_ascii=False, indent=4)
        pbar.update(1)
    
    save_path = file_path.split(".json")[0] + "_fact_v1.json"
    with open(save_path,"w",encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
        
    

if __name__ == "__main__":
    fact_seperator("yi-lightning",file_path)  
    