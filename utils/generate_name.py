import csv_to_json
import os
import time
import logging
import openai
import json
import re

cache_path = "YOUR_CACHE_DATA_SAVING_PATH"
log = logging.getLogger()
API_BASE = "https://api.lingyiwanwu.com/v1"
API_KEY = "YOUR_API_KEY"

def name_extractor(name_json_path):
    name_list = []
    data = csv_to_json.read_json(name_json_path)
    for idx,question in enumerate(data):
        name = question[17:-2]
        name_list.append(name)
    return name_list

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

def name_generator(real_name_data,model_name,num_of_gen):
    if API_KEY is not None:
        openai.api_key = API_KEY
    if API_BASE is not None:
        openai.api_base = API_BASE
        
    message = f'''
[Instrction]:
Please generate {str(num_of_gen)} names of people who appear to exist but do not actually exist based on the following names in the Namelist. Please output in list format.
Strategies that can be used include but are not limited to confusion reorganization and word fine-tuning: 

[Namelist]:
{str(real_name_data)}

[Names you generate]:
'''
    messages = [
        {"role": "system", "content": "You are an intelligent assistant."},
        {"role": "user", "content": message},
    ]
    chat = send_request(model_name,messages)
    reply = chat.choices[0].message.content
    model_reply = dict()
    model_reply[message] = str(reply)
    with open(f"{cache_path}/{model_name}_name_generator_reply.json","w",encoding="utf-8") as file:
        json.dump(model_reply, file, ensure_ascii=False, indent=4)
    
    if "please provide" in reply.lower():
        return ""
    if "to assist you" in reply.lower():
        return ""
    if "as an ai language model" in reply.lower():
        return ""
    
    fake_name_list = []
    if '[' in reply and ']' in reply:
        _reply = reply[reply.find('[')+1:reply.find(']')]
        if ',' in _reply:
            fake_name_list = _reply.split(",")
        else:
            print("Please process the response manually.")
            return reply
    else:
        pattern = re.compile(r'\b\d+\..+\n')
        matches = pattern.findall(reply)
        if len(matches) > 0:
            for match in matches:
                fake_name = (match[match.find('.')+1:]).lstrip(' ').rstrip('\n')
                fake_name_list.append(fake_name)
        else:
            print("Please process the response manually.")
            return reply
                
    return fake_name_list
    
    


if __name__ == "__main__":
    real_name_ques_path = "/data/zhangyuhao/robust_uncertainty/data/test.json"
    model_name = "yi-lightning"
    num_of_gen = 100
    name_list = name_extractor(real_name_ques_path)
    fake_name_list = name_generator(name_list,model_name,num_of_gen)
    with open(f"{cache_path}/{model_name}_fake_name_list.json","w",encoding="utf-8") as file:
        json.dump(fake_name_list, file, ensure_ascii=False, indent=4)
