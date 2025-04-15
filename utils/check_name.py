import json
import openai
import csv_to_json
import logging
import time
from tqdm import tqdm

cache_path = "YOUR_CACHE_DATA_SAVING_PATH"
result_path = "YOUR_DATA_SAVING_PATH"
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

def name_checker(model_name,cache_path,num_of_gen):
    if API_KEY is not None:
        openai.api_key = API_KEY
    if API_BASE is not None:
        openai.api_base = API_BASE
    data = csv_to_json.read_json(f"{cache_path}/{model_name}_fake_name_list.json")
    model_reply = dict()
    model_answer = dict()
    final_check_dict = dict()
    pbar = tqdm(total=len(data))
    for idx,fake_name in enumerate(data):
        message = f'''
[Instruction]:
Does {fake_name} a real person? Please provide 'Yes' or 'No' as the answer based on the facts.

[Answer]:
'''
        messages = [
            {"role": "system", "content": "You are an intelligent assistant."},
            {"role": "user", "content": message},
        ]
        
        reply_list = []
        answer_list = []
        yes_cnt = 0
        no_cnt = 0
        
        for i in range(num_of_gen):
            chat = send_request(model_name,messages)
            reply = chat.choices[0].message.content
            reply_list.append(reply)
            
            if "yes" in reply or "Yes" in reply:
                answer_list.append("Yes")
                yes_cnt = yes_cnt + 1
            elif "no" in reply or "No" in reply:
                answer_list.append("No")
                no_cnt = no_cnt + 1
        if yes_cnt > no_cnt:
            final_check = "Yes"
        elif yes_cnt < no_cnt:
            final_check = "No"
        else:
            chat = send_request(model_name,messages)
            reply = chat.choices[0].message.content
            reply_list.append(reply)
            if "yes" in reply or "Yes" in reply:
                answer_list.append("Yes")
                final_check = "Yes"
            elif "no" in reply or "No" in reply:
                answer_list.append("No")
                final_check = "No"   
        final_check_dict[fake_name] = final_check 
        model_reply[message] = reply_list
        model_answer[fake_name] = answer_list
        pbar.update(1)
    
    final_fake_name_list = []
    for name, check_result in final_check_dict.items():
        if check_result == "No":
            final_fake_name_list.append(name)   
    
    with open(f"{cache_path}/{model_name}_name_checker_reply.json","w",encoding="utf-8") as file:
        json.dump(model_reply, file, ensure_ascii=False, indent=4)
    with open(f"{cache_path}/{model_name}_name_checker_answer.json","w",encoding="utf-8") as file:
        json.dump(model_answer, file, ensure_ascii=False, indent=4)
    with open(f"{result_path}/{model_name}_cleaned_fake_names.json","w",encoding="utf-8") as file:
        json.dump(final_fake_name_list, file, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    name_checker("yi-lightning",cache_path,3)  
    