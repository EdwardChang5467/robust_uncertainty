import json
import csv_to_json

save_file = "/data/zhangyuhao/robust_uncertainty/data/yi-lightning_fake_name_yi-lightning_label_from_7sample.json"
yi_check_file = "/data/zhangyuhao/robust_uncertainty/cache/yi-lightning_name_checker_answer_sample7.json"

def get_yi_label(yi_check_file,save_file):
    data = csv_to_json.read_json(yi_check_file)
    yi_label_dict = dict()
    for name,answers in data.items():
        yi_label_dict[name] = answers[0]
    with open(save_file,"w",encoding="utf-8") as file:
        json.dump(yi_label_dict, file, ensure_ascii=False, indent=4)


get_yi_label(yi_check_file,save_file)
    