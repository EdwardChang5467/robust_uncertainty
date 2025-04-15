import csv_to_json
import json

data = csv_to_json.read_json("/data/zhangyuhao/robust_uncertainty/data/yi-lightning_fake_name_label.json")
trap_questions = []
for name,labels in data.items():
    if labels["manual"] == "No":
        trap_questions.append("Tell me a bio of "+name+".\n")
with open("/data/zhangyuhao/robust_uncertainty/data/trap_question/trap_questions.json","w",encoding="utf-8") as file:
    json.dump(trap_questions, file, ensure_ascii=False, indent=4)