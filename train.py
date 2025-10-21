import json
from template_opt import Reasoner
import argparse
import os
import datetime
import hydra
import logging
from omegaconf import DictConfig
from utils import extract_objective,bool_accuracy, extract_and_execute_code,Agent
import json_repair
import openai
@hydra.main(version_base=None, config_path="conf", config_name="train")

def train(cfg:DictConfig)->None:
    task = cfg.task_name
    api_key = cfg.api_key
    llm = cfg.llm
    base_url = cfg.url
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d-%H:%M:%S")
    output_dir = cfg.outpath
    path = cfg.path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    reasoner = Reasoner(
            user_input = None,
            api_key= api_key,
            llm= llm,
            need_check = True,
            base_url=base_url,
            template_path= cfg.template_path,
            tree_path= cfg.tree_path
        )
    id = 0
    with open(path, 'r', encoding='utf-8', errors='replace') as file:
        content = file.read()

    lines = content.splitlines()
    for line in lines:
        line.strip()
        id+=1

        input = json.loads(line)['question']
        solution_step = json.loads(line)["Response"]
        truth = json.loads(line)['answer']
        


        logging.info(f"Problem Id: {id}\n")
        logging.info(input)
        reasoner.update_input(input)
        update,father_type, son_type = reasoner.dynamic_adjustment(solution_step, truth)
        tmp = {"update":update, "father_type":father_type, "son_type":son_type}
        save_path = os.path.join(cfg.outpath, f"train_{task}.json")
        with open(save_path, 'a+', encoding='utf-8') as file:
            json_str = json.dumps(tmp)
            file.write(json_str + '\n')

if __name__ == "__main__":

    train()