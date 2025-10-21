import json
from template_opt import Reasoner
import argparse
import os
import datetime
import hydra
import logging
from omegaconf import DictConfig
from utils import extract_objective, bool_accuracy
import concurrent.futures
import multiprocessing
import signal
def process_item(line, cfg):
    # 获取当前进程的 ID
    data = json.loads(line)
    id = data["id"]

    # 创建日志文件路径
    log_filename = os.path.join(cfg.outpath, f"logs/problem_{id}.log")

    # 配置日志记录器
    logging.root.handlers = []

    # 重新配置日志记录器，只写入到文件
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename)
        ]
    )

    try:
        data = json.loads(line)
        input = data['question']
        truth = data['ground_truth']
        id = data["id"]

        logging.info(input)
        if truth != "infeasible":
            truth = float(truth)
        MILP = """
Let's solve a Mixed Integer Linear Programming (MILP) problem. You will be given a problem description that may include mathematical expressions or only natural language. Your goal is to formulate the problem as a MILP and provide a solution.
Input:
"""
        user_input = MILP + input
        reasoner = Reasoner(
            user_input=user_input,
            api_key=cfg.api_key,
            llm=cfg.llm,
            need_check=True,
            base_url=cfg.url,
            template_path=cfg.template_path,
            tree_path=cfg.tree_path
        )
        
        problem_type, tree_end, result = reasoner.reason_run()
        value = extract_objective(result)
        compare = bool_accuracy(value, truth)
        judge = {'id':id, 'type': problem_type,  'accuracy': compare, 'tree_end': tree_end, 'solve': value, 'truth': truth}
        tmp = {'id':id, 'input': input, 'result': result}

            
        solve_file = os.path.join(cfg.outpath, f"solve_{cfg.task_name}.json")
        with open(solve_file, 'a', encoding='utf-8') as file:
            file.write(json.dumps(judge) + '\n')
        
        judge_file = os.path.join(cfg.outpath, f"judge_{cfg.task_name}.json")
        with open(judge_file, 'a', encoding='utf-8') as file:
            file.write(json.dumps(tmp) + '\n')
        
        print(f"Processed line: {id}")
        return True
    except Exception as e:
        logging.error(f"Error processing line: {line}. Error: {e}")
        return False

def signal_handler(signal, frame):
    #print("Main process received signal. Terminating workers...")
    for p in multiprocessing.active_children():
        p.terminate()
        p.join()
    #print("All workers terminated. Exiting main process.")
    exit(0)


@hydra.main(version_base=None, config_path="conf", config_name="test")
def run_test(cfg: DictConfig) -> None:
    output_dir = cfg.outpath
    path = cfg.path
    
    # 确保日志目录存在
    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    with open(path, 'r') as file:
        lines = file.readlines()

    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = []
        id = 0
        for line in lines:
            id+=1

            #if id not in list:
            #    continue
            futures.append(executor.submit(process_item, line, cfg))
        try:
            for future in concurrent.futures.as_completed(futures):
                try:
                    success = future.result()
                    if not success:
                        print(f"Processing failed for a line.")
                except Exception as e:
                    print(f"Task generated an exception: {e}")
        except KeyboardInterrupt:
            #print("Main process received KeyboardInterrupt. Terminating workers...")
            for future in futures:
                future.cancel()


        
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    run_test()