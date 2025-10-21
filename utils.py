
import re 
import json
import io
import sys
import logging
import gurobipy as gp
from gurobipy import GRB
from contextlib import redirect_stdout, redirect_stderr

class Agent:
    def __init__(self, system_message, client, llm="gpt-3.5-turbo", dialog_round=20, **kwargs):
        self.system_message = [
            {"role": "system", "content": system_message}
        ]
        self.dialog_round = dialog_round
        self.messages = []
        self.client = client
        self.kwargs = kwargs
        self.llm = llm

    def generate_reply(self, prompt = None, tem = 0.) -> str:
        self.messages.append({
            "role": "user",
            "content": prompt,	
        })
    
        new_messages = []
        new_messages.extend(self.system_message)
    
        if len(self.messages) > self.dialog_round:
            self.messages = self.messages[-self.dialog_round:]

        new_messages.extend(self.messages)

        #logging,info(new_messages)
        completion = self.client.chat.completions.create(
            model=self.llm,
            messages=new_messages,
            temperature=tem,
        )

        assistant_message = completion.choices[0].message.content
        self.messages.append({"role": "system", "content": assistant_message})

        return assistant_message
        
    def clear(self):
        self.messages = []
        return
    def system_update(self,system):
        self.system_message = system

def get_model_stats_from_code(code):
    """
    执行代码并从模型中获取统计信息
    """
    try:
        # 捕获输出，避免打印到控制台
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # 创建全局命名空间
            exec_globals = {
                'gp': gp,
                'GRB': GRB,
            }
            
            # 添加所有内置函数
            exec_globals.update(__builtins__ if isinstance(__builtins__, dict) else {})
            
            # # 创建本地命名空间
            # exec_locals = {}
            
            # 执行代码
            partial_code = ""
            last_line = ""
            bogus_context = None
            def code_split(code):
                code_lines = code.split('\n')
                code_list = []
                i = 0
                while i < len(code_lines):
                    if code_lines[i] == '' or (code_lines[i][0] != ' ' and code_lines[i][0] != ']' and code_lines[i][0] != ')' and code_lines[i][0] != '}' and code_lines[i][0:4] != 'elif'  and code_lines[i][0:4] != 'else'):
                        code_list.append(code_lines[i])
                    else:
                        code_block = code_list.pop()
                        code_block += '\n' + code_lines[i]
                        code_list.append(code_block)
                    i += 1
                # code_list.append('\n'.join(code_lines[-17:]))
                return code_list
        
            code_list = code_split(code)


            
            for i in range(len(code_list)):
                last_line = code_list[i]
                partial_code += last_line + "\n"
                bogus_context = last_line
                exec(last_line, exec_globals, exec_globals)
        
            # 从本地或全局命名空间中获取模型
            model = exec_globals.get('model')
            if model is None:
                # 在全局命名空间中查找
                for key, value in exec_globals.items():
                    if isinstance(value, gp.Model):
                        model = value
                        break
            
            if model is None:
                raise ValueError("未能找到Gurobi模型")
            
            # 更新模型以计算统计信息
            model.update()
            
            # 获取统计信息
            stats = {
                'variable_number': int(model.NumVars),
                'binary_variable_number': int(model.NumBinVars),
                'integer_variable_number': int(model.NumIntVars - model.NumBinVars),
                'constraint_number': int(model.NumConstrs),
                'nonzero_number': int(model.NumNZs), 
                'runtime_seconds': float(model.Runtime)
            }
            
            # 清理模型
            model.dispose()
            
            return stats
            
    except Exception as e:
        return {
            'variable_number': 0,
            'binary_variable_number': 0,
            'integer_variable_number': 0,
            'constraint_number': 0,
            'nonzero_number': 0,
            'runtime_seconds': 0,
            'error': f"{type(e).__name__}: {str(e)}"
        }

def replace_fraction(match):
    x, y = map(float, match.groups())
    return str(x / y)

def extract_json(output):

    if "```json" in output:
        output = output[output.find("```json") + 7 :]
        output = output[: output.rfind("```")]

    # go back until the last character is a }
    #print(output[-1])
    while output[-1] != "}":
        output = output[:-1]

    # go forward until the first character is a {
    while output[0] != "{":
        output = output[1:]

    # if there are '$' in the output, remove them
    if "$" in output:
        output = output.replace("$", "")

    if "\text" in output:
        output = output.replace("\text", "\\text")

    output = output.replace("\\", "\\\\")
    output = output.replace("\\", "\\\\")
    output = output.replace("\\\\quad", "\\\\")
    #output = output.replace("\n", "")
    output = re.sub(r'(\d+)/(\d+)', replace_fraction, output)
    #print(type(output))
    return json.loads(output)
    #return output

def extract_and_execute_code(text):
    # Possible start and end markers
    code_start_markers = ["```python", "```Python", "```"]
    code_end_marker = "```"

    # Find python part
    code_start_index = -1
    code_start_marker_used = None
    for marker in code_start_markers:
        code_start_index = text.lower().find(marker.lower())
        if code_start_index != -1:
            code_start_marker_used = marker
            break

    # If find code
    if code_start_index != -1:
        # Try to find the end point
        code_end_index = text.find(code_end_marker, code_start_index + len(code_start_marker_used))
        
        # If not, we assume the code is appended to the end of the text
        if code_end_index == -1:
            code_end_index = len(text)
        
        # Extract the code
        code_str = text[code_start_index + len(code_start_marker_used):code_end_index].strip()
        
        # Clean up the code string
        for marker in code_start_markers:
            code_str = code_str.replace(marker, "")
        code_str = code_str.replace(code_end_marker, "").strip()
        
        # Create a stream
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        # Execute the code
        try:
            exec(code_str, globals())
        except Exception as e:
            # Primary output
            sys.stdout = old_stdout
            return f"An error occurred: {e}", code_str
        
        # Extract the output
        sys.stdout = old_stdout
        return new_stdout.getvalue(), code_str
    else:
        return "No Python code found in the provided string.", None


def extract_single_objective(gurobi_output):
    
    optimal_objective_match = re.search(r"Optimal objective\s+([\d\.e\+\-]+)", gurobi_output)
    #print(optimal_objective_match)
    if optimal_objective_match:
        optimal_objective_value = optimal_objective_match.group(1)
        return optimal_objective_value
    else:
        
        best_objective_match = re.search(r"Best objective ([\d\.e\+\-]+)", gurobi_output)
        #print(best_objective_match)
        if best_objective_match :
            best_objective_value =best_objective_match.group(1) #float()
            if best_objective_value != "-":
                return best_objective_value

def extract_objective(gurobi_output):
    match = extract_single_objective(gurobi_output)
    if match:
        value = float(match)
    else:
        value = 'infeasible'
    return value

def bool_accuracy(value,truth):
    acc = False
    #print("value is")
    #print(value)
    #print("truth is")
    #print(truth)
    if value == truth:
        acc = True
    elif value!='infeasible' and truth !='infeasible' :
        truth = float(truth)
        if truth == 0:
            acc = (abs(value-truth) <= 0.05)
        elif (abs(value-truth)/truth)<= 0.05:
            acc = True
    return acc

def write_template(problem_type, template, template_path):
    with open(template_path, "a") as file:
        json_str = json.dumps({"problem": problem_type, "template": template})
        file.write(json_str + '\n')
        

def correct_template(problem_type, template, template_path):
    with open(template_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    for i, line in enumerate(lines):
        line_data = json.loads(line)
        if line_data['problem'] == problem_type:
            line_data['template'] = template
            lines[i] = json.dumps(line_data) + '\n'
            
    with open(template_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)




if __name__ == '__main__':
    with open("optmath_dataset/dataset_first_100_with_stats.json", 'r', encoding='utf-8') as file:
        lines = json.load(file)

    print(f"Total lines: {len(lines)}")
    
