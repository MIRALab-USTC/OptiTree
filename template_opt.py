import openai
from template_retrieve import TemplateMatcher
from utils import Agent, extract_and_execute_code
import logging
import json
from utils import extract_objective, bool_accuracy, write_template, correct_template, extract_json, get_model_stats_from_code
import json_repair
import time
reason_prompt = [
"""
You are a mathematical formulator working with a team of optimization experts. The objective is to tackle a complex optimization problem.
""",
"""
You are an expert in problem analysis and can apply previous problem-solving approaches to new issues. The user will provide a specific task description and a thought template. Your goal is to analyze the user's task and generate a specific solution based on the thought template. If the instantiated solution involves Python code, only provide the code and let the compiler handle it. If the solution does not involve code, provide a final answer that is easy to extract from the text.
It should be noted that all the python code should be within one code block, the answer should not include more than one code block! And strictly follow the thought-template to instantiate the gurobi code but you should also adjust the input parameter according to the user input!
       
User Input:
{user_input}
Thought template:
{thought_template}

Please analyze the above user task description and thought template, and generate a specific, detailed solution. If the solution involves Python code, only provide the code. If not, provide a clear and extractable final answer.        
"""
        ]
inspector_prompt =     """
You are an excellent python programming master who are proficient in analyzing and editing python code, and you are also good at understanding the real-world problem. Your task is:
1. Analyze the given python code
2. Edit the input code to make sure the edited code is correct and could run and solve the problem correctly.  
Your respond should follow the format below:
```python
## Edited code here
```
        """

correct_prompt = [
    """
You are a mathematical formulator working with a team of optimization experts. The objective is to tackle a complex optimization problem.
""", 
"""
Please list the steps to formulate a {problem_type} problem and correct the provided erroneous solution steps. You need to provide a corrected solution process which uses the gurobi code to solve it based on the given right answer and recommendations. Please output in JSON format.
Additionally, based on the description of similarities and differences, You should also determine whether to :
   - Overwrite the existing solution_step in the old template.
   - Add a new solution_step to the template.
 


Input Problem: {question}
Erroneous Solution Steps: {old_template}
Right Answer: {right_answer}
Similarities: {similarity}
Differences: {different}
Recommendations: {recommendation}


Here is an  output template example. Don't add any another item:

{{
    "problem_type": "Travelling Sales Person Problem",
    "description": "Given a list of cities and the distances between each pair, what is the shortest possible route that visits each city ​exactly once and returns to the starting city?",
    "reason_flow": [
        "[Define Decision Variables] Define decision variables for edges \\( x_{{ij}} \\) and possibly auxiliary variables for MTZ \\( u_i \\)",
        "[Define Objective Function] Sum of distances multiplied by \\( x_{{ij}} \\)",
        "[Define Degree Constraints] Each node entered and exited exactly once",
        "[Define Subtour Elimination Constraints] Subtour eliminationvia MTZ or callbacks",
        "[Comprehensive Verification] Check the common errors in the optimization model",
        "[Write Gurobi Code] Write the Gurobi code the solve the problem"
    ],
    "example_application": {{
        "example_problem": "",
        "solution_steps": [
            "**Define Variables**:\n   - Binary variables \\( x_{{ij}} \\) for each directed edge from city \\( i \\) to city \\( j \\) (\\( i \\neq j \\)).\n   - Continuous variables \\( u_i \\) for each city \\( i \\) (except the starting city) to enforce subtour elimination.",
            "**Objective Function**:\n   - Minimize the total distance: \\( \\text{{minimize}} \\sum_{{i,j}} d_{{ij}} x_{{ij}} \\).",
            "**Constraints**:\n   - **Assignment Constraints**: Each city is entered and exited exactly once.\n     - \\( \\sum_{{j \\neq i}} x_{{ij}} = 1 \\quad \\forall i \\)\n     - \\( \\sum_{{i \\neq j}} x_{{ij}} = 1 \\quad \\forall j \\)\n   - **Subtour Elimination (MTZ)**:\n     - \\( u_i - u_j + n x_{{ij}} \\leq n - 1 \\quad \\forall i, j \\neq 1, i \\neq j \\)\n     - \\( 1 \\leq u_i \\leq n - 1 \\quad \\forall i \\neq 1 \\)",
            "### Gurobi Code:\n```python\nimport gurobipy as gp\nfrom gurobipy import GRB\n\n# Distance matrix (example with 4 cities)\nn = 4\ndist = [\n    [0, 2, 9, 10],\n    [2, 0, 6, 4],\n    [9, 6, 0, 8],\n    [10, 4, 8, 0]\n]\n\n# Create model\nm = gp.Model(\'TSP\')\n\n# Variables: x[i,j] = 1 if edge i->j is used\nx = m.addVars(n, n, vtype=GRB.BINARY, name=\'x\')\nfor i in range(n):\n    x[i,i].UB = 0  # Disable self-loops\n\n# Assignment constraints\nfor i in range(n):\n    m.addConstr(gp.quicksum(x[i,j] for j in range(n) if j != i) == 1, name=f"out_{{i}}")\n    m.addConstr(gp.quicksum(x[j,i] for j in range(n) if j != i) == 1, name=f"in_{{i}}")\n\n# MTZ variables and constraints\nu = m.addVars(range(1, n), lb=1, ub=n-1, vtype=GRB.CONTINUOUS, name=\'u\')\n\nfor i in range(1, n):\n    for j in range(1, n):\n        if i != j:\n            m.addConstr(u[i] - u[j] + n * x[i,j] <= n - 1, name=f"mtz_{{i}}_{{j}}")\n\n# Objective function\nobj = gp.quicksum(x[i,j] * dist[i][j] for i in range(n) for j in range(n) if i != j)\nm.setObjective(obj, GRB.MINIMIZE)\n\n# Solve\nm.optimize()\n\n# Extract and print solution\nif m.status == GRB.OPTIMAL:\n    print(\'Optimal tour:\')\n    current = 0\n    tour = [current]\n    visited = set([current])\n    while len(visited) < n:\n        for j in range(n):\n            if j != current and x[current,j].X > 0.5:\n                tour.append(j)\n                current = j\n                visited.add(current)\n                break\n    tour.append(0)  # Return to start\n    print(\' -> \'.join(map(str, tour)))\n    print(f\'Total distance: {{m.ObjVal}}\')\nelse:\n    print(\'No solution found.\')\n```",
            "### Common Errors to Avoid:\n1. **Incorrect Subtour Elimination**: Ensure MTZ constraints exclude the starting city and are applied to correct indices.\n2. **Indexing Mistakes**: Use consistent 0-based or 1-based indexing for cities.\n3. **Self-Loops**: Explicitly disable \\( x_{{ii}} \\) variables.\n4. **Bounds on MTZ Variables**: Set \\( u_i \\) bounds correctly (\\( 1 \\leq u_i \\leq n-1 \\)).\n5. **Objective Function**: Ensure distances are correctly paired with \\( x_{{ij}} \\) and exclude \\( i = j \\).\n\nBy following these steps and avoiding common pitfalls, you can effectively model and solve TSP using Gurobi."
        ],
    }}
}}

Important:
- Use plain JSON without markdown syntax
- Ensure all quotes are properly escaped
- Include all required keys: problem_type, description, reason_flow, example_application
- problem_type must match the {right_answer} exactly
- the template solution_step must use the gurobi code to solve it.

"""

]

diffenrent_prompt = [
"""
You are a mathematical formulator working with a team of optimization experts. The objective is to tackle a complex optimization problem.You are a Template Optimization Assistant. Your task is to modify a template based on the input parameters and return the template in the required format.
""",
"""    
Your task is to compare an unknown problem-solving process with a known correct process to determine its correctness and identify the differences between them.

### Input:
- **Problem Description**: {problem_description}
- **Correct Process**: {truth}
- **Unknown Process**: {predict}

### Tasks:
1. **Compare Processes**:
   - List the similarities between the correct process and the unknown process.
   - List the differences between the correct process and the unknown process.Importantly ignore the impact of using different solvers such as gurobi and coptpy.
   - Analyze whether these differences impact the correctness of the unknown process,Importantly ignore the impact of using different solvers such as gurobi and coptpy.

2. **Assess Correctness**:
   - Determine if the unknown process is correct for probelm description based on the comparison.Importantly, if this unknown process leads to any errors, it must return incorrect.
   - If incorrect, identify the specific issues and suggest improvements.

3. **Output**:
   - Return your findings in JSON format, ensuring the structure is correct for parsing with Python's `json.loads()` function.

### Output Format:
```json
{{
    "is_correct": <boolean_value>,
    "similarities": [<list_of_similarities>],
    "differences": [<list_of_differences>],
    "reasoning": "<reasoning_process>",
    "recommendations": "<improvement_suggestions>"
}} 
```
- Importantly that I'm going to use python json.loads() function to parse the json file, so please make sure the format is correct (don't add ',' before enclosing '}}' or ']' characters.
- Generate the complete json file and don't omit anything.
- Use '```json' and '```' to enclose the json file.
"""       
           
]

direct_template ="""
import json
import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB

# Create a new model
model = gp.Model('model')

# define parameters

# define variables

# define constraints

# define objective 

# Optimize the model
model.optimize()
status = model.status

obj_val = None
# Check whether the model is infeasible, has infinite solutions, or has an optimal solution
if status == gp.GRB.INFEASIBLE:
    obj_val = "infeasible"
elif status == gp.GRB.UNBOUNDED:
    obj_val = "unbounded"
elif status == gp.GRB.OPTIMAL:
    obj_val = model.objVal
time = model.TimeLimit
print("Timecost":,time)
print("Objective Value:", obj_val)
"""

other_llm_prompt = [
"""
You are a mathematical formulator working with a team of optimization experts. The objective is to tackle a complex optimization problem.You are a Template Optimization Assistant. Your task is to modify a template based on the input parameters and return the template in the required format.
""",
"""
You are an expert in problem analysis and can apply previous problem-solving approaches to new issues. The user will provide a specific task description and a thought template. Your goal is to analyze the user's task and generate a specific solution based on the thought template and initial solution. If the instantiated solution involves Python code, only provide the code and let the compiler handle it. If the solution does not involve code, provide a final answer that is easy to extract from the text.
It should be noted that all the python code should be within one code block, the answer should not include more than one code block! And strictly follow the thought-template to instantiate the gurobi code but you should also adjust the input parameter according to the user input!
       
User Input:
{user_input}
Thought template:
{thought_template}
Initial solution:
{initial_solution}
Please analyze the above user task description and thought template, and generate a specific, detailed solution. If the solution involves Python code, only provide the code. If not, provide a clear and extractable final answer.        
"""
]

class Reasoner:
   
    def __init__(self, user_input, api_key=None, llm='deepseek-reasoner', need_check=True,base_url='https://api.deepseek.com/v1/',template_path = 'template.jsonl', tree_path = "tree_mapping.jsonl"):
        self.api_key = api_key
        self.llm = llm
        self.base_url = base_url
        self.client = openai.Client(
        api_key= self.api_key,
        base_url= self.base_url,
        )   
        self.template_path = template_path
        self.tree_path = tree_path
        self.template_matcher = TemplateMatcher(llm=self.llm, client=self.client, template_path=self.template_path, tree_path=self.tree_path)
        self.user_input = user_input
        self.need_check = need_check
              
    def update_input(self,new_input):
        self.user_input = new_input
        
    def template_retrieve(self):
        
        self.problem_type, self.tree_end, self.thought_template = self.template_matcher.final_template(self.user_input)
                
    def reason(self):
        """
        if self.llm == "gpt-4o":
            temp = "gpt-4o"
        else:
            temp = "deepseek-reasoner"
            """
        reasoner = Agent(reason_prompt[0], self.client, self.llm)
        coder = Agent(inspector_prompt, self.client, self.llm)
        coder.clear()
        reasoner.clear()
        input_prompt = reason_prompt[1].format(
            user_input = self.user_input,
            thought_template = self.thought_template
        )

        logging.info("-"*40)
        logging.info("step3: Get the answer! Call a Reasoner!")
        logging.info("-"*40)
        self.result = reasoner.generate_reply(input_prompt)
        #print(self.result)
        self.final_result, code_str = extract_and_execute_code(self.result)
        #logging.info(code_str)
        if self.need_check:
            self.count = 0
            self.inter_input = f"""
            User_input:{self.user_input}
            {code_str}
            {self.final_result}
            """
            
            self.inter_result = self.final_result
            logging.info("-"*40)
            logging.info("step4: Debug the code!")
            logging.info("-"*40)
            inter_code_str = code_str
            while(('An error occurred' in self.inter_result) or (self.inter_result == '') or (self.inter_result == 'None')):
                self.inter_input = f"""
            User_input:{self.user_input}
            {inter_code_str}
            {self.inter_result}
            """

                logging.info(f"The code is error,and the correct step is {self.count}.Call a coder!")
                self.inter_input = coder.generate_reply(self.inter_input)
                #logging.info(self.inter_input)
                self.inter_result, inter_code_str = extract_and_execute_code(self.inter_input)
                self.count = self.count + 1
                if self.count > 3:
                    break
            self.final_result = self.inter_result
            self.code_str = inter_code_str 
            self.stats = get_model_stats_from_code(self.code_str)
            logging.info(self.code_str)
        #print("**** what happen")
        
          
    def reason_run(self):
        start = time.perf_counter()
        self.template_retrieve()
        tem = time.perf_counter()
        self.reason()
        end = time.perf_counter()
        
        return self.problem_type, self.tree_end, self.final_result #, self.stats #, (tem - start), (end - tem)
    
    def dynamic_adjustment(self, solution_step, truth):
        
        self.template_retrieve()
        self.reason()
        logging.info("-"*40)
        logging.info("step 5: template dynamic_adjustment!")
        logging.info("-"*40)
        predict = self.code_str
        update_flag = False
        value = extract_objective(self.final_result)
        accuracy = bool_accuracy(value, truth)
        subtype = None
        if not accuracy:
            update_flag = True
            logging.info(" the accuracy is not right! start correcting!")
            logging.info("-"*40)
            logging.info("step 5.1: formulate problem type and description! Call a formulator!")
            logging.info("-"*40)
            subtype, sub_description = self.template_matcher.formulate_problem(self.user_input, self.problem_type)
            logging.info("-"*40)
            logging.info("step 5.2: get template by the solution step of truth! Call a collector!")
            logging.info("-"*40)
            self.template_matcher.template_collect(self.user_input, subtype, sub_description, solution_step)
            
            logging.info("-"*40)
            logging.info("step 5.3: adjust the tree structure! Call a adjustor!")
            logging.info("-"*40)
            self.template_matcher.tree_adjustment(self.problem_type, subtype, sub_description)
        
        return update_flag, self.problem_type, subtype
                

       
    def direct_reason(self):
        if self.llm == "gpt-4o":
            temp = "o3-mini"
        else:
            temp = "deepseek-reasoner"
        input = "You are an expert in problem analysis and can apply previous problem-solving approaches to new issues.the solution involves Gurobi code, only provide the code.It should be noted that all the python code should be within one code block, the answer should not include more than one code block!"
        reasoner = Agent(reason_prompt[0], self.client, temp)
        reasoner.clear()
        self.result = reasoner.generate_reply(input+self.user_input)
        self.final_result, self.code_str = extract_and_execute_code(self.result)
        self.stats = get_model_stats_from_code(self.code_str)
        logging.info(self.code_str)
        return self.final_result, self.stats
    
   