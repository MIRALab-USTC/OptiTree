from transformers import AutoModel
import json
import os
import pickle
import numpy as np
from functools import lru_cache
import argparse
import json_repair
from utils import Agent,  extract_json, replace_fraction

import openai
import logging
prompt_template = [
"""
You are a mathematical formulator working with a team of optimization experts. The objective is to tackle a complex optimization problem.
""",
"""
You are given a specific problem and its current basic problem type. You are also provided with a list of subtypes for this basic problem type.
Input problem: {input_problem}.
its current basic problem type:{basic_type}

You are given a list of defined subtypes and the description of subtypes: {template_info}.

Your task is to determine if the specific problem belongs to one of the given subtypes.
If it does, return the subtype directly.Importantly, you must return the only subtype verbatim(don't return the description) which is provided in the list.
If it does not, return "subtype not find".

Also, return a boolean value indicating whether the input problem belongs to the defined subtypes.



Return only the final answer in the following JSON format:
```json
{{
    "matching_subtype": "<matching_subtype>",
    "reasoning": "<reasoning_process>"
    "belongs_to_subtypes": <boolean_value>
}}
```
- Note that I'm going to use python json.loads() function to parse the json file, so please make sure the format is correct (don't add ',' before enclosing '}}' or ']' characters.
- Generate the complete json file and don't omit anything.
- Use '```json' and '```' to enclose the json file.
""",
"""
You are given a specific problem . You are also provided with a list of problem_types 

- **Specific Problem**: {specific_problem}
You are given a list of defined problem_type: {list_of_problem_types}.

Your task is to determine if the specific problem belongs to one of the given problem_types.
If it does, return the problem_type directly.Importantly, you must return the problem_type which is provided in the list.
If it does not, return "problem_type not find".


Also, return a boolean value indicating whether the input problem belongs to the defined problem_types.

Return only the final answer in the following JSON format:
```json
{{
    "matching_problem_type": "<matching_probelm_type>",
    "reasoning": "<reasoning_process>"
    "belongs_to_problem_types": <boolean_value>
}}
```
- Note that I'm going to use python json.loads() function to parse the json file, so please make sure the format is correct (don't add ',' before enclosing '}}' or ']' characters.
- Generate the complete json file and don't omit anything.
- Use '```json' and '```' to enclose the json file.
"""

]

formulate_problem = [
"""
You are a mathematical formulator working with a team of optimization experts. The objective is to tackle a complex optimization problem.
""",
"""
You are given a specific problem, its current basic problem type, and the description of the basic problem type.

Your task is to determine the more specific subtype of the given basic problem type that the specific problem belongs to, and provide a more detailed description of this subtype.

- **Specific Problem**: {specific_problem}
- **Current Basic Problem Type**: {current_basic_problem_type}
- **Description of Basic Problem Type**: {description_of_basic_problem_type}

Please provide the following information in JSON format:

```json
{{
    "current_basic_problem_type": "{current_basic_problem_type}",
    "description_of_basic_problem_type": {description_of_basic_problem_type},
    "formulated_subtype": "<subtype>",
    "description_of_subtype":{{
    "description": "<the more precise of description>",
    "constraints": 
    {{
      "<get the Constraint 1>": "Detailed description of constraint 1",
      "<get the Constraint 2>": "Detailed description of constraint 2"
    }}
}}
}}
```
""",
"""
You are given a specific combinatorial optimization problem.

Your task is to summarize the industrial scene type of this specific problem and provide a detailed description of this type.

Importantly, you must classify the problem into more precise industrial scenarios, such as the Traveling Salesman Problem (TSP) , facility location problem, Parallel Machine Scheduling and so on,.Avoid using broad categories such as linear programming, mixed-integer optimization, integer optimization, Integer Linear Programming Problem and so on.

- **Specific Problem**: {specific_problem}

Please provide the following information in JSON format:

```json
{{
    "industrial_scene_type": "<industrial_scene_type>",
    "description_of_type":{{
    "description": "<the more precise of description>",
    "constraints": 
    {{
      "<get the Constraint 1>": "Detailed description of constraint 1",
      "<get the Constraint 2>": "Detailed description of constraint 2"
    }}
    }}
}}
```
Here is an output example:
{{
    'industrial_scene_type': 'Maximum Flow Problem',
    'description_of_type': {{
    'description': 'The Maximum Flow Problem involves determining the highest possible flow that can be routed through a directed graph from a specified source node to a sink node, while adhering to the capacity limits of the edges. This problem is foundational in network flow theory and has applications in transportation networks, communication systems, supply chain logistics, and resource distribution. The solution must respect edge capacities, flow directionality, and conservation laws at intermediate nodes.', 
    'constraints': {{
        'Directed Graph': 'Flow can only travel in the direction specified by the edges in the graph.', 
        'Capacity Constraints': "The flow on each edge must be non-negative and cannot exceed the edge's maximum capacity.", 
        'Flow Conservation': 'For every node except the source and sink, the total incoming flow must equal the total outgoing flow.'
    }}, 
}}
}}
"""

]

collect_template =[
"""
You are a mathematical formulator working with a team of optimization experts. The objective is to tackle a complex optimization problem.
""", 
"""
Please list the step to formulate a {problem_type} problem and use the gurobi code to solve it. You need to record some errors that easy to make during the formulation process. Please output a json format.

You are given a specific combinatorial optimization problem, its solution process, and the problem type along with its description.


- **Problem Type**: {problem_type}
- **Description of Problem Type**: {description}
- **Specific Problem**: {specific_problem}
- **Solution step of Specific Problem**: {solution_step}

Your task is to return a template for this problem type, which includes the following five parts:

1. **problem_type**: The provided problem type.
2. **description**: The provided description of the problem type.
3. **reason_flow**: A detailed step-by-step reasoning process for solving a series of prolems which belong to problem type, according to the provided solution process of the specific problem.
4. **example_application**: A detailed example application that matches the specific problem and its solution process.
5. **increment**: a list.


Additonanlly, in the solution steps, Gurobi code is included. You must only use the fixed Gurobi code mentioned below in the solution steps. This is the fixed Gurobi code ————"### Gurobi Code:\n```python\nimport json\nimport numpy as np\nimport math\nimport gurobipy as gp\nfrom gurobipy import GRB\n\n# Create a new model\nmodel = gp.Model('model')\n\n# define parameters\n\n# define variables\n\n# define constraints\n\n# define objective \n\n# Optimize the model\nmodel.optimize()\nstatus = model.status\n\nobj_val = None\n# Check whether the model is infeasible, has infinite solutions, or has an optimal solution\nif status == gp.GRB.INFEASIBLE:\n    obj_val = \"infeasible\"\nelif status == gp.GRB.UNBOUNDED:\n    obj_val = \"unbounded\"\nelif status == gp.GRB.OPTIMAL:\n    obj_val = model.objVal\ntime = model.TimeLimit\nprint(\"Timecost\":,time)\nprint(\"Objective Value:\", obj_val)\n```"
Please provide the following information in JSON format:

Here is an template example:

{{
    "problem_type": "Travelling Sales Person Problem",
    'description': {{
        'description': "The Transportation Problem involves optimizing the shipment of goods from multiple distribution centers to various destinations to minimize total transportation costs while meeting all destination demands. In this scenario, a company with four distribution centers (A, B, C, D) must supply five destinations (1, 2, 3, 4, 5) such that each destination's demand is fully satisfied. The key objective is to determine the optimal shipment quantities from each center to each destination that result in the lowest possible transportation costs. This problem is a linear programming model where decision variables represent the amount shipped from each center to each destination, subject to supply and demand limitations.", 
        'constraints': {{'Supply Constraints': 'The total quantity of goods transported from each distribution center to all destinations must not exceed the available supply at that center.', 'Demand Constraints': "The total quantity of goods received by each destination from all distribution centers must exactly match the destination's specified demand.", 'Non-Negative Transportation': 'The amount of goods transported from any distribution center to a destination must be non-negative (i.e., shipments cannot have negative quantities).'}}
    }}
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
            "### Gurobi Code:\n```python\nimport json\nimport numpy as np\nimport math\nimport gurobipy as gp\nfrom gurobipy import GRB\n\n# Create a new model\nmodel = gp.Model('model')\n\n# define parameters\n\n# define variables\n\n# define constraints\n\n# define objective \n\n# Optimize the model\nmodel.optimize()\nstatus = model.status\n\nobj_val = None\n# Check whether the model is infeasible, has infinite solutions, or has an optimal solution\nif status == gp.GRB.INFEASIBLE:\n    obj_val = \"infeasible\"\nelif status == gp.GRB.UNBOUNDED:\n    obj_val = \"unbounded\"\nelif status == gp.GRB.OPTIMAL:\n    obj_val = model.objVal\ntime = model.TimeLimit\nprint(\"Timecost\":,time)\nprint(\"Objective Value:\", obj_val)\n```"，
            "### Common Errors to Avoid:\n1. **Incorrect Subtour Elimination**: Ensure MTZ constraints exclude the starting city and are applied to correct indices.\n2. **Indexing Mistakes**: Use consistent 0-based or 1-based indexing for cities.\n3. **Self-Loops**: Explicitly disable \\( x_{{ii}} \\) variables.\n4. **Bounds on MTZ Variables**: Set \\( u_i \\) bounds correctly (\\( 1 \\leq u_i \\leq n-1 \\)).\n5. **Objective Function**: Ensure distances are correctly paired with \\( x_{{ij}} \\) and exclude \\( i = j \\).\n\nBy following these steps and avoiding common pitfalls, you can effectively model and solve TSP using Gurobi."
        ],
    }},
    "increment":[]
}}

Important:
- Use plain JSON without markdown syntax
- Ensure all quotes are properly escaped
- Include all required keys: problem_type, description, reason_flow, example_application,increment
"""
]

tree_construct =[
"""
 You are a mathematical formulator working with a team of optimization experts. The objective is to tackle a complex optimization problem.
""" ,
 """
You are given a primary problem type and its description. You are also provided with a list of other problem types, each with its own description .

According description, your task is to determine which problem types in the list are subtypes of the given primary problem type. Return a list of problem types that are identified as subtypes of the primary problem type.To be more specific, The subtype contains the constraint form of the primary problem type.
Importantly, the returned subtypes must be provided in the list, if not have any problem type which is subtype of the given primary problem type in the list, return an empty list.

- **Primary Problem Type**: {primary_problem_type}
- **Description of Primary Problem Type**: {description_type}
- **List of Problem Types**: {list_of_problem_types}

Please provide the following information in JSON format:

```json
{{
    "primary_problem_type": "{primary_problem_type}",
    "matching_subtypes": ["<problem_type>"]

}}
"""]

'''
class TemplateMatcher:
    def __init__(self, llm, client,
                 template_path='template.jsonl'):
        self.template_path = template_path
        self.template_dict = self.load_json_file()
        self.client = client
        self.model = llm
    

    def load_json_file(self):
        data_dict = {}
        id = 0
        for line in (open(self.template_path)):
            id+=1
            #logging.info(id)    
            data = json.loads(line)
            data_dict[data["problem"]] = data["template"]
            """
            try:
                data = json.loads(line)
                data_dict[data["problem"]] = data["template"]
            except json.JSONDecodeError as e:
                logging.info(f"Error decoding JSON: {e}")
                #logging.info(f"Offending line: {line.strip()}")
            """
        return data_dict

    """       
    def get_template_by_labels(self, chapter, section, method):

        

        for item in self.template_dict[chapter][section]:
            if item['template_name'] == method:
                return {
                    'chapter': chapter,
                    'section': section,
                    'method': item
                    }

        return None
    """
    def get_template_by_labels(self, interpret):
       
        if interpret in self.template_dict: 
            return self.template_dict[interpret]
        
        return None
    
    def template_reduce(self, problem):
        interpreter = Agent(prompt_template[0], self.client, self.model)
        interpreter.clear()
        logging.info("-"*40)
        logging.info("step1: Get the problem type! Call an Interpreter!")
        logging.info("-"*40)
        prompt = prompt_template[1]
        
        prompt = prompt.format(
                    problem_types = ",".join(self.template_dict.keys()),
                    input_problem = problem
                )
        #logging.info(prompt)
       
        answer = interpreter.generate_reply(prompt)
        logging.info(answer)
        

        inpertretation = extract_json(answer)
        problem_type = inpertretation["problem_type"]
        reasoning = inpertretation["reasoning"]
        belongs = inpertretation["belongs_to_defined_types"]
        """
        structure = json.dumps(
            {
                'specific_type': inpertretation['specific_type'],
                
            }, 
            indent=4
        )
        """
        return problem_type, reasoning, belongs

    def final_template(self, problem):
        problem_type, reasoning, belongs = self.template_reduce(problem)
        if belongs:
            template=self.get_template_by_labels(problem_type)
        else: 
            #logging.info(type(problem_type))
            template = self.template_collect(problem_type)
        logging.info("-"*40)
        logging.info("step2: Retrieve the template!")
        logging.info("-"*40)
        logging.info(template)         
        return problem_type, belongs, template
    
    def template_collect(self, problem_type):
        collector = Agent(collect_template[0], self.client, self.model)
        collector.clear()
        #logging.info("Call a Collector!")
        prompt = collect_template[1]
        #logging.info(problem_type)
        prompt = prompt.format(input_problem = problem_type)
        #logging.info(prompt)
        #logging.info('-------------------------------------------')
        answer = collector.generate_reply(prompt)
        #logging.info(answer)
        #logging.info('-------------------------------------------')
        return answer
    
    def matcher_update(self):
        path = self.template_path
        self.template_dict = self.load_json_file()


'''

class TemplateMatcher:
    def __init__(self, client, llm,
                 template_path='tree_update.jsonl', tree_path = "tree_mapping.jsonl"):
        self.template_path = template_path
        self.tree_path = tree_path
        self.template_dict = self.load_template()
        self.tree = self.load_tree()
        self.client = client
        self.model = llm
    

    def load_template(self):
        data_dict = {}
        for line in (open(self.template_path)):
            data :dict= json.loads(line)
            #logging.info(data)
            id = data.pop("id")
            #data.pop("reason_flow")
            #data.pop("example_application")
            data_dict[id] = data
        return data_dict
    
    def load_tree(self):
        with open(self.tree_path, "r", encoding="utf-8") as file:
            json_content = file.read()

            tree_dict = json_repair.loads(json_content)
            #tree_dict = json.loads(file)
            #logging.info(tree_dict)
        return tree_dict
    """       
    def get_template_by_labels(self, chapter, section, method):

        

        for item in self.template_dict[chapter][section]:
            if item['template_name'] == method:
                return {
                    'chapter': chapter,
                    'section': section,
                    'method': item
                    }

        return None
    """
    """
    def get_template_by_labels(self, interpret):
       
        if interpret in self.template_dict: 
            return self.template_dict[interpret]
        
        return None
    """
    
    def get_template_id(self, problem_type):
        for key, template in self.template_dict.items():
            if template["problem_type"] == problem_type:
                
                return key
            
    def template_reduce(self, problem, basic_type, sublist):
        interpreter = Agent(prompt_template[0], self.client, self.model)
        interpreter.clear()
        if basic_type == "Mixed Integer Programming":
            input = prompt_template[2].format(
                specific_problem = problem,
                list_of_problem_types = sublist
            )
            answer = interpreter.generate_reply(input)
            logging.info(answer)
            inpertretation = json_repair.loads(answer)
            
            reasoning = ""#inpertretation["reasoning"]
            belongs = inpertretation["belongs_to_problem_types"]
            subtype = inpertretation["matching_problem_type"]
        else:
            prompt = prompt_template[1]
            prompt = prompt.format(
                    basic_type = basic_type,
                    template_info =  sublist, 
                    input_problem = problem
                )
            answer = interpreter.generate_reply(prompt)
            logging.info(answer)
            inpertretation = json_repair.loads(answer)
            reasoning = ""#inpertretation["reasoning"]
            belongs = inpertretation["belongs_to_subtypes"]
            subtype = inpertretation["matching_subtype"]

        return  subtype, reasoning, belongs

    def final_template(self, problem):
        #print(f'tree len:{len(self.tree)}')
        current_type = self.template_dict[0]["problem_type"]
        list = [self.template_dict[id]["problem_type"] for id in self.tree[str(0)]]
        logging.info("-"*40)
        logging.info("step1: Get the problem type! Call an Interpreter!")
        logging.info("-"*40)
        #interpreter = Agent(prompt_template[0], self.client, self.model)
        #interpreter.clear()
        subtype, reasoning, belongs = self.template_reduce(problem, current_type, list)
        #end = False
        
        id  = 1
        while belongs:
            id += 1
            
            current_type = subtype
            #if id > 3:
            #   break
            index = self.get_template_id(current_type)
            logging.info(f'index: {index}')
            if len(self.tree[str(index)]) == 0:
                break
            else:
                template_info = [{"problem_type":self.template_dict[id]["problem_type"], "description":self.template_dict[id]["description"]} for id in self.tree[str(index)]]
                subtype, reasoning, belongs = self.template_reduce(problem, current_type, template_info)
        """
        if belongs:
            id = self.get_template_id(problem_type)
            template=self.template_dict[id]
        else: 
            template = self.template_collect(problem_type)
        
        logging.info(template)"""
        
        logging.info("-"*40)
        logging.info("step2: Retrieve the template!")
        logging.info("-"*40)
        
        id = self.get_template_id(current_type)
        template=self.template_dict[id]
        logging.info(template)
        end = False
        if len(self.tree[str(id)]) == 0:
                end = True     
        return current_type, end, template
    
    def formulate_problem(self, problem, current_type):
        id = self.get_template_id(current_type)
        description = self.template_dict[id]["description"]
        formulator = Agent(formulate_problem[0],self.client, self.model)
        formulator.clear()
        if id != 0:
            input  = formulate_problem[1].format(
            current_basic_problem_type = current_type, 
            description_of_basic_problem_type = description,
            specific_problem = problem
        )
            answer = formulator.generate_reply(input)
            logging.info(answer)
            answer = json_repair.loads(answer) 
            problem_type = answer["formulated_subtype"]
            description_type = answer["description_of_subtype"]
        else:
            input  = formulate_problem[2].format(
            specific_problem = problem
        )
            answer = formulator.generate_reply(input)
            logging.info(answer)
            answer = json_repair.loads(answer) 
            problem_type = answer['industrial_scene_type']
            description_type = answer['description_of_type']
            
        
        return problem_type, description_type
    
    def template_collect(self, problem, problem_type, problem_decription, solution_step):
        collector = Agent(collect_template[0], self.client, self.model)
        collector.clear()

        input = collect_template[1].format(
            specific_problem = problem,
            problem_type = problem_type,
            description = problem_decription,
            solution_step = solution_step
        )
        answer = collector.generate_reply(input)
        template = json_repair.loads(answer)
        id = len(self.template_dict)
        self.template_dict[id] = template
        
        save_template = {"id":id, **template}
        logging.info(save_template)
        with open(self.template_path, "a+", encoding='utf-8') as file:
            json_str = json.dumps(save_template)
            file.write(json_str + '\n')
            
    def tree_adjustment(self, father_type, sub_type, sub_description):
        father_id = self.get_template_id(father_type)
        sub_id = len(self.tree)
        
        logging.info(f"brothers: {self.tree[str(father_id)]}")
        
        self.tree[str(sub_id)] = []
        
        if len(self.tree[str(father_id)]) != 0 and father_id != 0:
            
            template_info = [{"problem_type":self.template_dict[id]["problem_type"], "description":self.template_dict[id]["description"]} for id in self.tree[str(father_id)]]
            constructor = Agent(tree_construct[0],self.client, self.model)
            input = tree_construct[1].format(
                primary_problem_type = sub_type,
                description_type = sub_description,
                list_of_problem_types = template_info
                
            )
            answer = constructor.generate_reply(input)
            logging.info(answer)
            answer = json_repair.loads(answer)
            sonlist :list = answer["matching_subtypes"]
            if len(sonlist) != 0:
                son_id_list = [self.get_template_id(type)  for type in sonlist]
                self.tree[str(father_id)] = list(set(self.tree[str(father_id)]) - set(son_id_list))
                self.tree[str(sub_id)] = son_id_list
        
        self.tree[str(father_id)].append(sub_id)
        logging.info(f"update borthers:{self.tree[str(father_id)]}")
        logging.info(f"sons: {self.tree[str(sub_id)]}")
        
        new_tree = self.tree
        with open(self.tree_path, 'w', encoding='utf-8') as file:
                json_str = json.dumps(new_tree)
                file.write(json_str)
        
    """
    def matcher_update(self):
        #path = self.template_path
        self.template_dict = self.load_template()
        self.tree = self.load_tree()
    """

    
if __name__ == "__main__":
    
    template_path = "tree.jsonl"
    client = openai.Client(
        api_key= "sk-28c30dbc4b964513bd458e6a7270977d",
        base_url= "https://api.deepseek.com",
    )
    extractor = TemplateMatcher(client, "deepseek-chat")
    problem = "The capacitated warehouse location problem involves determining the optimal locations for a set number of warehouses to service customers at minimum cost, taking into account warehouse capacities, operating costs, and customer demand. The capacitated warehouse location problem is the problem of locating NumberOfLocations warehouses which have to service NumberOfCustomers customers, at minimum cost. Each customer has an associated demand CustomerDemand. There are constraints on the total demand that can be met from a warehouse, as specified by WarehouseCapacity. Costs are incurred when allocating service to customers from warehouses ServiceAllocationCost, and warehouses have a fixed operating cost WarehouseFixedCost. Additionally, there is a lower limit MinimumDemandFromWarehouse on the amount of demand that a warehouse must meet if it is opened, as well as constraints on the minimum MinimumOpenWarehouses and maximum MaximumOpenWarehouses number of warehouses that can be operational. The total number of potential warehouse locations is 10. The total number of customers to be serviced is 20. The demand of each customer is [117, 86, 69, 53, 110, 74, 136, 140, 126, 79, 54, 86, 114, 76, 136, 73, 144, 51, 53, 120]. The cost of allocating service from each warehouse to each customer is [[80, 94, 44, 51, 190, 44, 129, 178, 129, 91, 172, 119, 177, 150, 90, 51, 53, 97, 184, 87], [139, 33, 104, 135, 50, 176, 97, 121, 47, 29, 186, 163, 149, 108, 156, 169, 100, 160, 153, 85], [153, 36, 18, 170, 18, 181, 178, 68, 171, 106, 159, 110, 21, 106, 91, 29, 144, 140, 155, 116], [103, 59, 78, 125, 14, 11, 152, 95, 76, 173, 36, 148, 75, 132, 59, 153, 113, 74, 185, 71], [193, 186, 130, 145, 114, 150, 33, 154, 20, 75, 103, 30, 137, 131, 167, 32, 53, 150, 176, 166], [159, 130, 156, 65, 36, 59, 199, 124, 104, 72, 180, 73, 43, 152, 143, 90, 161, 65, 172, 141], [173, 121, 110, 127, 22, 159, 195, 137, 47, 10, 87, 11, 154, 66, 126, 60, 152, 54, 20, 25], [181, 34, 186, 152, 109, 195, 133, 198, 30, 65, 69, 19, 109, 143, 108, 196, 59, 133, 10, 123], [82, 113, 147, 21, 88, 24, 38, 16, 70, 122, 148, 192, 116, 108, 18, 20, 143, 18, 116, 142], [176, 170, 87, 91, 195, 183, 124, 89, 72, 97, 89, 23, 45, 196, 97, 27, 83, 81, 171, 148]]. The total capacity for each warehouse is [3010, 2910, 4530, 4720, 4920, 3750, 4930, 2970, 3310, 2460]. The lower limit on the demand that must be met from a warehouse if it is to be operational is [64, 55, 27, 71, 93, 90, 89, 87, 43, 50]. The minimum number of warehouses that need to be operational is 3. The maximum number of warehouses that can be operational is 8. The fixed operating cost of each warehouse is [8517, 5068, 9433, 6127, 6033, 5966, 7762, 9406, 6602, 7040].  Each customer demand must be met. Each warehouse can meet a maximum demand equal to its WarehouseCapacity. If a warehouse is open, it must meet at least the MinimumDemandFromWarehouse. At least MinimumOpenWarehouses warehouses must be operational. At most MaximumOpenWarehouses warehouses can be operational.  Minimize the total cost of servicing customers, including service allocation and operating costs of warehouses. "
    problem_type, end, template = extractor.final_template(problem)
    logging.info(template)