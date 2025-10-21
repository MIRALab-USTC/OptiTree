# OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling

This is the code of paper OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling. Haoyang Liu, Jie Wang, Yuyang Cai, Xiongwei Han, Yufei Kuang, Jianye HAO. NeurIPS 2025. 


## Dependencies
- **Python**: 3.8.20
- **openai**: 1.99.9
- **Gurobi**: 11.0.3  
- **Hydra**: 1.3.2 
- **NetworkX**: 3.1
- **Flask**: 2.3.3

To build the environment, use the provided Conda environment file:

```bash
conda env create -f environment.yaml
```

## Usage
Go to the root directory `OptiTree`. Put the datasets under the `./data` directory. Below is an illustration of the directory structure.
```
OptiTree/
├── conf/
│   ├──hydra/
        ├──or.yaml          # Configuration files
│   ├── train.yaml          # Training configuration
│   └── test.yaml           # Testing configuration
├── data/                   # Dataset directory
├── test_results/           # Output directory and log files
├── train_results/          # Output directory and log files
├── template_opt.py         # Main optimization logic
├── template_retrieve.py    # Statement thought matching module
├── train.py                # training
├── test.py           # testing
├── utils.py                # Utility functions
├── environment.yaml        # Conda environment file
└── README.md
```
The hyperparameter configurations are in `./conf/`.
The workflow of OptiTree is as following.    

### 1. Building and augmenting OptiTree

Using OR-Instruct as an example.
```bash
python train.py task_name=or_instruct llm=<LLM_name> url=<LLM_url> api_key=<your_api_key> path='data/or_instruct.jsonl'  template_path='tree_map/ds_tree.jsonl' tree_path='tree_map/ds_tree_mapping.jsonl'
```


### 2. Evaluation with OptiTree

Using Mamo ComplexLP as an example.
```bash
python test.py task_name=mamo_complex_lp llm=<LLM_name> url=<LLM_url> api_key=<your_api_key> path='data/mamo_complex_lp.jsonl' template_path='tree_map/ds_tree.jsonl' tree_path='tree_map/ds_tree_mapping.jsonl'
```

## Citation

If you find this code useful, please consider citing the following papers.

```bibtex
@inproceedings{optitree2025,
  title={OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling},
  author={Haoyang Liu and Jie Wang and Yuyang Cai and Xiongwei Han and Yufei Kuang and Jianye HAO},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={}
}
```