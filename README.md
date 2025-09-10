# LMDB: An Open Access Dataset of Experimental Parameters for Liver Manufacturing Extracted from Scientific Literature

This repository contains the code and data for the LMDB dataset, which extracts experimental parameters for liver manufacturing from scientific literature.

## Features

1. Literature screening. We collected relevant research literature from the Web of Science and PubMed databases using expert-curated search terms. The initial paper corpus was refined using GPT-4o-mini, a proprietary LLM with powerful summarization and classification capabilities, to exclude irrelevant publications.
2. Candidate Generation. Identify possible entity sets from literature sentences to serve as candidates for the next stage. We compared three models: GPT-4o-mini, GLM-4-Chat and LLaMA3-8B-Instruct. The latter two models were fine-tuned on the training set.
3. Template Completion. Experimental parameters were first identified from sentences to build an entity repository capturing all possible values for each entity, which was then mapped to a predefined JSON schema and generate structured experimental recordsWe compared our structured  pipeline with a direct extraction approach using GPT-4o-mini.

## Project Structure

```plaintext
.
├── Candidate Generation                   # Candidate Generation Module
│   ├── data                                # Datasets
│   │   ├── eval_data                       # Evaluation Data
│   │   │   ├── glm_eval_fix.json
│   │   │   ├── gpt_eval_fix.json
│   │   │   └── llama_eval_fix.json
│   │   ├── inference_data                  # Inference Results
│   │   │   ├── final_res_part_1.jsonl        # GLM Model Inference Results on Literature Collection - Part 1
│   │   │   ├── final_res_part_2.jsonl        # GLM Model Inference Results on Literature Collection - Part 2
│   │   │   ├── glm_infer.json                # Test Set Inference Results
│   │   │   ├── gpt_infer.json
│   │   │   └── llama_infer.json
│   │   ├── seed_data                       # Seed Data (Pre-training/Initialization)
│   │   │   ├── rag.json
│   │   │   ├── rag.pkl
│   │   │   ├── sen.json
│   │   │   ├── test.json
│   │   │   └── train.json
│   │   └── train_data                      # Training Data
│   │       └── train.json
│   ├── encode_by_bioBERT.py                # Encode Retrieval Dataset using BioBERT
│   ├── evaluate
│   │   └── eval.py                         # Model Evaluation Script
│   ├── inference                           # Inference Module
│   │   ├── batch_inference_glm.py        
│   │   ├── batch_inference_gpt.py        
│   │   ├── batch_inference_llama.py      
│   │   └── utils.py                      
│   ├── prompt.py                         
│   └── train                               # Training Module
│       ├── convert_data_format.py          # Data Format Conversion
│       ├── deepspeed_zero_stage2_config.json # Deepspeed Configuration
│       ├── glm4_lora_sft.yaml              # GLM LoRA SFT Configuration
│       ├── llama3_lora_sft.yaml            # LLaMA LoRA SFT Configuration
│       ├── sft.py                          # Standard SFT Script
│       ├── swanlog                         # Training Logs
│       │   └── build
│       ├── test.py                         # Testing Script
│
├── Direct Extraction                      # Direct Extraction Baseline Method
│   ├── inference.py                      
│   ├── test_data.json                    
│   └── utils.py                          
│
├── Template Completion                    # Template Completion Module
│   ├── data                                # Data and Mapping Tables
│   │   ├── final_data.json
│   │   ├── mapping table                   # Entity Mapping Tables
│   │   │   ├── Cell type.xlsx
│   │   │   ├── Chip material.xlsx
│   │   │   ├── Cross-linking agent.xlsx
│   │   │   ├── Manufacturing method.xlsx
│   │   │   └── Material in cell culture.xlsx
│   │   ├── numeric.json                    # Numerical Entity Data
│   │   ├── numeric.py                      # Numerical Processing Script
│   │   ├── test_data.json                  # Test Data
│   │   └── test_data.xlsx                  # Test Data (Excel)
│   ├── eval.py                             # Evaluation Script
│   ├── infer.bash                          # Our Pipeline Script
│   ├── infer_summary.bash                  # Literature Screening Summary Script
│   ├── prompt.py                         
│   ├── relation_extraction.py              # Relation Extraction Script
│   ├── summary.py                          # Summary Script
│   └── utils.py                          
│
├── README.md
```

## Usage

* The repository code requires transformers>=4.44.0.
* Before using, please update the file paths to match your local directories and provide your api_key
