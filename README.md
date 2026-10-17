# Task 5: Auto Tagging Support Tickets Using LLM

## Objective
Automatically tag support tickets into categories using a large language model (LLM). Compare zero-shot classification performance with few-shot prompting techniques, outputting the top 3 most probable tags per ticket.

## Methodology / Approach
Instead of relying on manual CSV downloads, this code automatically loads the professional `bitext/customer-support-intent-dataset` directly from the HuggingFace Hub. This dataset contains realistic free-text support queries and their actual intent classifications.

The script demonstrates two distinct LLM approaches:
1. **Zero-Shot Classification**: 
 - Uses the robust `facebook/bart-large-mnli` model.
 - Evaluates a text against a list of candidate tags without *any* prior training or examples.
 - Outputs the probabilities of the top 3 most likely tags.
2. **Few-Shot Prompt Engineering**: 
 - Uses a generative LLM (`google/flan-t5-small`).
 - The prompt provides the LLM with a specific task definition and two examples (few-shot) to guide its generation behavior.
 - Demonstrates how context steering improves formatting and accuracy for specific labels.
 
*(Note: For fine-tuning performance, the architecture used in Task 1 with `Trainer` and a classification head would be identical. Zero/Few-shot are presented here as they represent modern generative LLM capabilities).*

## Key Results
- The zero-shot pipeline is highly capable of assigning probabilities across a large multi-class array without training.
- Few-shot prompting effectively forces the LLM to adhere to a strict set of business logic tags.

## How to Run

### 1. Install Dependencies
```bash
pip install datasets transformers torch
```

### 2. Execute the Script
```bash
python task5_ticket_tagging.py
```
*Note: The first time you run this, it will download the open-source NLP models. Subsequent runs will be much faster.*
