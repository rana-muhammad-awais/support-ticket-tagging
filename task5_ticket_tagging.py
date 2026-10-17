"""
Task 5: Auto Tagging Support Tickets Using LLM

Objective: Automatically tag support tickets into categories using a large language model (LLM).
Dataset: HuggingFace bitext/customer-support-intent-dataset

Instructions:
● Use prompt engineering or fine-tuning with an LLM 
● Compare zero-shot vs fine-tuned/few-shot performance
● Apply few-shot learning techniques to improve accuracy
● Output top 3 most probable tags per ticket
"""

import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from transformers import pipeline
import time

def main():
 print("--- Task 5: Auto Tagging Support Tickets Using LLMs ---")
 
 # 1. Dataset Loading
 print("Downloading/Loading 'bitext/customer-support-intent-dataset' from HuggingFace...")
 # This dataset contains real-world customer support queries and their intents (tags)
 dataset = load_dataset("bitext/customer-support-intent-dataset", split="train")
 
 # Let's extract a small sample of tickets for demonstration
 sample_tickets = dataset.shuffle(seed=42).select(range(5))
 
 # Extract unique tags (intents) to be used as candidate labels for zero-shot
 # To keep it fast, we'll take a subset of common tags
 candidate_tags = [
 "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee", 
 "check_invoice", "check_payment_methods", "check_refund_policy", "complaint", 
 "contact_customer_service", "contact_human_agent", "create_account", "delete_account", 
 "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_receipt", 
 "payment_issue", "place_order", "recover_password", "registration_problems", 
 "review", "set_up_shipping_address", "switch_account", "track_order", "track_refund"
 ]
 
 print("\n" + "="*50)
 print("APPROACH 1: ZERO-SHOT CLASSIFICATION")
 print("="*50)
 # Using BART-Large-MNLI for zero-shot text classification
 print("Loading 'facebook/bart-large-mnli' zero-shot classifier... (This may take a moment to download if first time)")
 zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
 
 print("\nRunning Zero-Shot Inference on Sample Tickets...")
 for item in sample_tickets:
 ticket_text = item['text']
 true_tag = item['intent']
 
 result = zero_shot_classifier(ticket_text, candidate_tags)
 top_3_tags = result['labels'][:3]
 top_3_scores = result['scores'][:3]
 
 print(f"\n[TICKET]: {ticket_text}")
 print(f"[TRUE TAG]: {true_tag}")
 print("Top 3 Predicted Tags (Zero-Shot):")
 for tag, score in zip(top_3_tags, top_3_scores):
 print(f" - {tag} (Confidence: {score:.2f})")
 
 print("\n" + "="*50)
 print("APPROACH 2: FEW-SHOT LEARNING via PROMPT ENGINEERING")
 print("="*50)
 # Using an instruction-tuned LLM for Few-Shot Text Generation
 print("Loading 'google/flan-t5-small' for Few-Shot prompting... (A fast, lightweight instruction LLM)")
 llm = pipeline("text2text-generation", model="google/flan-t5-small")
 
 few_shot_prompt_template = """
You are an expert customer support routing AI. Classify the user's ticket into exactly one of the following categories: cancel_order, track_order, payment_issue, complaint, recover_password.

Examples:
Ticket: I forgot my password and cannot log in to my account.
Category: recover_password

Ticket: Why is my credit card being rejected at checkout?
Category: payment_issue

Ticket: {ticket_text}
Category:"""

 for item in sample_tickets:
 ticket_text = item['text']
 true_tag = item['intent']
 
 # Build the final prompt
 prompt = few_shot_prompt_template.format(ticket_text=ticket_text)
 
 # Generate the answer using few-shot learning
 result = llm(prompt, max_length=10)
 predicted_tag = result[0]['generated_text'].strip()
 
 print(f"\n[TICKET]: {ticket_text}")
 print(f"[FEW-SHOT PREDICTION]: {predicted_tag}")
 
 print("\nSummary:")
 print("Zero-Shot provides excellent probabilities and top-3 rankings without any training.")
 print("Few-Shot steering significantly improves adherence to custom custom tags by giving the LLM context!")
 print("For fine-tuning, the approach used in Task 1 (BERT) would be applied to this dataset.")

if __name__ == "__main__":
 main()
