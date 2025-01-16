#**Data preprocessing of the data**


import pandas as pd

# Load the uploaded CSV file to review its structure and content
file_path = '/content/unique_diseases_data.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print(data.head())

import ast
import re

# Function to clean and normalize symptom text
def clean_symptom_text(symptom):
    # Remove punctuation and special characters, and convert to lowercase
    symptom = re.sub(r'[^\w\s]', '', symptom).lower()
    # Trim extra spaces
    symptom = symptom.strip()
    return symptom

# Create a long-format dataframe with individual symptoms
long_format_data = []

for index, row in data.iterrows():
    disease_name = row['Disease Name']
    symptoms_raw = row['Symptoms']

    # Convert symptom strings to lists
    try:
        symptoms = ast.literal_eval(symptoms_raw)  # Safely evaluate the string as a list
    except:
        symptoms = []  # Handle cases where parsing fails

    # Process each symptom in the list
    for symptom in symptoms:
        cleaned_symptom = clean_symptom_text(symptom)
        if cleaned_symptom:  # Only keep non-empty strings
            long_format_data.append({'Disease Name': disease_name, 'Symptom': cleaned_symptom})

# Convert long-format data to a DataFrame
long_format_df = pd.DataFrame(long_format_data)

print(long_format_df)

# Import the necessary library
import pandas as pd

# Assuming 'long_format_df' is your DataFrame
long_format_df.to_excel('disease_symptoms.xlsx', index=False)

"""#**Llama Model for Data Structuring**

Structuring Objectives:

1. Convert descriptions of the symtoms to simplified and synonymous terms or words.

"""

pip install --upgrade transformers

!huggingface-cli login

"""##Model Testing"""

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Llama model
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Function to simplify symptoms
def simplify_symptom(description):
    """
    Simplifies a verbose symptom description using a causal language model.
    """
    prompt = f"Provide a medical term for the following symptom: {description}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=15,  # Enforce short output
        temperature=0.5,    # Reduce randomness for more deterministic output
        top_p=0.8,          # Restrict output to higher-probability terms
        eos_token_id=tokenizer.eos_token_id
    )
    # Decode and clean the output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt from output (if it appears)
    if ":" in result:
        result = result.split(":")[1].strip()
    return result

# Sample test symptoms
test_symptoms = [
    "tummy or back pain",
    "a pulsing feeling in your tummy",
    "bringing up milk or being sick during or shortly after feeding",
    "coughing or hiccupping when feeding",
    "the main symptom of acanthosis nigricans is patches of skin that are darker and thicker than usual"
]

# Test the model
print("Testing the Model with Refined Prompt and Parameters:\n")
for symptom in test_symptoms:
    simplified = simplify_symptom(symptom)
    print(f"Original: {symptom}")
    print(f"Simplified: {simplified}\n")

"""##Few Shot Inference and Parameter Fine-Tuning"""

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Llama model
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Few-shot examples
few_shot_examples = """
Convert the following symptom descriptions into concise medical terms:
- Symptom: tummy or back pain
  Simplified: abdomen pain, pain

- Symptom: a pulsing feeling in your tummy
  Simplified: pulsation

- Symptom: bringing up milk or being sick during or shortly after feeding
  Simplified: reflux

- Symptom: coughing or hiccupping when feeding
  Simplified: feeding difficulties

- Symptom: the main symptom of acanthosis nigricans is patches of skin that are darker and thicker than usual
  Simplified: discoloration
"""

# Function to simplify symptoms with post-processing
def simplify_symptom(description):
    """
    Simplifies a verbose symptom description using few-shot inference with post-processing.
    """
    # Append the new symptom to the few-shot examples
    prompt = f"{few_shot_examples}\n- Symptom: {description}\n  Simplified:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,  # Enforce short output
        temperature=0.5,    # Less randomness
        top_p=0.8,          # Restrict to higher-probability terms
        eos_token_id=tokenizer.eos_token_id
    )
    # Decode the output and clean it up
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Extract the part after "Simplified:" and before the next line
    simplified_term = result.split("Simplified:")[-1].strip().split("\n")[0].strip()
    return simplified_term

# Sample test symptoms
test_symptoms = [
    "tummy or back pain",
    "a pulsing feeling in your tummy",
    "bringing up milk or being sick during or shortly after feeding",
    "coughing or hiccupping when feeding",
    "the main symptom of acanthosis nigricans is patches of skin that are darker and thicker than usual"
]

# Test the model with refined few-shot prompting and post-processing
print("Testing the Model with Refined Few-Shot Prompting:\n")
for symptom in test_symptoms:
    simplified = simplify_symptom(symptom)
    print(f"Original: {symptom}")
    print(f"Simplified: {simplified}\n")

"""###GPU Loading"""

import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is enabled and ready!")
    device = torch.device("cuda")
else:
    print("Using CPU. GPU not available.")
    device = torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

"""## Structured Data Extraction"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Llama model and move it to GPU
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# Few-shot examples
few_shot_examples = """
Convert the following symptom descriptions into concise medical terms:
- Symptom: tummy pain
  Simplified: abdomen pain, pain

- Symptom: a pulsing feeling in your tummy
  Simplified: pulsation

- Symptom: bringing up milk or being sick during or shortly after feeding
  Simplified: reflux

- Symptom: coughing or hiccupping when feeding
  Simplified: feeding difficulties

- Symptom: the main symptom of acanthosis nigricans is patches of skin that are darker and thicker than usual
  Simplified: discoloration
"""

# Function to simplify symptoms using few-shot prompting
def simplify_symptom(description):
    """
    Simplifies a verbose symptom description using few-shot inference with GPU support.
    """
    # Append the new symptom to the few-shot examples
    prompt = f"{few_shot_examples}\n- Symptom: {description}\n  Simplified:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to GPU
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,  # Enforce short output
        temperature=0.5,    # Less randomness
        top_p=0.8,          # Restrict to higher-probability terms
        eos_token_id=tokenizer.eos_token_id
    )
    # Decode the output and clean it up
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Extract the part after "Simplified:" and before the next line
    simplified_term = result.split("Simplified:")[-1].strip().split("\n")[0].strip()
    return simplified_term

# Load the dataset
file_path = "/content/disease_symptoms.xlsx" 
data = pd.read_excel(file_path)

# Apply the function to simplify symptoms
data['Simplified_Symptom'] = data['Symptom'].apply(simplify_symptom)

# Save the results to a new Excel file
output_file_path = "/content/simplified_disease_symptoms.xlsx"
data.to_excel(output_file_path, index=False)

print(f"Simplified data saved to {output_file_path}")