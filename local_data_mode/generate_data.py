import os
import sys
import json
import random
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv('.env')
load_dotenv('../.env')

AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    print("❌ Error: Missing Azure OpenAI credentials in .env file.")
    print("Please ensure your .env file has AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT defined.")
    sys.exit(1)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

prompt = "A model that takes in an unstructured, raw conversation transcript between a doctor and a patient, and outputs a highly professional, structured clinical SOAP note containing Subjective, Objective, Assessment, and Plan sections."
temperature = 0.4
number_of_examples = 50

print("🚀 Starting local dataset generation...")
print(f"Deployment: {AZURE_OPENAI_DEPLOYMENT}")
print(f"Endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f"Target Examples: {number_of_examples}")

N_RETRIES = 3

@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
def generate_example(prompt, prev_examples, temperature=.5):
    messages=[
        {
            "role": "system",
            "content": f"You are generating data which will be used to train a machine learning model.\\n\\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\\n\\nYou will do so in this format:\\n```\\nprompt\\n-----------\\n$prompt_goes_here\\n-----------\\n\\nresponse\\n-----------\\n$response_goes_here\\n-----------\\n```\\n\\nOnly one prompt/response pair should be generated per turn.\\n\\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\\n\\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\\n\\nHere is the type of model we want to train:\\n`{prompt}`"
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 10:
            prev_examples = random.sample(prev_examples, 10)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=4000,
    )

    return response.choices[0].message.content

checkpoint_file = 'checkpoint_examples.json'
prev_examples = []

if os.path.exists(checkpoint_file):
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            prev_examples = json.load(f)
        print(f"🔄 Resuming from checkpoint: loaded {len(prev_examples)} already-generated examples.")
    except Exception as e:
        print(f"Could not load checkpoint: {e}. Starting from scratch.")

start_idx = len(prev_examples)
if start_idx < number_of_examples:
    for i in range(start_idx, number_of_examples):
        print(f"Generating example {i+1}/{number_of_examples}...")
        example = generate_example(prompt, prev_examples, temperature)
        prev_examples.append(example)
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(prev_examples, f, indent=2, ensure_ascii=False)
    print("✨ All examples generated successfully!")

# Parse prompts and responses
prompts = []
responses = []
for example in prev_examples:
    split_example = example.split('-----------')
    if len(split_example) >= 4:
        prompts.append(split_example[1].strip())
        responses.append(split_example[3].strip())
    else:
        print(f"Skipping malformed example: {example[:100]}...")

df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})
df = df.drop_duplicates()
print(f"Parsed {len(df)} unique examples.")

# Split into train and test sets
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

# Save the dataframes
train_df.to_json('train.jsonl', orient='records', lines=True)
test_df.to_json('test.jsonl', orient='records', lines=True)
print("💾 Saved train.jsonl and test.jsonl successfully!")

# Cleanup checkpoint
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print("🧹 Cleaned up temporary checkpoint file.")

print("\n🎉 DONE! You can now upload 'train.jsonl' and 'test.jsonl' to your Google Drive inside the 'gpt-llm-trainer' folder.")
