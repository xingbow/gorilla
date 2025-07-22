#!/usr/bin/env python3

import os
import sys
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# print environment variables for debugging
print("Loaded environment variables:")
for key in os.environ:  
    if key.startswith("AZURE_OPENAI_API_KEY"):
        print(f"{key}={os.getenv(key)}")
# Create Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-14"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Test common model names available in Azure OpenAI
test_models = [
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o",  
    "gpt-4o-mini"
]

print("Testing Azure OpenAI models with your configuration...")
print(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(f"API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
print()

working_models = []

for model in test_models:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(f"✅ SUCCESS: '{model}' - {response.choices[0].message.content}")
        working_models.append(model)
    except Exception as e:
        print(f"❌ FAILED: '{model}' - {str(e)}")

print("\n" + "="*50)
print("SUMMARY:")
if working_models:
    print(f"Working models: {working_models}")
    print(f"\nRecommended .env configuration:")
    print(f"AZURE_OPENAI_DEPLOYMENT=\"{working_models[0]}\"")
else:
    print("No models found. Please check your Azure OpenAI resource configuration.")
    print("Make sure you have deployed models in your Azure OpenAI resource.")