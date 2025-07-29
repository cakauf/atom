import os

import dotenv

url = "https://api.openai.com/v1"  # Replace with your API endpoint
env_file = dotenv.find_dotenv(".env") # Note: lives in parent repo
dotenv.load_dotenv(env_file)
api_key = [
    os.getenv("OPENAI_API_KEY"),  # Load API key from environment variable
    # You can add multiple API keys to improve concurrency performance.
]
