from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow CORS from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend URL for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store your Hugging Face API token securely
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_sentiment(input: TextInput):
    payload = {"inputs": input.text}
    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code != 200:
        return {"error": "Hugging Face API error", "details": response.json()}

    result = response.json()[0]
    # HuggingFace returns list like: [{'label': 'Positive', 'score': 0.98}, ...]
    sorted_result = sorted(result, key=lambda x: x['score'], reverse=True)
    top = sorted_result[0]

    return {"label": top['label'], "score": round(top['score'], 3)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)