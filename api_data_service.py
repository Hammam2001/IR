from fastapi import FastAPI
from pydantic import BaseModel
from data_service import preprocess_text 

app = FastAPI(title= "Data Preprocessing Service")

class TextRequest(BaseModel):
    text: str

@app.post("/preprocess")
def preprocess_text_api(request: TextRequest):
  
    cleaned_text = preprocess_text(request.text)
    
    return {
        "original_text": request.text,
        "processed_text": cleaned_text
    }