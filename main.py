from fastapi import FastAPI  
from transformers import pipeline  
from pydantic import BaseModel  
from typing import List, Dict  

class Item(BaseModel):  
    text: str  

class BatchItem(BaseModel):  
    texts: List[str]  

app = FastAPI()  
classifier = pipeline("sentiment-analysis")  

@app.get("/")  
def root():  
    return {"FastApi service started!"}  

@app.get("/{text}")  
def get_params(text: str):  
    return classifier(text)  

@app.post("/predict/")  
def predict(item: Item):  
    return classifier(item.text)  

@app.post("/predict_batch/")  
def predict_batch(batch_item: BatchItem):  
    results = classifier(batch_item.texts)
    positive_count = sum(1 for result in results if result['label'] == 'POSITIVE')  
    negative_count = sum(1 for result in results if result['label'] == 'NEGATIVE')  
    
    return {  
        "results": results,  
        "positive_count": positive_count,  
        "negative_count": negative_count  
    }