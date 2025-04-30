from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import os
import json
import logging
from typing import List, Optional
import asyncio
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from merging import merge_questions
from mongodb_loader import MongoDBLoader
from pymongo import MongoClient
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ondemand_pipeline")

app = FastAPI()

# Allow CORS for local development/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_PATH = '../cleaned_dataset/all_complete_dataset.csv'
MONGODB_URI = 'mongodb://localhost:27017/'
DB_NAME = 'nepalhacks'
COLLECTION_NAME = 'clusters'

class UserInput(BaseModel):
    cqid: str
    question: str
    answer: str
    details: str = ""
    # Add other fields as needed

# --- File Upload Endpoint ---
@app.post('/upload-csv')
async def upload_csv(file: UploadFile = File(...)):
    with open(DATASET_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Optionally, update MongoDB after upload
    loader = MongoDBLoader()
    loader.run_pipeline(os.path.basename(DATASET_PATH), drop_existing=True)
    return {"message": "CSV uploaded and database updated."}

# --- Trigger Processing Endpoint ---
@app.post('/trigger-processing')
async def trigger_processing():
    loader = MongoDBLoader()
    loader.run_pipeline(os.path.basename(DATASET_PATH), drop_existing=True)
    return {"message": "Processing triggered and database updated."}

# --- Return JSON Results for Frontend ---
@app.get('/results')
async def get_results():
    df = pd.read_csv(DATASET_PATH)
    return JSONResponse(df.to_dict(orient='records'))

# --- Accept User Actions (merge/update/delete) ---
@app.post('/merge')
async def merge_action(cqids: List[str] = Form(...)):
    df = pd.read_csv(DATASET_PATH)
    df = merge_questions(df, cqids)
    df.to_csv(DATASET_PATH, index=False)
    loader = MongoDBLoader()
    loader.run_pipeline(os.path.basename(DATASET_PATH), drop_existing=True)
    return {"message": f"Merged questions for cqids: {cqids}"}

@app.put('/update')
async def update_action(user_input: UserInput):
    df = pd.read_csv(DATASET_PATH)
    idx = df.index[df['cqid'] == user_input.cqid].tolist()
    if not idx:
        raise HTTPException(status_code=404, detail="Question not found")
    for col, val in user_input.dict().items():
        df.at[idx[0], col] = val
    df.to_csv(DATASET_PATH, index=False)
    loader = MongoDBLoader()
    loader.run_pipeline(os.path.basename(DATASET_PATH), drop_existing=True)
    return {"message": "Question updated and database updated."}

@app.delete('/delete')
async def delete_action(cqid: str = Form(...)):
    df = pd.read_csv(DATASET_PATH)
    if cqid not in df['cqid'].values:
        raise HTTPException(status_code=404, detail="Question not found")
    df = df[df['cqid'] != cqid]
    df.to_csv(DATASET_PATH, index=False)
    loader = MongoDBLoader()
    loader.run_pipeline(os.path.basename(DATASET_PATH), drop_existing=True)
    return {"message": "Question deleted and database updated."}

# --- MongoDB Helper ---
def get_mongo_collection():
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

# --- SSE Pipeline Trigger (unchanged) ---
async def sse_event_generator(user_input: UserInput):
    yield f"data: Loading dataset...\n\n"
    df = pd.read_csv(DATASET_PATH)
    await asyncio.sleep(0.1)
    yield f"data: Adding new user input...\n\n"
    new_row = user_input.dict()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    await asyncio.sleep(0.1)
    yield f"data: Merging redundancy for cqid {user_input.cqid}...\n\n"
    df = merge_questions(df, [user_input.cqid])
    await asyncio.sleep(0.1)
    yield f"data: Saving updated dataset...\n\n"
    df.to_csv(DATASET_PATH, index=False)
    await asyncio.sleep(0.1)
    yield f"data: Updating MongoDB...\n\n"
    loader = MongoDBLoader()
    loader.run_pipeline(os.path.basename(DATASET_PATH), drop_existing=False)
    await asyncio.sleep(0.1)
    yield f"data: Pipeline complete.\n\n"

@app.post('/trigger-pipeline')
async def trigger_pipeline(user_input: UserInput, request: Request):
    async def event_stream():
        async for event in sse_event_generator(user_input):
            yield event
    return StreamingResponse(event_stream(), media_type='text/event-stream')
