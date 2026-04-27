from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from fastapi.responses import StreamingResponse
from app.utils.handel_file_upload import handle_upload
from app.utils.train_pipeline import train_pipeline
import io
from app.utils.train_pipeline import download_model
from app.utils.storage.storage import data_storage , job_storage

router = APIRouter(prefix="/api/v1")



@router.post("/upload")
async def upload(file : UploadFile = File(...)):
   return await handle_upload(file)
    


@router.post("/train")
async def train(payload: dict):
    if payload is None:
        raise HTTPException(status_code=422, detail="Invalid payload")
    return await train_pipeline(payload)


@router.get("/download/{job_id}", response_class=StreamingResponse)
async def save_model(job_id: str):
    try:
        return await download_model(job_id)
    finally:
        if job_id in job_storage:
            del job_storage[job_id] 
