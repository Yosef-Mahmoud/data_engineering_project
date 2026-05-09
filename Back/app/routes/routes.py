from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.utils.handel_file_upload import handle_upload
from app.utils.train_pipeline import train_pipeline, download_model
from app.utils.storage.storage import job_storage

router = APIRouter(prefix="/api/v1")


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    return await handle_upload(file)


@router.post("/train")
async def train(payload: dict):
    if not payload:
        raise HTTPException(status_code=422, detail="Request body is empty.")
    return await train_pipeline(payload)


@router.get("/download/{job_id}", response_class=StreamingResponse)
async def save_model(job_id: str):
    """
    Bug-fix: the original code deleted job_storage[job_id] in BOTH this route's
    finally block AND inside download_model's finally block, causing a redundant
    (though guarded) double-delete.  Cleanup now lives only here, and
    download_model is a pure serialiser with no side-effects.
    """
    try:
        return await download_model(job_id)
    finally:
        if job_id in job_storage:
            del job_storage[job_id]