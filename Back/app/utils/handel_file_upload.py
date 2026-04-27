
from fastapi import APIRouter
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import uuid
from app.utils.storage.storage import data_storage
import io

async def handle_upload(file: UploadFile = File(...)):
    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .csv or .xlsx file.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(status_code=422, detail="The file was parsed but contains no data.")
    if len(df.columns) < 2:
        raise HTTPException(status_code=422, detail="The dataset must have at least 2 columns.")

    job_id = str(uuid.uuid4())
    data_storage[job_id] = df

    return {
        "job_id": job_id,
        "columns": df.columns.tolist(),
        "rows": df.head(5).values.tolist(),
        "dtypes": df.dtypes.apply(lambda x: x.name).to_dict()
    }
