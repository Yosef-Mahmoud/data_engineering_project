from fastapi import FastAPI
from app.routes.routes import router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your specific URL
    allow_methods=["*"],
    allow_headers=["*"],
)

