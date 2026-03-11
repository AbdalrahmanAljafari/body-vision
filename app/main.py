from fastapi import FastAPI
from app.api.routes.standardization import router as standardization_router

app = FastAPI(title="Body Image Standardization API")

app.include_router(standardization_router)