from fastapi import FastAPI
from app.api.routes.standardization import router as standardization_router

app = FastAPI(title="Body Vision API")

app.include_router(standardization_router)

@app.get("/")
def root():
    return {"message": "Body Vision API is running"}