from fastapi import FastAPI
from api.v1 import endpoints

app = FastAPI(title="scan-pdf")

app.include_router(endpoints.router, prefix="/api/v1")

@app.get("/")
def root():
    return {"status": "ok", "service": "scan-pdf"}
