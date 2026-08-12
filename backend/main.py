from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database.session import init_db
from backend.api.routes import router

app = FastAPI(
    title="Atlas ML Baseline Engine API",
    description="Automated baseline machine learning preprocessing, training, and evaluation service.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Atlas ML Baseline Engine is active.", "status": "online"}
