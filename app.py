from fastapi import FastAPI
from routes.tts import router as tts_route
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to EDGE tts"
    }

@app.get("/status")
async def status():
    return {
        "message": "Server is healthy"
    }

app.include_router(tts_route)
