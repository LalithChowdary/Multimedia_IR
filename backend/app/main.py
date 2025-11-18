import os
import shutil
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

# Add music_recognition to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'music_recognition'))

from fingerprint import generate_fingerprints
from database import FingerprintDB
from streaming import websocket_endpoint

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Middleware ---
# This allows our frontend (running on localhost:3000) to communicate with our backend.
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Loading ---
# Load the fingerprint database into memory when the app starts.
db = FingerprintDB()
db.load()

print("FastAPI server started. Fingerprint database loaded.")

# --- WebSocket Endpoint ---
@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket_endpoint(websocket, db)

# --- API Endpoints ---
@app.post("/identify")
async def identify_song(audio_file: UploadFile = File(...)):
    """
    Identifies a song from an uploaded audio file.

    The audio file is temporarily saved, fingerprinted, and then matched
    against the database.
    """
    # Create a temporary directory to store the uploaded file
    temp_dir = Path("backend/temp")
    temp_dir.mkdir(exist_ok=True)
    temp_file_path = temp_dir / audio_file.filename

    # Save the uploaded file
    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # Generate fingerprints for the uploaded sample
        sample_fingerprints = generate_fingerprints(str(temp_file_path))

        if not sample_fingerprints:
            return JSONResponse(status_code=400, content={"message": "Could not process audio file."})

        # Match against the database
        matches = db.get_matches(sample_fingerprints)

        # Prepare the response
        if matches:
            best_match = matches[0]
            result = {
                "match": True,
                "song_id": best_match[0],
                "confidence": best_match[1]
            }
        else:
            result = {"match": False}

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

    finally:
        # Clean up the temporary file
        if temp_file_path.exists():
            os.remove(temp_file_path)
        # Close the uploaded file
        audio_file.file.close()

@app.get("/")
def read_root():
    return {"message": "Shazam-like service is running. POST to /identify to recognize a song."}
