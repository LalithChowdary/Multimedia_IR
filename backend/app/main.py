import os
import shutil
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.responses import JSONResponse
from pathlib import Path

from backend.app.fingerprint import generate_fingerprints
from backend.app.database import FingerprintDB
from backend.app.websocket import websocket_endpoint

# --- FastAPI App Initialization ---
app = FastAPI()

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
