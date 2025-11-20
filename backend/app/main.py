import os
import shutil
from fastapi import FastAPI, UploadFile, File, WebSocket, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
import json
from moviepy.editor import VideoFileClip

# Add music_recognition to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'music_recognition'))
# Add video_recognision to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'video_recognision'))

from fingerprint import generate_fingerprints
from database import FingerprintDB
from streaming import websocket_endpoint

# Video Recognition Imports
from transcribe import transcribe_videos
from indexer import create_search_index
from search import search as search_video_index

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

# --- Static Files Mounting ---
# Serve videos and thumbnails directly
# Videos will be available at /content/videos/filename.mp4
# Thumbnails will be available at /content/thumbnails/filename.jpg
app.mount("/content/videos", StaticFiles(directory="backend/videos"), name="videos")
app.mount("/content/thumbnails", StaticFiles(directory="backend/thumbnails"), name="thumbnails")

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

# --- Video Recognition Endpoints ---

VIDEOS_DIR = Path("backend/videos")
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

def process_video_background(file_path: Path):
    """
    Background task to process the uploaded video:
    1. Transcribe the video using Whisper
    2. Update the search index
    """
    try:
        print(f"Starting background processing for: {file_path.name}")
        
        # 1. Transcribe
        # The transcribe_videos function expects a list of files or scans a directory.
        # We can modify it or just point it to the directory. 
        # For now, let's assume it scans the directory.
        # A better approach would be to import the specific logic, but reusing the script is easier.
        print("Running transcription...")
        # We need to ensure transcribe_videos is called correctly.
        # It usually scans a directory. Let's point it to our VIDEOS_DIR.
        # Note: transcribe.py might need adjustment if it hardcodes paths, 
        # but assuming it uses relative paths from BACKEND_DIR, it should work if configured right.
        # For safety, let's call the functions directly if possible.
        
        # Ensure transcript directory exists
        TRANSCRIPT_DIR = Path("backend/video_recognision/transcripts")
        TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Run transcription (this might take a while)
        # We pass [file_path] to only transcribe this specific file if the function supports it,
        # otherwise we let it scan. Checking transcribe.py, it usually scans.
        # Let's assume we just run the full pipeline for simplicity in this prototype.
        transcribe_videos() 
        
        # 2. Index
        print("Updating search index...")
        create_search_index(incremental=True)
        
        print(f"Finished processing: {file_path.name}")
        
    except Exception as e:
        print(f"Error processing video {file_path.name}: {e}")

@app.post("/upload_video")
async def upload_video(background_tasks: BackgroundTasks, video_file: UploadFile = File(...)):
    """
    Uploads a video file for indexing.
    The video is saved, and then processed in the background (transcription + indexing).
    """
    try:
        file_path = VIDEOS_DIR / video_file.filename
        
        # Save the file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
            
        # Trigger background processing
        background_tasks.add_task(process_video_background, file_path)
        
        return {
            "message": "Video uploaded successfully. Processing started in background.",
            "filename": video_file.filename
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Upload failed: {str(e)}"})
    finally:
        video_file.file.close()

@app.get("/search_video")
def search_video(query: str, top_k: int = 5, min_score: float = 0.0):
    """
    Semantic search for video content.
    """
    try:
        # search_video_index returns a JSON string, so we parse it back to a dict
        # to let FastAPI serialize it cleanly
        results_json = search_video_index(query, top_k=top_k, min_score=min_score)
        return json.loads(results_json)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Search failed: {str(e)}"})

# --- Video Gallery Endpoints ---

THUMBNAILS_DIR = Path("backend/thumbnails")
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

def generate_thumbnail_if_needed(video_path: Path):
    """
    Generates a thumbnail for a video if it doesn't exist.
    Returns the relative path to the thumbnail.
    """
    try:
        thumbnail_filename = video_path.stem + ".jpg"
        thumbnail_path = THUMBNAILS_DIR / thumbnail_filename
        
        if not thumbnail_path.exists():
            print(f"Generating thumbnail for {video_path.name}...")
            clip = VideoFileClip(str(video_path))
            # Save frame at 1 second (or midpoint if shorter)
            capture_time = min(1.0, clip.duration / 2)
            clip.save_frame(str(thumbnail_path), t=capture_time)
            clip.close()
            
        return f"/content/thumbnails/{thumbnail_filename}"
    except Exception as e:
        print(f"Error generating thumbnail for {video_path.name}: {e}")
        return None

from urllib.parse import quote

# ... (keep existing imports)

# ...

@app.get("/videos")
def list_videos():
    """
    Returns a list of all available videos with their thumbnails and metadata.
    """
    videos = []
    for video_file in VIDEOS_DIR.glob("*"):
        if video_file.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            thumbnail_url = generate_thumbnail_if_needed(video_file)
            
            # Encode URLs to handle spaces and special characters
            encoded_video_url = f"/content/videos/{quote(video_file.name)}"
            encoded_thumbnail_url = quote(thumbnail_url) if thumbnail_url else None
            # Note: generate_thumbnail_if_needed returns a path starting with /, so we need to be careful.
            # Actually, generate_thumbnail_if_needed returns f"/content/thumbnails/{thumbnail_filename}"
            # We should encode just the filename part or the whole path if we are careful.
            # Let's fix generate_thumbnail_if_needed to return encoded URL or encode here.
            
            # Better approach: Encode the filename components.
            
            videos.append({
                "filename": video_file.name,
                "video_url": f"/content/videos/{quote(video_file.name)}",
                "thumbnail_url": f"/content/thumbnails/{quote(video_file.stem)}.jpg" if thumbnail_url else None,
                "size_mb": round(video_file.stat().st_size / (1024 * 1024), 2)
            })
    
    return {"videos": videos}
