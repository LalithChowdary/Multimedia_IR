import os
import shutil
from fastapi import FastAPI, UploadFile, File, WebSocket, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
from typing import Optional

# Add music_recognition to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'music_recognition'))
# Add video_search to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'video_search'))

from fingerprint import generate_fingerprints
from database import FingerprintDB
from streaming import websocket_endpoint

# Video search imports
from search import get_search_engine

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

# Initialize video search engine
video_search = get_search_engine()

print("FastAPI server started.")
print("- Music fingerprint database loaded.")
print("- Video search engine initialized.")

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
    return {
        "message": "Multimedia IR System",
        "services": {
            "music_recognition": "POST to /identify or WebSocket to /ws/audio",
            "video_search": "POST to /video/upload, GET /video/search?q=query"
        }
    }


# ============================================================================
# VIDEO SEARCH ENDPOINTS
# ============================================================================

@app.post("/video/upload")
async def upload_video(
    video_file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """
    Upload and index a video for searchable transcripts.
    
    Args:
        video_file: Video file to upload
        language: Optional language code (e.g., 'en', 'es')
    
    Returns:
        Indexing results with video metadata
    """
    # Create videos directory if it doesn't exist
    videos_dir = Path("../videos")
    videos_dir.mkdir(exist_ok=True, parents=True)
    
    # Save uploaded video
    video_path = videos_dir / video_file.filename
    
    try:
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        # Index the video
        result = video_search.index_video(
            video_path=str(video_path),
            language=language
        )
        
        return JSONResponse(status_code=200, content=result)
        
    except Exception as e:
        # Clean up on error
        if video_path.exists():
            os.remove(video_path)
        
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to index video: {str(e)}"}
        )
    
    finally:
        video_file.file.close()


@app.get("/video/search")
async def search_videos(q: str, limit: int = 10):
    """
    Search videos by semantic query.
    
    Args:
        q: Search query text
        limit: Maximum number of results to return
    
    Returns:
        List of matching video segments with timestamps
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    
    try:
        results = video_search.search(q, n_results=limit)
        
        return JSONResponse(status_code=200, content={
            "query": q,
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Search failed: {str(e)}"}
        )


@app.get("/video/list")
async def list_videos():
    """
    List all indexed videos.
    
    Returns:
        List of videos with metadata
    """
    try:
        videos = video_search.list_indexed_videos()
        
        return JSONResponse(status_code=200, content={
            "videos": videos,
            "count": len(videos)
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list videos: {str(e)}"}
        )


@app.delete("/video/{video_id}")
async def delete_video(video_id: str):
    """
    Delete a video from the index.
    
    Args:
        video_id: ID of the video to delete
    
    Returns:
        Deletion confirmation
    """
    try:
        video_search.delete_video(video_id)
        
        return JSONResponse(status_code=200, content={
            "message": f"Video {video_id} deleted successfully"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete video: {str(e)}"}
        )


@app.get("/video/stats")
async def video_stats():
    """
    Get video search engine statistics.
    
    Returns:
        Statistics about indexed videos and segments
    """
    try:
        stats = video_search.get_stats()
        
        return JSONResponse(status_code=200, content=stats)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get stats: {str(e)}"}
        )
