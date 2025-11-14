import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from backend.app.database import FingerprintDB
from backend.app.fingerprint import (
    _generate_fingerprints_from_array,
    FFT_WINDOW_SIZE,
    FFT_HOP_LENGTH,
    SAMPLE_RATE
)

# --- Constants ---
# We'll analyze audio in chunks of this duration (in seconds).
# This is our "rolling window".
ANALYSIS_CHUNK_DURATION = 4

# We'll slide the window forward by this duration (in seconds).
# This overlap helps ensure we don't miss fingerprints that span two windows.
ANALYSIS_SLIDE_DURATION = 2

# Convert durations to buffer sizes in bytes.
# Audio is coming in as 16-bit integers (2 bytes per sample).
ANALYSIS_CHUNK_BYTES = int(ANALYSIS_CHUNK_DURATION * SAMPLE_RATE * 2)
ANALYSIS_SLIDE_BYTES = int(ANALYSIS_SLIDE_DURATION * SAMPLE_RATE * 2)


class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)


class AudioProcessor:
    """
    Handles the audio processing for a single WebSocket connection.
    Manages a rolling buffer and runs the identification logic.
    """
    def __init__(self, websocket: WebSocket, db: FingerprintDB):
        self.websocket = websocket
        self.db = db
        self.buffer = bytearray()
        self.processing_task = None

    async def process_audio_stream(self):
        """
        Continuously receives audio data, adds it to the buffer,
        and triggers analysis when the buffer is full.
        """
        try:
            while True:
                data = await self.websocket.receive_bytes()
                self.buffer.extend(data)

                # If the buffer is large enough, start or continue processing
                if len(self.buffer) >= ANALYSIS_CHUNK_BYTES and not self.processing_task:
                    self.processing_task = asyncio.create_task(self._analyze_buffer())

        except WebSocketDisconnect:
            print("Client disconnected.")
            if self.processing_task:
                self.processing_task.cancel()

    async def _analyze_buffer(self):
        """
        Analyzes the current audio buffer to identify a song.
        This runs as a background task.
        """
        print("Analyzing audio buffer...")
        
        # Take a snapshot of the buffer for analysis
        buffer_snapshot = self.buffer[:ANALYSIS_CHUNK_BYTES]
        
        # Convert the raw bytes (Int16) to a NumPy array of floats,
        # which is what our fingerprinting code expects.
        audio_array = np.frombuffer(buffer_snapshot, dtype=np.int16).astype(np.float32) / 32768.0

        # --- Use our refactored fingerprinting logic ---
        fingerprints = _generate_fingerprints_from_array(audio_array, "stream")
        matches = self.db.get_matches(fingerprints)
        
        if matches:
            best_match = matches[0]
            result = {
                "match": True,
                "song_id": best_match[0],
                "confidence": best_match[1]
            }
            await self.websocket.send_json(result)
            print(f"Sent match to client: {result}")

        # Slide the buffer window
        self.buffer = self.buffer[ANALYSIS_SLIDE_BYTES:]
        
        # Allow the next analysis to run
        self.processing_task = None


# --- WebSocket Endpoint ---
async def websocket_endpoint(websocket: WebSocket, db: FingerprintDB):
    """The main WebSocket endpoint for audio streaming."""
    manager = ConnectionManager()
    await manager.connect(websocket)
    
    processor = AudioProcessor(websocket, db)
    await processor.process_audio_stream()
    
    manager.disconnect(websocket)
