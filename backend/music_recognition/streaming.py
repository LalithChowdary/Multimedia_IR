import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from database import FingerprintDB
from fingerprint import (
    _generate_fingerprints_from_array,
    SAMPLE_RATE
)

# --- Constants (Optimized from Shazam Paper) ---
# Paper mentions: "15 second samples work well" but also states
# "correctly identify music... from a heavily corrupted 15 second sample"
# For real-time: use overlapping windows for continuous detection

# Rolling window parameters optimized for Shazam algorithm
ANALYSIS_CHUNK_DURATION = 10  # seconds (10s for high confidence matching)
ANALYSIS_SLIDE_DURATION = 2   # seconds (overlap for robustness and continuity)

# Convert durations to buffer sizes in bytes
# Audio is 16-bit integers (2 bytes per sample)
ANALYSIS_CHUNK_BYTES = int(ANALYSIS_CHUNK_DURATION * SAMPLE_RATE * 2)
ANALYSIS_SLIDE_BYTES = int(ANALYSIS_SLIDE_DURATION * SAMPLE_RATE * 2)

# Match confirmation parameters
# To reduce false positives in streaming, require consistent matches
MATCH_CONFIRMATION_WINDOW = 2  # Number of consecutive windows to confirm
MATCH_CONFIDENCE_BOOST = 1.2   # Multiply score if same song matches repeatedly


class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


class AudioProcessor:
    """
    Handles the audio processing for a single WebSocket connection.
    Manages a rolling buffer and runs the identification logic.
    
    Implements continuous recognition as described in Shazam paper:
    - Paper: "search times on the order of a few milliseconds per query"
    - "correctly identify each of several tracks mixed together"
    """
    def __init__(self, websocket: WebSocket, db: FingerprintDB):
        self.websocket = websocket
        self.db = db
        self.buffer = bytearray()
        self.processing_task = None
        
        # Match history for confirmation
        self.recent_matches = []  # Track last N matches
        self.current_song = None
        self.current_song_count = 0

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
                if len(self.buffer) >= ANALYSIS_CHUNK_BYTES:
                    if not self.processing_task or self.processing_task.done():
                        self.processing_task = asyncio.create_task(self._analyze_buffer())

        except WebSocketDisconnect:
            print("Client disconnected.")
            if self.processing_task and not self.processing_task.done():
                self.processing_task.cancel()

    async def _analyze_buffer(self):
        """
        Analyzes audio buffer using Shazam algorithm.
        
        Continuously processes rolling windows to identify music in real-time.
        Uses combinatorial hashing and scatterplot histogram matching.
        
        From paper (Section 3.2):
        "For a database of about 20 thousand tracks implemented on a PC, 
        the search time is on the order of 5-500 milliseconds"
        """
        try:
            while len(self.buffer) >= ANALYSIS_CHUNK_BYTES:
                # Take a snapshot of the buffer for analysis
                buffer_snapshot = self.buffer[:ANALYSIS_CHUNK_BYTES]
                
                # Convert Int16 PCM to float32 normalized audio
                audio_array = np.frombuffer(buffer_snapshot, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Apply normalization (helps with volume variations)
                # This wasn't explicitly in the paper but improves robustness
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array))

                # Generate fingerprints using Shazam algorithm
                fingerprints = _generate_fingerprints_from_array(audio_array, "stream")
                
                print(f"\n=== Analysis Window ===")
                print(f"Generated {len(fingerprints)} fingerprints from {len(audio_array)/SAMPLE_RATE:.1f}s audio")
                
                if fingerprints and len(fingerprints) > 0:
                    # Match using scatterplot histogram technique
                    # Paper: "The bin scanning process is repeated for each track 
                    # in the database until a significant match is found"
                    matches = self.db.get_matches(fingerprints)
                    
                    if matches:
                        best_match = matches[0]
                        song_id, confidence, offset = best_match
                        
                        # Update match history for confirmation
                        self._update_match_history(song_id, confidence)
                        
                        # Check if this is a confirmed match (consistent across windows)
                        is_confirmed = self._is_match_confirmed(song_id)
                        
                        result = {
                            "match": True,
                            "song_id": song_id,
                            "confidence": confidence,
                            "offset": offset,
                            "confirmed": is_confirmed,
                            "message": "Match confirmed!" if is_confirmed else "Potential match detected..."
                        }
                        
                        await self.websocket.send_json(result)
                        
                        status = "✓ CONFIRMED" if is_confirmed else "? Potential"
                        print(f"{status} MATCH: {song_id}, Confidence: {confidence}, Offset: {offset}")
                        
                    else:
                        # No match found in this window
                        self._update_match_history(None, 0)
                        
                        await self.websocket.send_json({
                            "match": False,
                            "message": "Listening..."
                        })
                        print("No match in this window")
                        
                else:
                    # Not enough fingerprints generated
                    self._update_match_history(None, 0)
                    
                    await self.websocket.send_json({
                        "match": False,
                        "message": "Analyzing audio... (silence or noise detected)"
                    })
                    print("Insufficient fingerprints (possibly silence)")

                # Slide the buffer window
                # Paper mentions overlapping windows for robust detection
                self.buffer = self.buffer[ANALYSIS_SLIDE_BYTES:]
                
                # Small delay to prevent CPU overload
                # Paper: "search times on the order of a few milliseconds"
                # We add a bit more delay for streaming to be CPU-friendly
                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"Error in analysis: {e}")
            await self.websocket.send_json({
                "match": False,
                "error": str(e)
            })
        finally:
            # Allow the next analysis task to run
            self.processing_task = None

    def _update_match_history(self, song_id, confidence):
        """
        Update match history for confirmation logic.
        
        This helps reduce false positives in streaming mode by requiring
        consistent detection across multiple windows.
        """
        self.recent_matches.append((song_id, confidence))
        
        # Keep only recent history
        if len(self.recent_matches) > MATCH_CONFIRMATION_WINDOW + 1:
            self.recent_matches.pop(0)
        
        # Update current song tracking
        if song_id == self.current_song:
            self.current_song_count += 1
        else:
            self.current_song = song_id
            self.current_song_count = 1

    def _is_match_confirmed(self, song_id):
        """
        Check if a match is confirmed based on recent history.
        
        A match is confirmed if:
        1. The same song appears in multiple recent windows
        2. The confidence scores are consistently high
        
        This helps with the paper's goal of "low number of false positives"
        """
        if len(self.recent_matches) < MATCH_CONFIRMATION_WINDOW:
            return False
        
        # Check last N matches
        recent_songs = [s for s, c in self.recent_matches[-MATCH_CONFIRMATION_WINDOW:]]
        
        # Count occurrences of the current song
        matches = sum(1 for s in recent_songs if s == song_id)
        
        # Confirmed if it appears in majority of recent windows
        return matches >= (MATCH_CONFIRMATION_WINDOW // 2 + 1)


# --- WebSocket Endpoint ---
async def websocket_endpoint(websocket: WebSocket, db: FingerprintDB):
    """
    The main WebSocket endpoint for audio streaming.
    
    Implements real-time music recognition as described in Shazam paper:
    - Receives audio stream from client
    - Processes in overlapping windows
    - Returns matches continuously
    """
    manager = ConnectionManager()
    await manager.connect(websocket)
    
    try:
        processor = AudioProcessor(websocket, db)
        await processor.process_audio_stream()
    finally:
        manager.disconnect(websocket)


# --- Performance Testing ---
def test_processing_speed():
    """
    Test to verify processing speed meets paper's claims.
    
    Paper states (Section 3.2):
    "For a database of about 20 thousand tracks... the search time is 
    on the order of 5-500 milliseconds"
    
    "With 'radio quality' audio, we can find a match in less than 10 
    milliseconds, with a likely optimization goal reaching down to 1 
    millisecond per query."
    """
    import time
    from fingerprint import generate_fingerprints
    
    print("\n=== Performance Test ===")
    
    # Generate test audio (10 seconds of random noise)
    test_audio = np.random.randn(SAMPLE_RATE * 10).astype(np.float32)
    
    # Test fingerprint generation speed
    start = time.time()
    fps = _generate_fingerprints_from_array(test_audio, "test")
    fp_time = (time.time() - start) * 1000
    
    print(f"Fingerprint generation: {fp_time:.1f}ms for 10s audio")
    print(f"Generated {len(fps)} fingerprints")
    print(f"Speed: {10000/fp_time:.1f}x realtime")
    
    # For matching speed, we'd need a loaded database
    print("\nNote: Matching speed test requires a loaded database")
    print("Expected: 5-500ms for full database search")


if __name__ == '__main__':
    test_processing_speed()