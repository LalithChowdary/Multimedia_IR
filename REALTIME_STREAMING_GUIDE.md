# Real-time Audio Fingerprinting System

## Architecture Overview

This system implements continuous audio streaming from a Next.js frontend to a FastAPI backend for real-time audio fingerprinting and song recognition.

## How It Works

### 1. Frontend (Next.js)

**File: `app/components/AudioRecorder.tsx`**

- **Start/Stop Recording Button**: Controls the audio streaming session
- **AudioWorklet Integration**: Uses `audio-processor.js` for efficient audio processing
- **WebSocket Connection**: Streams audio continuously to backend
- **Real-time UI Updates**: Displays matches as they are found without stopping

**File: `public/audio-processor.js`**

- **Float32 to Int16 Conversion**: Converts browser's Float32 PCM to Int16 format
- **Buffer Management**: Collects 4096 samples (~185ms) before sending
- **Continuous Streaming**: Sends chunks via WebSocket as they're ready

### 2. Backend (FastAPI)

**File: `backend/app/main.py`**

- **WebSocket Endpoint**: `/ws/audio` accepts continuous audio streams
- **CORS Configuration**: Allows frontend at localhost:3000

**File: `backend/app/websocket.py`**

- **Rolling Buffer**: Maintains a sliding window of audio (4 seconds)
- **Continuous Analysis**: Analyzes audio in 2-second sliding windows
- **Non-blocking Processing**: Uses asyncio for concurrent stream handling
- **Match Detection**: Sends JSON results back to frontend in real-time

**File: `backend/app/fingerprint.py`**

- **STFT Processing**: Short-Time Fourier Transform for spectral analysis
- **Peak Picking**: Identifies spectral peaks using maximum filter
- **Hash Generation**: Creates combinatorial hashes from peak pairs
- **Database Matching**: Compares against pre-computed fingerprints

## Data Flow

```
Microphone 
  ↓
AudioContext (22050 Hz, mono)
  ↓
AudioWorklet (audio-processor.js)
  ↓ Float32Array → Int16Array conversion
  ↓ Buffering (4096 samples)
  ↓
WebSocket Stream
  ↓
FastAPI Backend (/ws/audio)
  ↓
Rolling Buffer (4 seconds)
  ↓
STFT + Peak Detection
  ↓
Fingerprint Generation
  ↓
Database Matching
  ↓
JSON Response → Frontend UI
```

## Key Features

✅ **Continuous Streaming**: Audio streams without interruption
✅ **Rolling Window Analysis**: 4-second windows with 2-second overlap
✅ **Real-time Results**: Matches displayed as soon as found
✅ **Non-blocking**: Frontend continues streaming while backend processes
✅ **Efficient Format**: Int16 PCM for minimal bandwidth
✅ **Visual Feedback**: Live status updates and match displays

## Configuration Parameters

### Frontend
- **Sample Rate**: 22050 Hz (mono)
- **Buffer Size**: 4096 samples (~185ms per chunk)
- **WebSocket URL**: `ws://127.0.0.1:8000/ws/audio`

### Backend
- **Analysis Window**: 4 seconds
- **Slide Duration**: 2 seconds (50% overlap)
- **FFT Window**: 4096 samples
- **FFT Hop Length**: 2048 samples
- **Peak Neighborhood**: 20x20

## Running the System

### Backend
```bash
cd /Users/lalith/Snu/sem5/IR/pro/code
source .venv/bin/activate
cd backend/app
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend
```bash
cd /Users/lalith/Snu/sem5/IR/pro/code
npm run dev
```

Open `http://localhost:3000` and click **Start** to begin streaming!

## How Continuous Analysis Works

1. **Frontend** continuously captures microphone audio
2. **AudioWorklet** converts and buffers audio into chunks
3. **WebSocket** streams chunks to backend in real-time
4. **Backend** accumulates chunks into a rolling buffer
5. **Analysis Task** runs asynchronously:
   - Takes 4-second snapshots
   - Generates fingerprints via STFT + peak detection
   - Matches against database
   - Sends results back if match found
   - Slides window by 2 seconds and repeats
6. **Frontend** displays matches while continuing to stream

## Benefits of This Architecture

- **Low Latency**: Matches found within seconds of audio playing
- **No Interruption**: System keeps listening even after finding a match
- **Resource Efficient**: Sliding windows prevent memory buildup
- **Scalable**: Async processing handles multiple connections
- **Robust**: Overlap ensures no audio is missed between windows
