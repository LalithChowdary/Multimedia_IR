'use client';

import { useState, useRef, useEffect } from 'react';

// --- Type Definitions ---
interface MatchResult {
    match: boolean;
    song_id?: string;
    confidence?: number;
    offset?: number;
    confirmed?: boolean;
    message?: string;
}

// --- Component ---
export default function AudioRecorder() {
    // --- State Management ---
    const [isRecording, setIsRecording] = useState(false);
    const [status, setStatus] = useState('Idle. Press Start to begin.');
    const [match, setMatch] = useState<MatchResult | null>(null);
    const [isListening, setIsListening] = useState(false);

    // --- Refs for Audio and WebSocket Objects ---
    // We use refs to hold objects that don't need to trigger re-renders
    // when they change, like the WebSocket or AudioContext.
    const socketRef = useRef<WebSocket | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const audioStreamRef = useRef<MediaStream | null>(null);
    const audioWorkletNodeRef = useRef<AudioWorkletNode | null>(null);

    // --- Effect for Cleanup ---
    // This ensures that we close connections and release the microphone
    // when the component is unmounted.
    useEffect(() => {
        return () => {
            stopRecording();
        };
    }, []);

    // --- Core Functions ---

    const startRecording = async () => {
        setMatch(null);
        setStatus('Connecting...');
        setIsListening(false);

        try {
            // 1. Get Microphone Access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioStreamRef.current = stream;

            // 2. Create and Configure AudioContext
            // Use 8KHz sample rate to match Shazam algorithm (as per Wang's paper)
            const context = new AudioContext({ sampleRate: 8000 });
            audioContextRef.current = context;

            // 3. Load the AudioWorklet
            await context.audioWorklet.addModule('/audio-processor.js');
            const workletNode = new AudioWorkletNode(context, 'audio-processor');
            audioWorkletNodeRef.current = workletNode;

            // 4. Setup WebSocket Connection
            const ws = new WebSocket('ws://127.0.0.1:8000/ws/audio');
            socketRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket connection established.');
                setStatus('Recording... Listening for music.');
                setIsRecording(true);
                setIsListening(true);

                // 5. Connect the Audio Pipeline
                // Microphone -> AudioWorklet -> WebSocket
                const source = context.createMediaStreamSource(stream);
                source.connect(workletNode);

                // 6. Handle Messages from the AudioWorklet
                // The worklet sends us processed audio chunks (Int16Array).
                workletNode.port.onmessage = (event) => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(event.data);
                    }
                };
            };

            // 7. Handle Messages from the Server (Continuous Updates)
            ws.onmessage = (event) => {
                const result: MatchResult = JSON.parse(event.data);
                console.log('Received from server:', result);
                
                if (result.match) {
                    // Match found! Display it and continue listening
                    setMatch(result);
                    if (result.confirmed) {
                        setStatus('✓ Confirmed Match! Still listening...');
                    } else {
                        setStatus(result.message || 'Potential match detected...');
                    }
                } else {
                    // Still listening, no match yet
                    setStatus(result.message || 'Listening for music...');
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setStatus('Error connecting to server. Please try again.');
                stopRecording();
            };

            ws.onclose = () => {
                console.log('WebSocket connection closed.');
                if (isRecording) {
                    setStatus('Connection lost. Please try again.');
                    stopRecording();
                }
            };

        } catch (error) {
            console.error('Error starting recording:', error);
            setStatus('Could not start recording. Please grant microphone permission.');
        }
    };

    const stopRecording = () => {
        if (socketRef.current) {
            socketRef.current.close();
            socketRef.current = null;
        }
        if (audioStreamRef.current) {
            audioStreamRef.current.getTracks().forEach(track => track.stop());
            audioStreamRef.current = null;
        }
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        if (audioWorkletNodeRef.current) {
            audioWorkletNodeRef.current.port.close();
            audioWorkletNodeRef.current = null;
        }
        setIsRecording(false);
        setIsListening(false);
        if (status.startsWith('Recording') || status.startsWith('Match found')) {
            setStatus('Idle. Press Start to begin.');
        }
    };

    const handleToggleRecording = () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    };

    // --- UI Rendering ---
    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', fontFamily: 'sans-serif', color: '#333', padding: '20px' }}>
            <h1>🎵 Real-time Audio Recognition</h1>
            
            {/* Status Display */}
            <div style={{ 
                marginBottom: '20px', 
                padding: '15px 30px', 
                backgroundColor: isListening ? '#e8f5e9' : '#f5f5f5',
                borderRadius: '10px',
                border: isListening ? '2px solid #4caf50' : '2px solid #ddd',
                minWidth: '300px',
                textAlign: 'center'
            }}>
                <p style={{ margin: 0, fontSize: '16px', fontWeight: '500' }}>
                    {isListening && '🎤 '}
                    {status}
                </p>
            </div>

            {/* Record/Stop Button */}
            <button
                onClick={handleToggleRecording}
                style={{
                    padding: '15px 40px',
                    fontSize: '18px',
                    cursor: 'pointer',
                    borderRadius: '50px',
                    border: 'none',
                    backgroundColor: isRecording ? '#e74c3c' : '#2ecc71',
                    color: 'white',
                    fontWeight: 'bold',
                    transition: 'all 0.3s',
                    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                    minWidth: '150px'
                }}
                onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'scale(1.05)';
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'scale(1)';
                }}
            >
                {isRecording ? '⏹ Stop' : '▶ Start'}
            </button>

            {/* Match Result Display */}
            {match && match.match && (
                <div style={{ 
                    marginTop: '30px', 
                    padding: '25px', 
                    border: match.confirmed ? '3px solid #2e7d32' : '2px solid #ff9800', 
                    borderRadius: '15px', 
                    backgroundColor: match.confirmed ? '#e8f5e9' : '#fff3e0',
                    minWidth: '400px',
                    boxShadow: match.confirmed 
                        ? '0 4px 12px rgba(46, 125, 50, 0.3)' 
                        : '0 4px 12px rgba(255, 152, 0, 0.2)'
                }}>
                    <h2 style={{ 
                        margin: '0 0 15px 0', 
                        color: match.confirmed ? '#2e7d32' : '#e65100' 
                    }}>
                        {match.confirmed ? '✓✓ Confirmed Match!' : '? Potential Match'}
                    </h2>
                    <div style={{ fontSize: '16px' }}>
                        <p style={{ margin: '10px 0' }}>
                            <strong>Song:</strong> <span style={{ color: '#1976d2' }}>{match.song_id}</span>
                        </p>
                        <p style={{ margin: '10px 0' }}>
                            <strong>Confidence:</strong> <span style={{ color: '#f57c00' }}>{match.confidence}</span>
                        </p>
                        {match.offset !== undefined && (
                            <p style={{ margin: '10px 0' }}>
                                <strong>Time Offset:</strong> <span style={{ color: '#7b1fa2' }}>{match.offset} frames</span>
                            </p>
                        )}
                        {!match.confirmed && (
                            <p style={{ margin: '15px 0 0 0', fontSize: '14px', color: '#e65100', fontStyle: 'italic' }}>
                                Waiting for confirmation across multiple windows...
                            </p>
                        )}
                    </div>
                </div>
            )}

            {/* Info Text */}
            <div style={{ 
                marginTop: '30px', 
                padding: '15px', 
                backgroundColor: '#f9f9f9', 
                borderRadius: '10px',
                maxWidth: '500px',
                textAlign: 'center'
            }}>
                <p style={{ margin: 0, fontSize: '14px', color: '#666' }}>
                    <strong>Shazam Algorithm</strong> - Using 10-second windows with 3-second overlap.
                    Matches are confirmed when detected consistently across multiple windows.
                </p>
            </div>
        </div>
    );
}
