'use client';

import { useState, useRef, useEffect } from 'react';

interface MatchResult {
    match: boolean;
    song_id?: string;
    confidence?: number;
    offset?: number;
    confirmed?: boolean;
    message?: string;
}

export default function AudioRecorder() {
    const [isRecording, setIsRecording] = useState(false);
    const [status, setStatus] = useState('');
    const [match, setMatch] = useState<MatchResult | null>(null);
    const [isListening, setIsListening] = useState(false);

    const socketRef = useRef<WebSocket | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const audioStreamRef = useRef<MediaStream | null>(null);
    const audioWorkletNodeRef = useRef<AudioWorkletNode | null>(null);

    useEffect(() => {
        return () => stopRecording();
    }, []);

    const startRecording = async () => {
        setMatch(null);
        setStatus('Connecting...');
        setIsListening(false);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioStreamRef.current = stream;

            const context = new AudioContext({ sampleRate: 8000 });
            audioContextRef.current = context;

            await context.audioWorklet.addModule('/audio-processor.js');
            const workletNode = new AudioWorkletNode(context, 'audio-processor');
            audioWorkletNodeRef.current = workletNode;

            const ws = new WebSocket('ws://127.0.0.1:8000/ws/audio');
            socketRef.current = ws;

            ws.onopen = () => {
                setStatus('Listening...');
                setIsRecording(true);
                setIsListening(true);
                const source = context.createMediaStreamSource(stream);
                source.connect(workletNode);
                workletNode.port.onmessage = (event) => {
                    if (ws.readyState === WebSocket.OPEN) ws.send(event.data);
                };
            };

            ws.onmessage = (event) => {
                const result: MatchResult = JSON.parse(event.data);
                if (result.match) {
                    setMatch(result);
                    if (result.confirmed) setStatus('Match Confirmed');
                    else setStatus('Analyzing...');
                } else {
                    setStatus('Listening...');
                }
            };

            ws.onerror = () => {
                setStatus('Connection Error');
                stopRecording();
            };

            ws.onclose = () => {
                if (isRecording) stopRecording();
            };

        } catch (error) {
            console.error(error);
            setStatus('Microphone Error');
        }
    };

    const stopRecording = () => {
        if (socketRef.current) {
            socketRef.current.close();
            socketRef.current = null;
        }
        
        if (audioStreamRef.current) {
            audioStreamRef.current.getTracks().forEach(t => t.stop());
            audioStreamRef.current = null;
        }

        if (audioContextRef.current) {
            try {
                if (audioContextRef.current.state !== 'closed') {
                    audioContextRef.current.close();
                }
            } catch (e) {
                console.error("Error closing AudioContext:", e);
            }
            audioContextRef.current = null;
        }

        if (audioWorkletNodeRef.current) {
            // Worklet ports don't strictly need explicit closing if the context is closed,
            // but it's good practice if supported.
            try {
                audioWorkletNodeRef.current.port.close();
            } catch (e) {
                // Ignore
            }
            audioWorkletNodeRef.current = null;
        }
        
        setIsRecording(false);
        setIsListening(false);
        setStatus('');
    };

    const handleToggle = () => isRecording ? stopRecording() : startRecording();

    return (
        <div style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            justifyContent: 'center',
            height: '100%',
            width: '100%',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        }}>
            {/* Status Pill */}
            <div style={{ 
                height: '24px',
                marginBottom: '40px',
                opacity: status ? 1 : 0,
                transition: 'opacity 0.3s ease'
            }}>
                <span style={{ 
                    fontSize: '13px', 
                    fontWeight: 600, 
                    letterSpacing: '0.5px',
                    color: isListening ? '#e74c3c' : '#999',
                    textTransform: 'uppercase'
                }}>
                    {status}
                </span>
            </div>

            {/* Main Record Button */}
            <button
                onClick={handleToggle}
                style={{
                    width: '120px',
                    height: '120px',
                    borderRadius: '50%',
                    border: 'none',
                    background: isRecording ? '#fff' : '#000',
                    color: isRecording ? '#000' : '#fff',
                    cursor: 'pointer',
                    boxShadow: isRecording 
                        ? '0 0 0 4px rgba(231, 76, 60, 0.2), 0 10px 40px rgba(231, 76, 60, 0.4)' 
                        : '0 10px 30px rgba(0,0,0,0.2)',
                    position: 'relative',
                    transition: 'all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    outline: 'none'
                }}
            >
                {isRecording ? (
                    <div style={{ 
                        width: '40px', 
                        height: '40px', 
                        background: '#e74c3c', 
                        borderRadius: '4px',
                        animation: 'pulse 1.5s infinite'
                    }} />
                ) : (
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                        <line x1="12" y1="19" x2="12" y2="23"></line>
                        <line x1="8" y1="23" x2="16" y2="23"></line>
                    </svg>
                )}
            </button>

            {/* Result Display */}
            <div style={{ 
                marginTop: '60px', 
                textAlign: 'center',
                minHeight: '100px',
                opacity: match && match.match ? 1 : 0,
                transform: match && match.match ? 'translateY(0)' : 'translateY(20px)',
                transition: 'all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1)'
            }}>
                {match && match.match && (
                    <div>
                        <h2 style={{ 
                            margin: '0 0 10px 0', 
                            fontSize: '28px', 
                            fontWeight: 700, 
                            color: '#222',
                            letterSpacing: '-0.5px'
                        }}>
                            {match.song_id?.replace(/_/g, ' ').replace(/\.(mp3|wav)$/, '')}
                        </h2>
                        <p style={{ 
                            margin: 0, 
                            fontSize: '14px', 
                            color: '#888',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '10px'
                        }}>
                            {match.confirmed ? (
                                <span style={{ color: '#2ecc71', fontWeight: 600 }}>✓ Confirmed</span>
                            ) : (
                                <span>Confidence: {match.confidence}</span>
                            )}
                        </p>
                    </div>
                )}
            </div>

            <style jsx>{`
                @keyframes pulse {
                    0% { transform: scale(1); opacity: 1; }
                    50% { transform: scale(0.9); opacity: 0.8; }
                    100% { transform: scale(1); opacity: 1; }
                }
            `}</style>
        </div>
    );
}