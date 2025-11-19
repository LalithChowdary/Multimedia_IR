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
    
    // Visualization Refs
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const ring1Ref = useRef<HTMLDivElement>(null);
    const ring2Ref = useRef<HTMLDivElement>(null);
    const ring3Ref = useRef<HTMLDivElement>(null);
    const buttonRef = useRef<HTMLButtonElement>(null);

    useEffect(() => {
        return () => stopRecording();
    }, []);

    const startRecording = async () => {
        setMatch(null);
        setStatus('Listening...');
        setIsListening(false);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioStreamRef.current = stream;

            const context = new AudioContext({ sampleRate: 8000 });
            audioContextRef.current = context;

            await context.audioWorklet.addModule('/audio-processor.js');
            const workletNode = new AudioWorkletNode(context, 'audio-processor');
            audioWorkletNodeRef.current = workletNode;

            // Setup Analyser for Visualization
            const analyser = context.createAnalyser();
            analyser.fftSize = 1024;
            analyser.smoothingTimeConstant = 0.8;
            analyserRef.current = analyser;

            const ws = new WebSocket('ws://127.0.0.1:8000/ws/audio');
            socketRef.current = ws;

            ws.onopen = () => {
                setIsRecording(true);
                setIsListening(true);
                
                const source = context.createMediaStreamSource(stream);
                source.connect(workletNode);
                source.connect(analyser);

                workletNode.port.onmessage = (event) => {
                    if (ws.readyState === WebSocket.OPEN) ws.send(event.data);
                };

                visualize();
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

    const visualize = () => {
        if (!analyserRef.current) return;

        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteFrequencyData(dataArray);

        // Calculate frequency bands
        let bassSum = 0;
        for (let i = 1; i < 10; i++) bassSum += dataArray[i];
        const bass = bassSum / 9 / 255;

        let midSum = 0;
        for (let i = 10; i < 64; i++) midSum += dataArray[i];
        const mid = midSum / 54 / 255;

        let highSum = 0;
        for (let i = 64; i < 200; i++) highSum += dataArray[i];
        const high = highSum / 136 / 255;

        // Apply transforms
        if (ring1Ref.current) {
            ring1Ref.current.style.transform = `scale(${1 + bass * 0.8})`;
            ring1Ref.current.style.opacity = `${0.4 + bass * 0.6}`;
        }

        if (ring2Ref.current) {
            ring2Ref.current.style.transform = `scale(${1 + mid * 1.4})`;
            ring2Ref.current.style.opacity = `${0.2 + mid * 0.5}`;
        }

        if (ring3Ref.current) {
            ring3Ref.current.style.transform = `scale(${1 + high * 1.8})`;
            ring3Ref.current.style.opacity = `${0.1 + high * 0.4}`;
        }

        if (buttonRef.current) {
            buttonRef.current.style.transform = `scale(${1 + bass * 0.15})`;
            buttonRef.current.style.boxShadow = `0 10px 40px rgba(231, 76, 60, ${0.4 + bass * 0.6})`;
        }

        animationFrameRef.current = requestAnimationFrame(visualize);
    };

    const stopRecording = () => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }

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
            try {
                audioWorkletNodeRef.current.port.close();
            } catch (e) {
                // Ignore
            }
            audioWorkletNodeRef.current = null;
        }
        
        // Reset rings
        if (ring1Ref.current) ring1Ref.current.style.transform = 'scale(1)';
        if (ring2Ref.current) ring2Ref.current.style.transform = 'scale(1)';
        if (ring3Ref.current) ring3Ref.current.style.transform = 'scale(1)';
        if (buttonRef.current) {
            buttonRef.current.style.transform = 'scale(1)';
            buttonRef.current.style.boxShadow = '0 10px 30px rgba(0,0,0,0.2)';
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
                marginBottom: '60px', // More space for the rings
                opacity: status ? 1 : 0,
                transition: 'opacity 0.3s ease',
                zIndex: 10
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

            {/* Record Button Container with Rings */}
            <div style={{ position: 'relative', width: '120px', height: '120px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                
                {/* Visualization Rings */}
                <div ref={ring3Ref} style={{
                    position: 'absolute', width: '100%', height: '100%', borderRadius: '50%',
                    backgroundColor: 'rgba(231, 76, 60, 0.2)',
                    transition: 'transform 0.05s linear, opacity 0.05s linear',
                    pointerEvents: 'none',
                    zIndex: 1
                }} />
                <div ref={ring2Ref} style={{
                    position: 'absolute', width: '100%', height: '100%', borderRadius: '50%',
                    backgroundColor: 'rgba(231, 76, 60, 0.3)',
                    transition: 'transform 0.05s linear, opacity 0.05s linear',
                    pointerEvents: 'none',
                    zIndex: 2
                }} />
                <div ref={ring1Ref} style={{
                    position: 'absolute', width: '100%', height: '100%', borderRadius: '50%',
                    backgroundColor: 'rgba(231, 76, 60, 0.4)',
                    transition: 'transform 0.05s linear, opacity 0.05s linear',
                    pointerEvents: 'none',
                    zIndex: 3
                }} />

                {/* Main Button */}
                <button
                    ref={buttonRef}
                    onClick={handleToggle}
                    style={{
                        width: '120px',
                        height: '120px',
                        borderRadius: '50%',
                        border: 'none',
                        background: isRecording ? '#fff' : '#000',
                        color: isRecording ? '#000' : '#fff',
                        cursor: 'pointer',
                        position: 'relative',
                        zIndex: 10, // Above rings
                        boxShadow: isRecording 
                            ? '0 0 0 2px rgba(231, 76, 60, 0.1)' 
                            : '0 10px 30px rgba(0,0,0,0.2)',
                        transition: 'all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        outline: 'none'
                    }}
                >
                    {isRecording ? (
                        <div style={{ 
                            width: '30px', 
                            height: '30px', 
                            background: '#e74c3c', 
                            borderRadius: '6px',
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
            </div>

            {/* Result Display */}
            <div style={{ 
                marginTop: '60px', 
                textAlign: 'center',
                minHeight: '100px',
                opacity: match && match.match ? 1 : 0,
                transform: match && match.match ? 'translateY(0)' : 'translateY(20px)',
                transition: 'all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1)',
                zIndex: 10
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
        </div>
    );
}