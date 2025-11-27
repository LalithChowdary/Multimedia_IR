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
    const [isUploading, setIsUploading] = useState(false);
    const [showPlayer, setShowPlayer] = useState(false);
    const [showSuccessAnimation, setShowSuccessAnimation] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    
    const fileInputRef = useRef<HTMLInputElement>(null);
    const audioRef = useRef<HTMLAudioElement>(null);

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
                    if (result.confirmed) {
                        setStatus('Match Confirmed');
                        // Trigger success animation
                        setShowSuccessAnimation(true);
                        // Auto-stop recording and show player after animation
                        setTimeout(() => {
                            stopRecording();
                            setShowSuccessAnimation(false);
                            setShowPlayer(true);
                        }, 1000); // 1.0s for animation
                    } else {
                        setStatus('Analyzing...');
                    }
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

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        // Validate file type
        const validTypes = ['audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/ogg', 'audio/m4a', 'audio/x-m4a'];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(mp3|wav|ogg|m4a)$/i)) {
            setStatus('Invalid file type. Please upload an audio file.');
            setTimeout(() => setStatus(''), 3000);
            return;
        }

        setIsUploading(true);
        setMatch(null);
        setStatus('Analyzing audio clip...');

        try {
            const formData = new FormData();
            formData.append('audio_file', file);

            const response = await fetch('http://127.0.0.1:8000/identify', {
                method: 'POST',
                body: formData,
            });

            const result: MatchResult = await response.json();

            if (result.match) {
                setMatch(result);
                setStatus('Match Found!');
                // Trigger success animation
                setShowSuccessAnimation(true);
                // Show player after animation
                setTimeout(() => {
                    setShowSuccessAnimation(false);
                    setShowPlayer(true);
                }, 1000); // 1.0s for animation
            } else {
                setStatus('No match found');
                setTimeout(() => setStatus(''), 3000);
            }
        } catch (error) {
            console.error('Upload error:', error);
            setStatus('Error analyzing file');
            setTimeout(() => setStatus(''), 3000);
        } finally {
            setIsUploading(false);
            // Reset file input
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    // Audio Player Controls
    const togglePlayPause = () => {
        if (audioRef.current) {
            if (isPlaying) {
                audioRef.current.pause();
            } else {
                audioRef.current.play();
            }
            setIsPlaying(!isPlaying);
        }
    };

    const handleTimeUpdate = () => {
        if (audioRef.current) {
            setCurrentTime(audioRef.current.currentTime);
        }
    };

    const handleLoadedMetadata = () => {
        if (audioRef.current) {
            setDuration(audioRef.current.duration);
        }
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newTime = parseFloat(e.target.value);
        if (audioRef.current) {
            audioRef.current.currentTime = newTime;
            setCurrentTime(newTime);
        }
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const getSongFileName = (songName: string) => {
        // Map song names to their file names
        // Handle cases like "Night Change" -> "Night Change.mp3"
        return songName;
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

            {/* Upload Button */}
            <div style={{ 
                marginTop: '30px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '10px'
            }}>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*,.mp3,.wav,.ogg,.m4a"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                />
                <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isRecording || isUploading}
                    style={{
                        padding: '12px 28px',
                        borderRadius: '25px',
                        border: '1.5px solid #e0e0e0',
                        background: isUploading ? '#f5f5f5' : '#fff',
                        color: isUploading ? '#999' : '#333',
                        fontSize: '13px',
                        fontWeight: 600,
                        cursor: isRecording || isUploading ? 'not-allowed' : 'pointer',
                        transition: 'all 0.2s ease',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        opacity: isRecording ? 0.5 : 1,
                        boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
                        outline: 'none'
                    }}
                    onMouseEnter={(e) => {
                        if (!isRecording && !isUploading) {
                            e.currentTarget.style.borderColor = '#000';
                            e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.08)';
                        }
                    }}
                    onMouseLeave={(e) => {
                        if (!isRecording && !isUploading) {
                            e.currentTarget.style.borderColor = '#e0e0e0';
                            e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.04)';
                        }
                    }}
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    {isUploading ? 'Analyzing...' : 'Upload Audio Clip'}
                </button>
                <p style={{
                    margin: 0,
                    fontSize: '11px',
                    color: '#aaa',
                    textAlign: 'center'
                }}>
                    Upload a 5-10 second audio clip
                </p>
            </div>

            {/* Success Animation Overlay - Seamless Reveal */}
            {showSuccessAnimation && match && (
                <div style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    zIndex: 9999,
                    pointerEvents: 'none',
                    backgroundImage: `url(http://127.0.0.1:8000/content/audio_thumbnails/${encodeURIComponent(getSongFileName(match.song_id || ''))}.png)`,
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                    animation: 'revealAndBlur 1s cubic-bezier(0.25, 1, 0.5, 1) forwards'
                }} />
            )}

            {/* Music Player Overlay */}
            {showPlayer && match && (
                <div style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    zIndex: 10000,
                    animation: 'fadeIn 0.5s ease'
                }}>
                    {/* Blurred Background */}
                    <div style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundImage: `url(http://127.0.0.1:8000/content/audio_thumbnails/${encodeURIComponent(getSongFileName(match.song_id || ''))}.png)`,
                        backgroundSize: 'cover',
                        backgroundPosition: 'center',
                        filter: 'blur(60px) brightness(0.7)',
                        transform: 'scale(1.2)'
                    }} />

                    {/* Gradient Overlay */}
                    <div style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: 'linear-gradient(180deg, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.8) 100%)'
                    }} />

                    {/* Close Button */}
                    <button
                        onClick={() => {
                            setShowPlayer(false);
                            if (audioRef.current) {
                                audioRef.current.pause();
                                setIsPlaying(false);
                            }
                        }}
                        style={{
                            position: 'absolute',
                            top: '30px',
                            right: '30px',
                            background: 'rgba(255,255,255,0.1)',
                            border: 'none',
                            borderRadius: '50%',
                            width: '40px',
                            height: '40px',
                            color: '#fff',
                            fontSize: '24px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            zIndex: 10,
                            transition: 'background 0.2s ease'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.2)'}
                        onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
                    >
                        ×
                    </button>

                    {/* Player Content */}
                    <div style={{
                        position: 'relative',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        height: '100%',
                        padding: '40px',
                        zIndex: 1
                    }}>
                        {/* Album Art */}
                        <div style={{
                            marginBottom: '40px',
                            animation: 'scaleIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)'
                        }}>
                            <img
                                src={`http://127.0.0.1:8000/content/audio_thumbnails/${encodeURIComponent(getSongFileName(match.song_id || ''))}.png`}
                                alt={match.song_id}
                                style={{
                                    width: '400px',
                                    height: '400px',
                                    borderRadius: '12px',
                                    boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
                                    objectFit: 'cover'
                                }}
                            />
                        </div>

                        {/* Song Name */}
                        <h2 style={{
                            color: '#fff',
                            fontSize: '36px',
                            fontWeight: 700,
                            margin: '0 0 40px 0',
                            textAlign: 'center',
                            letterSpacing: '-0.5px'
                        }}>
                            {match.song_id}
                        </h2>

                        {/* Progress Bar */}
                        <div style={{
                            width: '100%',
                            maxWidth: '600px',
                            marginBottom: '20px'
                        }}>
                            <input
                                type="range"
                                min="0"
                                max={duration || 0}
                                value={currentTime}
                                onChange={handleSeek}
                                style={{
                                    width: '100%',
                                    height: '6px',
                                    borderRadius: '3px',
                                    outline: 'none',
                                    appearance: 'none',
                                    background: `linear-gradient(to right, #fff ${(currentTime / duration) * 100}%, rgba(255,255,255,0.3) ${(currentTime / duration) * 100}%)`,
                                    cursor: 'pointer'
                                }}
                            />
                            <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                marginTop: '8px',
                                color: 'rgba(255,255,255,0.7)',
                                fontSize: '13px'
                            }}>
                                <span>{formatTime(currentTime)}</span>
                                <span>-{formatTime(duration - currentTime)}</span>
                            </div>
                        </div>

                        {/* Playback Controls */}
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '30px'
                        }}>
                            <button
                                onClick={togglePlayPause}
                                style={{
                                    width: '70px',
                                    height: '70px',
                                    borderRadius: '50%',
                                    border: 'none',
                                    background: '#fff',
                                    color: '#000',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontSize: '24px',
                                    boxShadow: '0 8px 20px rgba(0,0,0,0.3)',
                                    transition: 'transform 0.2s ease'
                                }}
                                onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
                                onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                            >
                                {isPlaying ? '❚❚' : '▶'}
                            </button>
                        </div>

                        {/* Hidden Audio Element */}
                        <audio
                            ref={audioRef}
                            src={`http://127.0.0.1:8000/content/audio/${encodeURIComponent(getSongFileName(match.song_id || ''))}.mp3`}
                            onTimeUpdate={handleTimeUpdate}
                            onLoadedMetadata={handleLoadedMetadata}
                            onEnded={() => setIsPlaying(false)}
                        />
                    </div>
                </div>
            )}

            {/* CSS Animations */}
            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                @keyframes scaleIn {
                    from { 
                        transform: scale(0.8);
                        opacity: 0;
                    }
                    to { 
                        transform: scale(1);
                        opacity: 1;
                    }
                }
                @keyframes revealAndBlur {
                    0% {
                        clip-path: circle(0px at center);
                        filter: blur(0px) brightness(1);
                        transform: scale(1);
                    }
                    10% {
                        clip-path: circle(20px at center);
                    }
                    100% {
                        clip-path: circle(150vmax at center);
                        filter: blur(60px) brightness(0.7);
                        transform: scale(1.2);
                    }
                }
                input[type="range"]::-webkit-slider-thumb {
                    appearance: none;
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                    background: #fff;
                    cursor: pointer;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                }
                input[type="range"]::-moz-range-thumb {
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                    background: #fff;
                    cursor: pointer;
                    border: none;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                }
            `}</style>
        </div>
    );
}