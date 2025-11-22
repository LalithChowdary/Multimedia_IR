'use client';

import { useState, useRef, useEffect } from 'react';

interface SearchResult {
    video_name: string;
    timestamp: string;
    text: string;
    similarity_score: number;
    start_seconds: number;
    end_seconds: number;
}

interface VideoItem {
    filename: string;
    video_url: string;
    thumbnail_url: string;
    size_mb: number;
}

export default function VideoSearch() {
    const [activeTab, setActiveTab] = useState<'search' | 'upload'>('search');
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<SearchResult[]>([]);
    const [videos, setVideos] = useState<VideoItem[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [processingFile, setProcessingFile] = useState<string | null>(null);
    const [progressStatus, setProgressStatus] = useState('');
    const [progressStep, setProgressStep] = useState(0);
    
    const [uploadStatus, setUploadStatus] = useState('');
    const [playingVideo, setPlayingVideo] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Fetch videos on load
    useEffect(() => {
        fetchVideos();
    }, []);

    const fetchVideos = async () => {
        try {
            const response = await fetch('http://127.0.0.1:8000/videos');
            const data = await response.json();
            setVideos(data.videos || []);
        } catch (error) {
            console.error('Error fetching videos:', error);
        }
    };

    const handleSearch = async () => {
        if (!query.trim()) return;
        
        setIsSearching(true);
        try {
            const response = await fetch(`http://127.0.0.1:8000/search_video?query=${encodeURIComponent(query)}`);
            const data = await response.json();
            setResults(data.results || []);
        } catch (error) {
            console.error('Search failed:', error);
        } finally {
            setIsSearching(false);
        }
    };

    const handleDelete = async (filename: string) => {
        if (!confirm(`Are you sure you want to delete "${filename}"?`)) return;

        try {
            const response = await fetch(`http://127.0.0.1:8000/delete_video/${filename}`, {
                method: 'DELETE',
            });

            if (response.ok) {
                // Refresh video list
                fetchVideos();
                // Clear search results if they contain the deleted video
                setResults(prev => prev.filter(r => r.video_name !== filename));
            } else {
                alert('Failed to delete video');
            }
        } catch (error) {
            console.error('Error deleting video:', error);
            alert('Error deleting video');
        }
    };
    useEffect(() => {
        let intervalId: NodeJS.Timeout;

        if (processingFile) {
            intervalId = setInterval(async () => {
                try {
                    const response = await fetch(`http://127.0.0.1:8000/status/${processingFile}`);
                    const data = await response.json();
                    const status = data.status;
                    
                    setProgressStatus(status);

                    // Map status to progress percentage
                    if (status === 'Starting...') setProgressStep(10);
                    else if (status === 'Transcribing...') setProgressStep(40);
                    else if (status === 'Indexing...') setProgressStep(80);
                    else if (status === 'Complete') {
                        setProgressStep(100);
                        setProcessingFile(null); // Stop polling
                        setUploadStatus('Processing complete!');
                        fetchVideos(); // Refresh list
                    } else if (status.startsWith('Error')) {
                        setProcessingFile(null);
                        setUploadStatus(status);
                    }
                } catch (error) {
                    console.error('Error polling status:', error);
                }
            }, 1000);
        }

        return () => {
            if (intervalId) clearInterval(intervalId);
        };
    }, [processingFile]);

    const handleClear = () => {
        setQuery('');
        setResults([]);
    };

    const handleUpload = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!fileInputRef.current?.files?.[0]) return;

        const file = fileInputRef.current.files[0];
        const formData = new FormData();
        formData.append('video_file', file);

        setUploadStatus('Uploading...');
        setProgressStep(0);
        
        try {
            const response = await fetch('http://127.0.0.1:8000/upload_video', {
                method: 'POST',
                body: formData,
            });
            
            if (response.ok) {
                const data = await response.json();
                setUploadStatus('Upload complete. Processing...');
                setProcessingFile(data.filename); // Start polling
                if (fileInputRef.current) fileInputRef.current.value = '';
            } else {
                setUploadStatus('Upload failed.');
            }
        } catch (error) {
            console.error('Upload error:', error);
            setUploadStatus('Error uploading file.');
        }
    };

    return (
        <div style={{ 
            width: '100%', 
            height: '100%', // Fill parent
            maxWidth: '800px',
            margin: '0 auto',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            display: 'flex',
            flexDirection: 'column',
            position: 'relative'
        }}>
            {/* Fixed Header Section */}
            <div style={{
                zIndex: 100,
                backgroundColor: '#fff',
                paddingTop: '10px',
                paddingBottom: '20px',
                marginBottom: '10px',
                borderBottom: '1px solid rgba(0,0,0,0.05)',
                flexShrink: 0 // Don't shrink
            }}>
                {activeTab === 'search' ? (
                    <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                        <div style={{ 
                            flex: 1,
                            display: 'flex', 
                            alignItems: 'center', 
                            background: '#f5f5f7', 
                            borderRadius: '12px',
                            padding: '8px 16px',
                        }}>
                            <span style={{ fontSize: '18px', marginRight: '10px', color: '#888' }}>🔍</span>
                            <input 
                                type="text" 
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                                placeholder="Find moments in videos..." 
                                style={{
                                    flex: 1,
                                    border: 'none',
                                    background: 'transparent',
                                    fontSize: '16px',
                                    outline: 'none',
                                    color: '#333',
                                    padding: '8px 0'
                                }}
                            />
                            {query && !isSearching && (
                                <button
                                    onClick={handleClear}
                                    style={{
                                        background: 'none',
                                        border: 'none',
                                        cursor: 'pointer',
                                        fontSize: '18px',
                                        color: '#999',
                                        padding: '0 8px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        transition: 'color 0.2s ease'
                                    }}
                                    onMouseEnter={(e) => e.currentTarget.style.color = '#333'}
                                    onMouseLeave={(e) => e.currentTarget.style.color = '#999'}
                                    title="Clear search"
                                >
                                    ✕
                                </button>
                            )}
                            {isSearching && <span style={{ fontSize: '12px', color: '#999' }}>Searching...</span>}
                        </div>
                        <button
                            onClick={() => setActiveTab('upload')}
                            style={{
                                background: '#000',
                                color: '#fff',
                                border: 'none',
                                borderRadius: '12px',
                                padding: '0 20px',
                                height: '46px',
                                fontSize: '14px',
                                fontWeight: 600,
                                cursor: 'pointer',
                                whiteSpace: 'nowrap',
                                transition: 'opacity 0.2s ease'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.opacity = '0.8'}
                            onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
                        >
                            + Add Video
                        </button>
                    </div>
                ) : (
                    <div style={{ display: 'flex', alignItems: 'center', height: '46px' }}>
                        <button
                            onClick={() => setActiveTab('search')}
                            style={{
                                background: 'none',
                                border: 'none',
                                fontSize: '16px',
                                fontWeight: 600,
                                color: '#000',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '5px',
                                padding: 0
                            }}
                        >
                            ← Back to Search
                        </button>
                    </div>
                )}
            </div>

            {/* Scrollable Content Area */}
            <div style={{ 
                flex: 1, 
                overflowY: 'auto', 
                paddingBottom: '20px',
                scrollbarWidth: 'thin' // For Firefox
            }}>
            {activeTab === 'search' ? (
                // SEARCH VIEW
                <div style={{ animation: 'fadeIn 0.4s ease' }}>
                    
                    {/* Search Results */}
                    {results.length > 0 ? (
                         <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
                            {results.map((res, idx) => (
                                <div key={idx} style={{ 
                                    padding: '0',
                                    transition: 'transform 0.2s ease'
                                }}>
                                    <div style={{ 
                                        fontSize: '13px', 
                                        color: '#555', 
                                        marginBottom: '10px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'space-between'
                                    }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                            <span style={{ fontWeight: 600, color: '#000' }}>{res.video_name}</span>
                                            <span style={{ 
                                                background: '#e1f5fe', 
                                                color: '#0277bd',
                                                padding: '2px 8px', 
                                                borderRadius: '12px',
                                                fontFamily: 'monospace',
                                                fontSize: '12px'
                                            }}>{res.timestamp}</span>
                                        </div>
                                        <span style={{ fontSize: '12px', color: '#999' }}>
                                            Match: {Math.round(res.similarity_score * 100)}%
                                        </span>
                                    </div>
                                    
                                    {/* Video Player for Result */}
                                    <div style={{
                                        width: '100%',
                                        aspectRatio: '16/9',
                                        backgroundColor: '#000',
                                        borderRadius: '12px',
                                        overflow: 'hidden',
                                        marginBottom: '10px'
                                    }}>
                                        <video
                                            src={`http://127.0.0.1:8000/content/videos/${encodeURIComponent(
                                                videos.find(v => {
                                                    const baseName = v.filename.replace(/\.[^/.]+$/, "");
                                                    return res.video_name.includes(baseName);
                                                })?.filename || res.video_name
                                            )}#t=${res.start_seconds},${res.end_seconds}`}
                                            controls
                                            playsInline
                                            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                        />
                                    </div>

                                    <p style={{ 
                                        margin: 0, 
                                        fontSize: '14px', 
                                        color: '#666', 
                                        lineHeight: '1.5',
                                        fontStyle: 'italic',
                                        paddingLeft: '10px',
                                        borderLeft: '3px solid #eee'
                                    }}>
                                        "{res.text}"
                                    </p>
                                </div>
                            ))}
                        </div>
                    ) : !isSearching && query ? (
                        <div style={{ textAlign: 'center', color: '#aaa', fontSize: '14px', marginTop: '20px' }}>
                            No results found.
                        </div>
                    ) : (
                        // Default Grid View (YouTube Style)
                        <div>
                            <h3 style={{ fontSize: '14px', fontWeight: 600, color: '#666', marginBottom: '20px' }}>All Videos</h3>
                            <div style={{ 
                                display: 'grid', 
                                gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', 
                                gap: '20px' 
                            }}>
                                {videos.map((video, idx) => (
                                    <div key={idx} style={{ cursor: 'pointer' }}>
                                        <div 
                                            onClick={() => setPlayingVideo(video.filename)}
                                            style={{ 
                                                width: '100%', 
                                                aspectRatio: '16/9', 
                                                backgroundColor: '#000', 
                                                borderRadius: '12px', 
                                                overflow: 'hidden', 
                                                marginBottom: '10px',
                                                position: 'relative'
                                            }}
                                        >
                                            {playingVideo === video.filename ? (
                                                <video
                                                    src={`http://127.0.0.1:8000${video.video_url}`}
                                                    controls
                                                    autoPlay
                                                    muted
                                                    playsInline
                                                    style={{ width: '100%', height: '100%', objectFit: 'contain', backgroundColor: '#000' }}
                                                />
                                            ) : (
                                                <>
                                                    {video.thumbnail_url ? (
                                                        <img 
                                                            src={`http://127.0.0.1:8000${video.thumbnail_url}`} 
                                                            alt={video.filename}
                                                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                                        />
                                                    ) : (
                                                        <div style={{ 
                                                            width: '100%', 
                                                            height: '100%', 
                                                            display: 'flex', 
                                                            alignItems: 'center', 
                                                            justifyContent: 'center',
                                                            color: '#ccc'
                                                        }}>
                                                            No Preview
                                                        </div>
                                                    )}
                                                    <div style={{
                                                        position: 'absolute',
                                                        top: '50%',
                                                        left: '50%',
                                                        transform: 'translate(-50%, -50%)',
                                                        width: '40px',
                                                        height: '40px',
                                                        borderRadius: '50%',
                                                        backgroundColor: 'rgba(0,0,0,0.6)',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'center',
                                                        color: 'white',
                                                        fontSize: '20px'
                                                    }}>
                                                        ▶
                                                    </div>
                                                    <div style={{
                                                        position: 'absolute',
                                                        bottom: '8px',
                                                        right: '8px',
                                                        background: 'rgba(0,0,0,0.7)',
                                                        color: 'white',
                                                        fontSize: '10px',
                                                        padding: '2px 6px',
                                                        borderRadius: '4px',
                                                        fontWeight: 600
                                                    }}>
                                                        VIDEO
                                                    </div>
                                                </>
                                            )}
                                        </div>
                                        <div style={{ fontSize: '14px', fontWeight: 600, color: '#222', marginBottom: '4px', lineHeight: '1.3' }}>
                                            {video.filename.replace(/\.[^/.]+$/, "")}
                                        </div>
                                        <div style={{ fontSize: '12px', color: '#888' }}>
                                            {video.size_mb} MB
                                        </div>
                                    </div>
                                ))}
                                {videos.length === 0 && (
                                    <div style={{ gridColumn: '1 / -1', textAlign: 'center', color: '#999', padding: '40px' }}>
                                        No videos available.
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            ) : (
                // UPLOAD VIEW
                <div style={{ 
                    textAlign: 'center', 
                    animation: 'fadeIn 0.4s ease',
                    marginTop: '20px'
                }}>
                    <div 
                        onClick={() => fileInputRef.current?.click()}
                        style={{ 
                            border: '1px dashed #ccc', 
                            borderRadius: '12px', 
                            padding: '60px 20px',
                            cursor: 'pointer',
                            backgroundColor: '#fafafa',
                            transition: 'background 0.2s ease'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f0f0f0'}
                        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#fafafa'}
                    >
                        <p style={{ fontSize: '24px', marginBottom: '10px' }}>📄</p>
                        <p style={{ margin: 0, fontSize: '15px', color: '#666' }}>
                            Click to select a video file
                        </p>
                        <input 
                            type="file" 
                            ref={fileInputRef}
                            accept="video/*"
                            style={{ display: 'none' }}
                            onChange={handleUpload}
                        />
                    </div>
                    
                    {/* Progress Bar */}
                    {(processingFile || uploadStatus) && (
                        <div style={{ marginTop: '30px', maxWidth: '400px', margin: '30px auto 0' }}>
                            <div style={{ 
                                display: 'flex', 
                                justifyContent: 'space-between', 
                                marginBottom: '8px',
                                fontSize: '13px',
                                fontWeight: 500,
                                color: '#333'
                            }}>
                                <span>{progressStatus || uploadStatus}</span>
                                {processingFile && <span>{progressStep}%</span>}
                            </div>
                            
                            <div style={{ 
                                width: '100%', 
                                height: '6px', 
                                backgroundColor: '#eee', 
                                borderRadius: '3px',
                                overflow: 'hidden'
                            }}>
                                <div style={{ 
                                    width: `${processingFile ? progressStep : uploadStatus.includes('complete') ? 100 : 0}%`, 
                                    height: '100%', 
                                    backgroundColor: uploadStatus.includes('failed') ? '#e74c3c' : '#000',
                                    transition: 'width 0.5s ease'
                                }} />
                            </div>
                            
                            {processingFile && (
                                <p style={{ fontSize: '12px', color: '#888', marginTop: '10px' }}>
                                    Processing: {processingFile}
                                </p>
                            )}
                        </div>
                    )}
                </div>
            )}
            </div>
            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(5px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            `}</style>
        </div>
    );
}
