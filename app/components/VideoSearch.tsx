'use client';

import { useState, useRef, useEffect } from 'react';

interface SearchResult {
    video_name: string;
    timestamp: string;
    text: string;
    similarity_score: number;
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
    const [uploadStatus, setUploadStatus] = useState('');
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

    const handleUpload = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!fileInputRef.current?.files?.[0]) return;

        const file = fileInputRef.current.files[0];
        const formData = new FormData();
        formData.append('video_file', file);

        setUploadStatus('Uploading...');
        
        try {
            const response = await fetch('http://127.0.0.1:8000/upload_video', {
                method: 'POST',
                body: formData,
            });
            
            if (response.ok) {
                setUploadStatus('Upload complete. Processing in background...');
                if (fileInputRef.current) fileInputRef.current.value = '';
                // Refresh video list after a delay
                setTimeout(fetchVideos, 2000);
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
                {/* Minimal Tab Switcher */}
                <div style={{ 
                    display: 'flex', 
                    justifyContent: 'center', 
                    gap: '40px', 
                    marginBottom: '20px' 
                }}>
                    <button 
                        onClick={() => setActiveTab('search')}
                        style={{
                            background: 'none',
                            border: 'none',
                            fontSize: '14px',
                            fontWeight: 600,
                            color: activeTab === 'search' ? '#000' : '#999',
                            cursor: 'pointer',
                            paddingBottom: '5px',
                            borderBottom: activeTab === 'search' ? '2px solid #000' : '2px solid transparent',
                            transition: 'all 0.2s ease'
                        }}
                    >
                        Search
                    </button>
                    <button 
                        onClick={() => setActiveTab('upload')}
                        style={{
                            background: 'none',
                            border: 'none',
                            fontSize: '14px',
                            fontWeight: 600,
                            color: activeTab === 'upload' ? '#000' : '#999',
                            cursor: 'pointer',
                            paddingBottom: '5px',
                            borderBottom: activeTab === 'upload' ? '2px solid #000' : '2px solid transparent',
                            transition: 'all 0.2s ease'
                        }}
                    >
                        Upload
                    </button>
                </div>

                {activeTab === 'search' && (
                    <div style={{ 
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
                        {isSearching && <span style={{ fontSize: '12px', color: '#999' }}>Searching...</span>}
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
                         <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                            {results.map((res, idx) => (
                                <div key={idx} style={{ 
                                    padding: '0',
                                    transition: 'transform 0.2s ease'
                                }}>
                                    <div style={{ 
                                        fontSize: '12px', 
                                        color: '#888', 
                                        marginBottom: '6px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '8px'
                                    }}>
                                        <span style={{ fontWeight: 600, color: '#000' }}>{res.video_name}</span>
                                        <span>•</span>
                                        <span style={{ 
                                            background: '#f0f0f0', 
                                            padding: '2px 6px', 
                                            borderRadius: '4px',
                                            fontFamily: 'monospace'
                                        }}>{res.timestamp}</span>
                                    </div>
                                    <p style={{ 
                                        margin: 0, 
                                        fontSize: '15px', 
                                        color: '#444', 
                                        lineHeight: '1.6' 
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
                                        <div style={{ 
                                            width: '100%', 
                                            aspectRatio: '16/9', 
                                            backgroundColor: '#eee', 
                                            borderRadius: '12px', 
                                            overflow: 'hidden',
                                            marginBottom: '10px',
                                            position: 'relative'
                                        }}>
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
                    {uploadStatus && (
                        <p style={{ 
                            marginTop: '20px', 
                            fontSize: '13px',
                            color: uploadStatus.includes('failed') ? '#e74c3c' : '#2ecc71' 
                        }}>
                            {uploadStatus}
                        </p>
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
