'use client';

import { useState, useRef } from 'react';

interface SearchResult {
    video_name: string;
    timestamp: string;
    text: string;
    similarity_score: number;
}

export default function VideoSearch() {
    const [activeTab, setActiveTab] = useState<'search' | 'upload'>('search');
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<SearchResult[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [uploadStatus, setUploadStatus] = useState('');
    const fileInputRef = useRef<HTMLInputElement>(null);

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
                setUploadStatus('Upload complete. Processing...');
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
            maxWidth: '700px',
            margin: '0 auto',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        }}>
            {/* Minimal Tab Switcher */}
            <div style={{ 
                display: 'flex', 
                justifyContent: 'center', 
                gap: '40px', 
                marginBottom: '40px' 
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

            {activeTab === 'search' ? (
                // SEARCH VIEW
                <div style={{ animation: 'fadeIn 0.4s ease' }}>
                    <div style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        background: '#fff',
                        border: '1px solid #eaeaea',
                        borderRadius: '12px',
                        padding: '8px 16px',
                        boxShadow: '0 2px 10px rgba(0,0,0,0.02)',
                        marginBottom: '30px'
                    }}>
                        <span style={{ fontSize: '18px', marginRight: '10px' }}>🔍</span>
                        <input 
                            type="text" 
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                            placeholder="Find moments in videos..." 
                            style={{
                                flex: 1,
                                border: 'none',
                                fontSize: '16px',
                                outline: 'none',
                                color: '#333',
                                padding: '10px 0'
                            }}
                        />
                        {isSearching && <span style={{ fontSize: '12px', color: '#999' }}>Searching...</span>}
                    </div>

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
                        {results.length === 0 && !isSearching && query && (
                            <div style={{ textAlign: 'center', color: '#aaa', fontSize: '14px', marginTop: '20px' }}>
                                No results found.
                            </div>
                        )}
                    </div>
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
            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(5px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            `}</style>
        </div>
    );
}