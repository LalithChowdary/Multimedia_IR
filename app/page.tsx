'use client';

import { useState } from 'react';
import AudioRecorder from "./components/AudioRecorder";
import VideoSearch from "./components/VideoSearch";

export default function Home() {
  const [mode, setMode] = useState<'music' | 'video'>('music');

  return (
    <main style={{
      height: "100vh", // Fixed height
      backgroundColor: "#ffffff",
      color: "#111",
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden' // No body scroll
    }}>
      {/* Minimal Header */}
      <header style={{
        padding: '30px 40px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ fontWeight: 700, fontSize: '18px', letterSpacing: '-0.5px' }}>
          Audio<span style={{ color: '#999' }}>Vision</span>
        </div>
        
        <nav style={{ 
          background: '#f4f4f4', 
          padding: '4px', 
          borderRadius: '30px',
          display: 'flex',
          gap: '5px'
        }}>
          <button
            onClick={() => setMode('music')}
            style={{
              background: mode === 'music' ? '#fff' : 'transparent',
              border: 'none',
              color: mode === 'music' ? '#000' : '#777',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              padding: '8px 20px',
              borderRadius: '25px',
              boxShadow: mode === 'music' ? '0 2px 8px rgba(0,0,0,0.08)' : 'none',
              transition: 'all 0.2s ease'
            }}
          >
            Music
          </button>
          <button
            onClick={() => setMode('video')}
            style={{
              background: mode === 'video' ? '#fff' : 'transparent',
              border: 'none',
              color: mode === 'video' ? '#000' : '#777',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer',
              padding: '8px 20px',
              borderRadius: '25px',
              boxShadow: mode === 'video' ? '0 2px 8px rgba(0,0,0,0.08)' : 'none',
              transition: 'all 0.2s ease'
            }}
          >
            Video
          </button>
        </nav>
      </header>

      {/* Main Content */}
      <div style={{
        flex: 1,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: '20px',
        overflow: 'hidden' // Prevent container scroll
      }}>
        <div style={{ 
          width: '100%', 
          maxWidth: '800px',
          height: '100%', // Fill height for internal scrolling
          position: 'relative'
        }}>
            {mode === 'music' ? (
                <div style={{ 
                    animation: 'fadeIn 0.4s ease',
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                }}>
                    <AudioRecorder />
                </div>
            ) : (
                <div style={{ 
                    animation: 'fadeIn 0.4s ease',
                    height: '100%'
                }}>
                    <VideoSearch />
                </div>
            )}
        </div>
      </div>

      <footer style={{
        textAlign: 'center',
        padding: '30px',
        fontSize: '12px',
        color: '#ccc'
      }}>
        Multimedia Systems &copy; 2025
      </footer>

      <style jsx>{`
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </main>
  );
}
