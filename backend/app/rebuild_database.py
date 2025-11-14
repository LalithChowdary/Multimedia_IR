"""
Rebuild database with optimized parameters for real-world matching.

This script will:
1. Clear the old database
2. Re-fingerprint all songs with new parameters
3. Save the new database
4. Test matching capability

Usage:
    python rebuild_database.py <music_directory>
"""

import os
import sys
from pathlib import Path
from fingerprint import generate_fingerprints
from database import FingerprintDB

# Supported audio formats
SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma'}

def find_audio_files(directory):
    """
    Recursively find all audio files in directory.
    """
    audio_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in SUPPORTED_FORMATS:
                full_path = os.path.join(root, file)
                audio_files.append(full_path)
    
    return audio_files


def rebuild_database(music_dir, max_songs=None):
    """
    Rebuild the entire fingerprint database.
    
    Args:
        music_dir: Directory containing music files
        max_songs: Maximum number of songs to process (None = all)
    """
    print("="*60)
    print("DATABASE REBUILD WITH OPTIMIZED PARAMETERS")
    print("="*60)
    
    # Find all audio files
    print(f"\nScanning directory: {music_dir}")
    audio_files = find_audio_files(music_dir)
    
    if not audio_files:
        print(f"❌ No audio files found in {music_dir}")
        return
    
    print(f"✓ Found {len(audio_files)} audio files")
    
    if max_songs:
        audio_files = audio_files[:max_songs]
        print(f"  (Processing first {max_songs} songs)")
    
    # Initialize database
    db = FingerprintDB()
    
    print("\n" + "="*60)
    print("FINGERPRINTING SONGS")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"\n[{i}/{len(audio_files)}] Processing: {os.path.basename(audio_file)}")
            
            # Generate fingerprints
            fingerprints = generate_fingerprints(audio_file)
            
            if len(fingerprints) > 0:
                # Add to database
                db.add_song(audio_file, fingerprints)
                success_count += 1
            else:
                print(f"  ⚠️  No fingerprints generated - skipping")
                fail_count += 1
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            fail_count += 1
    
    # Save database
    print("\n" + "="*60)
    print("SAVING DATABASE")
    print("="*60)
    
    db.save()
    
    # Print statistics
    print("\n" + "="*60)
    print("REBUILD COMPLETE")
    print("="*60)
    
    print(f"\n✓ Successfully processed: {success_count} songs")
    if fail_count > 0:
        print(f"✗ Failed: {fail_count} songs")
    
    stats = db.get_stats()
    
    print(f"\n📊 Database Statistics:")
    print(f"   Total fingerprints: {stats['total_fingerprints']:,}")
    print(f"   Unique hashes: {stats['unique_hashes']:,}")
    print(f"   Songs in database: {stats['total_songs']}")
    print(f"   Avg fingerprints per hash: {stats['avg_fingerprints_per_hash']:.1f}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR TESTING")
    print("="*60)
    
    print("\n1. Test with a song from the database:")
    print(f"   python debug_matching.py \"{audio_files[0]}\"")
    
    print("\n2. Record a song from speakers and test:")
    print(f"   python debug_matching.py \"{audio_files[0]}\" recorded.wav")
    
    print("\n3. Adjust parameters if needed based on test results")
    print("   - Edit fingerprint.py constants")
    print("   - Rebuild database")
    
    return db


def quick_test(db, test_file):
    """
    Quick test of database matching.
    """
    print("\n" + "="*60)
    print("QUICK MATCHING TEST")
    print("="*60)
    
    print(f"\nGenerating fingerprints for: {test_file}")
    from fingerprint import generate_fingerprints
    test_fps = generate_fingerprints(test_file)
    
    print(f"Generated {len(test_fps)} fingerprints")
    
    print("\nSearching database...")
    matches = db.get_matches(test_fps)
    
    if matches:
        print(f"\n✅ Found {len(matches)} match(es)!")
        for i, match in enumerate(matches[:3], 1):
            song_id, score, offset = match
            print(f"\n  Match #{i}:")
            print(f"    Song: {os.path.basename(song_id)}")
            print(f"    Confidence score: {score}")
            print(f"    Time offset: {offset}")
    else:
        print("\n❌ No matches found")
        print("This shouldn't happen with the same file!")
        print("Check your parameter settings.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rebuild_database.py <music_directory> [max_songs]")
        print("\nExample:")
        print("  python rebuild_database.py ~/Music")
        print("  python rebuild_database.py ~/Music 100  # Process first 100 songs")
        sys.exit(1)
    
    music_dir = sys.argv[1]
    max_songs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not os.path.isdir(music_dir):
        print(f"Error: {music_dir} is not a directory")
        sys.exit(1)
    
    # Rebuild database
    db = rebuild_database(music_dir, max_songs)
    
    # Ask if user wants to test
    print("\n" + "="*60)
    print("Would you like to run a quick test? (y/n)")
    response = input("> ").strip().lower()
    
    if response == 'y':
        # Find first audio file for testing
        audio_files = find_audio_files(music_dir)
        if audio_files:
            quick_test(db, audio_files[0])
    
    print("\n✅ Database rebuild complete!")
    print("You can now test with captured audio using:")
    print("  python debug_matching.py <original> <captured>")