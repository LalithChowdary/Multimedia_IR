import pickle
import os
from collections import defaultdict
import numpy as np

# --- Constants ---
# Get the absolute path to the database directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'database', 'fingerprints.db')

# Shazam algorithm matching parameters
MIN_MATCH_THRESHOLD = 5  # Minimum aligned hashes for a valid match
CLUSTER_TOLERANCE = 2     # Tolerance for time offset clustering (frames)

class FingerprintDB:
    """
    A simple in-memory database for storing and retrieving audio fingerprints.
    
    The database is a dictionary where:
    - The keys are the fingerprint hashes.
    - The values are a list of tuples, where each tuple is (song_id, time_offset).
    """
    def __init__(self):
        self.db = defaultdict(list)

    def add_song(self, song_id: str, fingerprints: set):
        """
        Adds a song's fingerprints to the database.
        
        Args:
            song_id: A unique identifier for the song (e.g., the filename).
            fingerprints: A set of (hash, (song_id, time_offset)) tuples.
        """
        for hash_val, (s_id, time_offset) in fingerprints:
            self.db[hash_val].append((song_id, time_offset))

    def get_matches(self, fingerprints: set, threshold: int = MIN_MATCH_THRESHOLD):
        """
        Find matching songs using Shazam's scatterplot histogram technique.
        
        Implementation of the algorithm described in Wang's paper:
        1. For each hash in sample, find matching hashes in database
        2. Create time-pair scatterplot for each song
        3. Calculate delta_t histogram (offset differences)
        4. Find diagonal line (cluster) in scatterplot
        5. Score is the size of the largest cluster
        
        Args:
            fingerprints: Set of (hash, (song_id, time_offset)) tuples from sample
            threshold: Minimum number of aligned hashes for valid match
        
        Returns:
            List of (song_id, confidence_score, offset) tuples, sorted by confidence
        """
        # Dictionary to store time pairs for each song
        # Structure: {song_id: [(sample_time, db_time), ...]}
        time_pairs = defaultdict(list)
        
        # Step 1: Find all matching hashes and collect time pairs
        for hash_val, (_, sample_time) in fingerprints:
            if hash_val in self.db:
                for song_id, db_time in self.db[hash_val]:
                    time_pairs[song_id].append((sample_time, db_time))
        
        # Step 2: For each song, analyze the time-pair scatterplot
        results = []
        
        for song_id, pairs in time_pairs.items():
            if len(pairs) < threshold:
                continue  # Not enough matches to consider
            
            # Step 3: Calculate delta_t for each time pair
            # delta_t = db_time - sample_time (should be constant for matches)
            deltas = np.array([db_t - sample_t for sample_t, db_t in pairs])
            
            # Step 4: Find the largest cluster using histogram
            # Sort deltas for efficient cluster detection
            sorted_deltas = np.sort(deltas)
            
            # Find largest cluster of similar delta values
            best_cluster_size = 0
            best_offset = 0
            
            i = 0
            while i < len(sorted_deltas):
                # Count consecutive deltas within tolerance
                current_delta = sorted_deltas[i]
                cluster_size = 1
                j = i + 1
                
                while j < len(sorted_deltas):
                    if abs(sorted_deltas[j] - current_delta) <= CLUSTER_TOLERANCE:
                        cluster_size += 1
                        j += 1
                    else:
                        break
                
                # Update best cluster if this is larger
                if cluster_size > best_cluster_size:
                    best_cluster_size = cluster_size
                    best_offset = int(current_delta)
                
                i = j if j > i + 1 else i + 1
            
            # Only add if cluster meets threshold
            if best_cluster_size >= threshold:
                results.append((song_id, best_cluster_size, best_offset))
        
        # Sort by confidence (cluster size) descending
        return sorted(results, key=lambda x: x[1], reverse=True)

    def save(self):
        """Saves the fingerprint database to a file."""
        with open(DB_PATH, 'wb') as f:
            pickle.dump(self.db, f)

    def load(self):
        """Loads the fingerprint database from a file."""
        try:
            with open(DB_PATH, 'rb') as f:
                self.db = pickle.load(f)
        except FileNotFoundError:
            print("Database file not found. Starting with an empty database.")
            self.db = defaultdict(list)

    def get_stats(self):
        """Returns statistics about the fingerprint database."""
        total_fingerprints = sum(len(matches) for matches in self.db.values())
        unique_hashes = len(self.db)
        
        # Count songs (unique song_ids across all fingerprints)
        songs = set()
        for matches in self.db.values():
            for song_id, _ in matches:
                songs.add(song_id)
        
        return {
            'total_fingerprints': total_fingerprints,
            'unique_hashes': unique_hashes,
            'total_songs': len(songs),
            'avg_fingerprints_per_hash': total_fingerprints / unique_hashes if unique_hashes > 0 else 0
        }

if __name__ == '__main__':
    # Example usage:
    db = FingerprintDB()
    
    # 1. Add a "song" to the database (using dummy data)
    # In a real scenario, these would come from generate_fingerprints()
    dummy_fingerprints = {
        (12345, ('song1', 10)),
        (67890, ('song1', 25)),
        (11111, ('song1', 40)),
        (22222, ('song2', 15)),
    }
    db.add_song('song1', dummy_fingerprints)
    db.save()
    print("Database saved.")

    # 2. Load the database
    db_loaded = FingerprintDB()
    db_loaded.load()
    print("Database loaded.")

    # 3. Find matches for a "sample"
    # This sample has two fingerprints that match song1 with a consistent time offset.
    sample_fingerprints = {
        (12345, ('sample', 5)),  # Matches song1. Offset diff = 10 - 5 = 5
        (67890, ('sample', 20)), # Matches song1. Offset diff = 25 - 20 = 5
        (99999, ('sample', 30)), # No match
    }
    
    matches = db_loaded.get_matches(sample_fingerprints, threshold=2)
    print(f"Found matches: {matches}")
