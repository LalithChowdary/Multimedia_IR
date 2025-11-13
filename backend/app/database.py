import pickle
from collections import defaultdict

# --- Constants ---
DB_PATH = 'backend/database/fingerprints.db'

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

    def get_matches(self, fingerprints: set, threshold: int = 5):
        """
        Finds songs that match a given set of fingerprints from a sample clip.

        Args:
            fingerprints: A set of fingerprints from the sample clip.
            threshold: The minimum number of matching fingerprints to be considered a match.

        Returns:
            A list of (song_id, confidence_score) tuples, sorted by confidence.
        """
        # This dictionary will store potential matches and their time offsets.
        # { song_id: { time_offset_diff: count } }
        matches = defaultdict(lambda: defaultdict(int))

        # For each fingerprint from the sample, find matching fingerprints in the DB.
        for hash_val, (_, sample_time) in fingerprints:
            if hash_val in self.db:
                for song_id, db_time in self.db[hash_val]:
                    # The key insight for matching:
                    # If the sample is from a song in our DB, the *difference*
                    # in time offsets should be consistent.
                    time_offset_diff = db_time - sample_time
                    matches[song_id][time_offset_diff] += 1
        
        # The confidence score for a song is the maximum number of fingerprints
        # that align at a specific time offset.
        results = []
        for song_id, offset_counts in matches.items():
            if offset_counts:
                max_confidence = max(offset_counts.values())
                if max_confidence >= threshold:
                    results.append((song_id, max_confidence))

        # Sort results by confidence score in descending order.
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
