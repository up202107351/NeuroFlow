import sqlite3
from datetime import datetime
import os
from pathlib import Path
import hashlib
import secrets

DATABASE_DIR = Path('./app_data')
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_NAME = DATABASE_DIR / 'neuroflow_history.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row # Access columns by name
    return conn

def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            last_login TIMESTAMP,
            remember_token TEXT
        )
    ''')

    # Session Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_type TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_seconds INTEGER,
            target_metric_name TEXT, -- e.g., 'Relaxation', 'Focus'
            notes TEXT, -- Optional user notes
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')

    # Session Metrics (Granular Data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            prediction_label TEXT, -- 'Relaxed', 'Neutral', 'Focused', etc.
            is_on_target BOOLEAN, -- True if meeting session goal (e.g., prediction == 'Relaxed')
            raw_score REAL, -- Optional: if your classifier outputs a continuous score
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')

    # Session Summary (Aggregated Data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_summary (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER UNIQUE NOT NULL, -- Ensures one summary per session
            time_on_target_seconds INTEGER,
            percent_on_target REAL,
            average_raw_score REAL, -- If applicable
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized/checked at {DATABASE_NAME}")

# --- User authentication functions ---

def hash_password(password, salt=None):
    """Hash password with salt using SHA-256"""
    if salt is None:
        # Generate a new random salt
        salt = secrets.token_hex(16)
    
    # Create the hash
    pw_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return pw_hash, salt

def register_user(username, password):
    """Register a new user with hashed password"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if username already exists
    cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return False, "Username already exists"
    
    # Hash the password
    password_hash, salt = hash_password(password)
    created_at = datetime.now()
    
    # Insert new user
    try:
        cursor.execute('''
            INSERT INTO users (username, password_hash, salt, created_at)
            VALUES (?, ?, ?, ?)
        ''', (username, password_hash, salt, created_at))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, user_id
    except Exception as e:
        conn.close()
        return False, str(e)

def authenticate_user(username, password):
    """Authenticate user by checking hashed password"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get user data
    cursor.execute('''
        SELECT user_id, password_hash, salt
        FROM users 
        WHERE username = ?
    ''', (username,))
    user_data = cursor.fetchone()
    
    if not user_data:
        conn.close()
        return False, "User not found"
    
    # Verify password
    stored_hash = user_data['password_hash']
    salt = user_data['salt']
    
    # Hash the provided password with the stored salt
    input_hash, _ = hash_password(password, salt)
    
    if input_hash == stored_hash:
        # Update last login time
        user_id = user_data['user_id']
        cursor.execute('''
            UPDATE users 
            SET last_login = ?
            WHERE user_id = ?
        ''', (datetime.now(), user_id))
        conn.commit()
        conn.close()
        return True, user_id
    else:
        conn.close()
        return False, "Incorrect password"

def set_remember_token(user_id, remember=True):
    """Set or clear the remember token for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if remember:
        # Generate a unique token
        token = secrets.token_hex(32)
        cursor.execute('''
            UPDATE users 
            SET remember_token = ?
            WHERE user_id = ?
        ''', (token, user_id))
    else:
        # Clear the token
        cursor.execute('''
            UPDATE users 
            SET remember_token = NULL
            WHERE user_id = ?
        ''', (user_id,))
    
    conn.commit()
    conn.close()
    return token if remember else None

def get_user_by_token(token):
    """Retrieve user by remember token"""
    if not token:
        print("No token provided to get_user_by_token()")
        return None
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print(f"Looking up user with token: {token[:10]}...")
    
    try:
        cursor.execute('''
            SELECT user_id, username, remember_token
            FROM users 
            WHERE remember_token = ?
        ''', (token,))
        
        user = cursor.fetchone()
        
        if user:
            print(f"Found user: {user['username']} with matching token")
            user_dict = dict(user)
            conn.close()
            return user_dict
        else:
            # If no user found, let's check what tokens actually exist in the database
            cursor.execute('SELECT username, remember_token FROM users WHERE remember_token IS NOT NULL')
            existing_tokens = cursor.fetchall()
            if existing_tokens:
                print(f"Found {len(existing_tokens)} users with tokens in database:")
                for u in existing_tokens:
                    # Show truncated tokens for comparison
                    print(f" - {u['username']}: {u['remember_token'][:10]}...")
            else:
                print("No users with tokens found in database")
            
            conn.close()
            return None
    except Exception as e:
        print(f"Database error in get_user_by_token(): {e}")
        conn.close()
        return None

# --- Functions to interact with the database (updated with user_id) ---

def start_new_session(user_id, session_type, target_metric_name=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    start_time = datetime.now()
    cursor.execute('''
        INSERT INTO sessions (user_id, session_type, start_time, target_metric_name)
        VALUES (?, ?, ?, ?)
    ''', (user_id, session_type, start_time, target_metric_name))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"Started new session ID: {session_id} of type {session_type} for user {user_id}")
    return session_id, start_time

def add_session_metric(session_id, prediction_label, is_on_target, raw_score=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now()
    cursor.execute('''
        INSERT INTO session_metrics (session_id, timestamp, prediction_label, is_on_target, raw_score)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, timestamp, prediction_label, is_on_target, raw_score))
    conn.commit()
    conn.close()

def end_session_and_summarize(session_id, end_time):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get start time to calculate duration
    cursor.execute("SELECT start_time FROM sessions WHERE session_id = ?", (session_id,))
    session_data = cursor.fetchone()
    if not session_data:
        conn.close()
        print(f"Error: Session ID {session_id} not found for ending.")
        return

    start_time_str = session_data['start_time']
    # SQLite stores timestamps as text by default if not specified otherwise. Convert.
    start_time = datetime.fromisoformat(start_time_str)
    duration_seconds = int((end_time - start_time).total_seconds())

    cursor.execute('''
        UPDATE sessions
        SET end_time = ?, duration_seconds = ?
        WHERE session_id = ?
    ''', (end_time, duration_seconds, session_id))

    # Calculate summary
    cursor.execute('''
        SELECT COUNT(*) as total_points, SUM(is_on_target) as on_target_points, AVG(raw_score) as avg_score
        FROM session_metrics
        WHERE session_id = ?
    ''', (session_id,))
    metrics_summary = cursor.fetchone()

    time_on_target_seconds = 0
    percent_on_target = 0.0
    avg_raw_score = metrics_summary['avg_score'] if metrics_summary and metrics_summary['avg_score'] is not None else None


    if metrics_summary and metrics_summary['total_points'] > 0:
        # This is a simplified calculation. For accurate time_on_target_seconds,
        # you'd need to sum durations between consecutive on_target points.
        # For now, let's assume each metric point represents an equal time slice (e.g., 1 second).
        # This needs to be refined based on how often you log metrics.
        # A better way is to calculate it when logging by taking diff between timestamps.
        # For simplicity here, using count of 'on_target' points * assumed_interval_per_point
        assumed_interval_per_point = 1 # Placeholder - needs to match your metric logging frequency
        time_on_target_seconds = metrics_summary['on_target_points'] * assumed_interval_per_point
        percent_on_target = (metrics_summary['on_target_points'] / metrics_summary['total_points']) * 100.0

    cursor.execute('''
        INSERT INTO session_summary (session_id, time_on_target_seconds, percent_on_target, average_raw_score)
        VALUES (?, ?, ?, ?)
    ''', (session_id, time_on_target_seconds, percent_on_target, avg_raw_score))

    conn.commit()
    conn.close()
    print(f"Ended and summarized session ID: {session_id}")


def get_all_sessions_summary(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.session_id, s.session_type, s.start_time, s.duration_seconds,
               ss.percent_on_target, ss.time_on_target_seconds
        FROM sessions s
        JOIN session_summary ss ON s.session_id = ss.session_id
        WHERE s.user_id = ?
        ORDER BY s.start_time DESC
    ''', (user_id,))
    sessions = cursor.fetchall() # Returns a list of sqlite3.Row objects
    conn.close()
    return sessions

def get_session_details(session_id):
    """ Fetches granular metrics for a specific session, e.g., for plotting. """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, prediction_label, is_on_target, raw_score
        FROM session_metrics
        WHERE session_id = ?
        ORDER BY timestamp ASC
    ''', (session_id,))
    details = cursor.fetchall()
    conn.close()
    return details

def log_session_event(session_id, event_type, prediction_label=None, value=None, level=None, is_on_target=False):
    """
    Log various session events, including EEG predictions and user interactions.
    
    Args:
        session_id: The active session ID
        event_type: Type of event (e.g., 'PREDICTION', 'USER_FEEDBACK', 'SCENE_CHANGE')
        prediction_label: For EEG predictions, the mental state label
        value: Normalized value (0.0-1.0) representing state intensity
        level: Discrete level (-3 to +4) of the mental state
        is_on_target: Whether this state meets the session's goal
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now()
    
    if event_type == 'PREDICTION':
        # For predictions, use the existing session_metrics table
        add_session_metric(session_id, prediction_label, is_on_target, raw_score=value)
        return
        
    # For other event types, we could add a new events table
    # This example uses the existing session_metrics table but marks
    # the event type in the prediction_label field
    cursor.execute('''
        INSERT INTO session_metrics (session_id, timestamp, prediction_label, is_on_target, raw_score)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, timestamp, f"{event_type}:{prediction_label}", is_on_target, value))
    
    conn.commit()
    conn.close()

def get_all_sessions_summary_legacy():
    """Legacy function for old DB schema without user_id"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.session_id, s.session_type, s.start_time, s.duration_seconds,
               ss.percent_on_target, ss.time_on_target_seconds
        FROM sessions s
        JOIN session_summary ss ON s.session_id = ss.session_id
        ORDER BY s.start_time DESC
    ''')
    sessions = cursor.fetchall() # Returns a list of sqlite3.Row objects
    conn.close()
    return sessions

def end_session(session_id):
    """Simple wrapper to end a session without calculating summary."""
    end_time = datetime.now()
    return end_session_and_summarize(session_id, end_time)

def ensure_remember_token_column_exists():
    """Make sure the remember_token column exists in the users table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if the column exists
    cursor.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    column_names = [col['name'] for col in columns]
    
    if 'remember_token' not in column_names:
        print("Adding remember_token column to users table...")
        cursor.execute("ALTER TABLE users ADD COLUMN remember_token TEXT")
        conn.commit()
        print("Column added successfully")
    else:
        print("remember_token column already exists")
        
    conn.close()

def save_session_band_data(bands_dict):
    """Save band power data for a session"""
    session_id = bands_dict["session_id"]
    
    # Connect to database
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    
    try:
        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_band_data (
            id INTEGER PRIMARY KEY,
            session_id INTEGER,
            timestamp REAL,
            alpha REAL,
            beta REAL,
            theta REAL,
            ab_ratio REAL,
            bt_ratio REAL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        ''')
        
        # Insert data in chunks for efficiency
        entries = []
        for i in range(len(bands_dict["timestamps"])):
            # Make sure we don't go out of bounds
            if i < len(bands_dict["alpha"]) and i < len(bands_dict["beta"]) and \
               i < len(bands_dict["theta"]) and i < len(bands_dict["ab_ratio"]) and \
               i < len(bands_dict["bt_ratio"]):
                entries.append((
                    session_id,
                    bands_dict["timestamps"][i],
                    bands_dict["alpha"][i],
                    bands_dict["beta"][i],
                    bands_dict["theta"][i],
                    bands_dict["ab_ratio"][i],
                    bands_dict["bt_ratio"][i]
                ))
        
        # Insert in a single transaction for speed
        cursor.executemany('''
        INSERT INTO session_band_data 
        (session_id, timestamp, alpha, beta, theta, ab_ratio, bt_ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', entries)
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving band data: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def save_session_eeg_data(session_id, eeg_data, timestamps):
    """Save EEG data for a session"""
    # This could be stored in a separate table or file due to size
    # For simplicity, I'll show a minimal SQLite implementation
    
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    
    try:
        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_eeg_data (
            id INTEGER PRIMARY KEY,
            session_id INTEGER,
            timestamp REAL,
            channel_0 REAL,
            channel_1 REAL,
            channel_2 REAL,
            channel_3 REAL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        ''')
        
        # Insert data in chunks
        entries = []
        for i in range(min(len(timestamps), len(eeg_data[0]))):
            entries.append((
                session_id,
                timestamps[i],
                eeg_data[0][i] if len(eeg_data) > 0 else 0,
                eeg_data[1][i] if len(eeg_data) > 1 else 0,
                eeg_data[2][i] if len(eeg_data) > 2 else 0,
                eeg_data[3][i] if len(eeg_data) > 3 else 0
            ))
            
            # Insert in batches to avoid memory issues with large datasets
            if len(entries) >= 1000:
                cursor.executemany('''
                INSERT INTO session_eeg_data 
                (session_id, timestamp, channel_0, channel_1, channel_2, channel_3)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', entries)
                entries = []
        
        # Insert any remaining entries
        if entries:
            cursor.executemany('''
            INSERT INTO session_eeg_data 
            (session_id, timestamp, channel_0, channel_1, channel_2, channel_3)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', entries)
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving EEG data: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

# Call initialize_database() once when your app starts or when this module is first imported.
if __name__ == "__main__":
    initialize_database()
    # Example usage (for testing this module standalone)
    # sid, st = start_new_session("Meditation-Video", "Relaxation")
    # add_session_metric(sid, "Relaxed", True, 0.8)
    # time.sleep(1)
    # add_session_metric(sid, "Neutral", False, 0.5)
    # time.sleep(1)
    # add_session_metric(sid, "Relaxed", True, 0.9)
    # end_session_and_summarize(sid, datetime.now())
    # print(get_all_sessions_summary())
else:
    # Ensures DB is ready when imported by the main app
    initialize_database()