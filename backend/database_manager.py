import sqlite3
from datetime import datetime
import os
from pathlib import Path
import hashlib
import secrets
import json
import numpy as np

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

    # Session Metrics (Batch storage)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            predictions_data TEXT, -- JSON array of predictions
            on_target_data TEXT, -- JSON array of is_on_target boolean values
            timestamps_data TEXT, -- JSON array of timestamps
            total_predictions INTEGER,
            on_target_count INTEGER,
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
            average_confidence REAL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')

    # Session Band Data (Batch storage)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_band_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER UNIQUE NOT NULL,
            timestamps_data TEXT NOT NULL, -- JSON array
            alpha_data TEXT NOT NULL, -- JSON array
            beta_data TEXT NOT NULL, -- JSON array
            theta_data TEXT NOT NULL, -- JSON array
            ab_ratio_data TEXT NOT NULL, -- JSON array
            bt_ratio_data TEXT NOT NULL, -- JSON array
            sample_count INTEGER NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')
    
    # Session EEG Data (Batch storage)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_eeg_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER UNIQUE NOT NULL,
            timestamps_data TEXT NOT NULL, -- JSON array
            channel_0_data TEXT NOT NULL, -- JSON array
            channel_1_data TEXT NOT NULL, -- JSON array
            channel_2_data TEXT NOT NULL, -- JSON array
            channel_3_data TEXT NOT NULL, -- JSON array
            sample_count INTEGER NOT NULL,
            sampling_rate REAL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    ''')

    conn.commit()
    conn.close()

def save_session_metrics_batch(session_id, predictions, on_target_flags, timestamps):
    """Save session metrics as optimized batch data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Validate inputs
        if not isinstance(session_id, int) or not predictions or not on_target_flags or not timestamps:
            return False
        
        if len(predictions) != len(on_target_flags) or len(predictions) != len(timestamps):
            return False
        
        # Check if session exists
        cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
        if not cursor.fetchone():
            return False
        
        # Convert arrays to JSON strings for storage
        predictions_json = json.dumps(predictions)
        on_target_json = json.dumps(on_target_flags)
        timestamps_json = json.dumps(timestamps)
        
        total_predictions = len(predictions)
        on_target_count = sum(on_target_flags)
        
        # Insert or replace session metrics
        cursor.execute('''
            INSERT OR REPLACE INTO session_metrics 
            (session_id, predictions_data, on_target_data, timestamps_data, total_predictions, on_target_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, predictions_json, on_target_json, timestamps_json, total_predictions, on_target_count))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error saving session metrics: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def save_session_band_data_batch(session_id, band_data_dict):
    """Save band power data as optimized batch"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Validate inputs
        if not isinstance(session_id, int) or not isinstance(band_data_dict, dict):
            return False
        
        required_keys = ["timestamps", "alpha", "beta", "theta", "ab_ratio", "bt_ratio"]
        if not all(key in band_data_dict for key in required_keys):
            return False
        
        # Check array lengths match
        lengths = [len(band_data_dict[key]) for key in required_keys]
        if len(set(lengths)) > 1 or lengths[0] == 0:
            return False
        
        # Check if session exists
        cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
        if not cursor.fetchone():
            return False
        
        # Convert arrays to JSON strings
        timestamps_json = json.dumps(band_data_dict["timestamps"])
        alpha_json = json.dumps(band_data_dict["alpha"])
        beta_json = json.dumps(band_data_dict["beta"])
        theta_json = json.dumps(band_data_dict["theta"])
        ab_ratio_json = json.dumps(band_data_dict["ab_ratio"])
        bt_ratio_json = json.dumps(band_data_dict["bt_ratio"])
        
        sample_count = len(band_data_dict["timestamps"])
        
        # Insert or replace band data
        cursor.execute('''
            INSERT OR REPLACE INTO session_band_data 
            (session_id, timestamps_data, alpha_data, beta_data, theta_data, 
             ab_ratio_data, bt_ratio_data, sample_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, timestamps_json, alpha_json, beta_json, theta_json,
              ab_ratio_json, bt_ratio_json, sample_count))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error saving band data: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def save_session_eeg_data_batch(session_id, eeg_data_dict, sampling_rate=256.0):
    """Save EEG data as optimized batch"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Validate inputs
        if not isinstance(session_id, int) or not isinstance(eeg_data_dict, dict):
            return False
        
        required_keys = ["timestamps", "channel_0", "channel_1", "channel_2", "channel_3"]
        if not all(key in eeg_data_dict for key in required_keys):
            return False
        
        # Check array lengths match
        lengths = [len(eeg_data_dict[key]) for key in required_keys]
        if len(set(lengths)) > 1 or lengths[0] == 0:
            return False
        
        # Check if session exists
        cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
        if not cursor.fetchone():
            return False
        
        # Convert arrays to JSON strings
        timestamps_json = json.dumps(eeg_data_dict["timestamps"])
        channel_0_json = json.dumps(eeg_data_dict["channel_0"])
        channel_1_json = json.dumps(eeg_data_dict["channel_1"])
        channel_2_json = json.dumps(eeg_data_dict["channel_2"])
        channel_3_json = json.dumps(eeg_data_dict["channel_3"])
        
        sample_count = len(eeg_data_dict["timestamps"])
        
        # Insert or replace EEG data
        cursor.execute('''
            INSERT OR REPLACE INTO session_eeg_data 
            (session_id, timestamps_data, channel_0_data, channel_1_data, 
             channel_2_data, channel_3_data, sample_count, sampling_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, timestamps_json, channel_0_json, channel_1_json,
              channel_2_json, channel_3_json, sample_count, sampling_rate))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error saving EEG data: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

# User authentication functions
def hash_password(password, salt=None):
    """Hash password with salt using SHA-256"""
    if salt is None:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return pw_hash, salt

def register_user(username, password):
    """Register a new user with hashed password"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return False, "Username already exists"
    
    password_hash, salt = hash_password(password)
    created_at = datetime.now()
    
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
    
    cursor.execute('''
        SELECT user_id, password_hash, salt
        FROM users 
        WHERE username = ?
    ''', (username,))
    user_data = cursor.fetchone()
    
    if not user_data:
        conn.close()
        return False, "User not found"
    
    stored_hash = user_data['password_hash']
    salt = user_data['salt']
    input_hash, _ = hash_password(password, salt)
    
    if input_hash == stored_hash:
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
        token = secrets.token_hex(32)
        cursor.execute('''
            UPDATE users 
            SET remember_token = ?
            WHERE user_id = ?
        ''', (token, user_id))
    else:
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
        return None
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT user_id, username, remember_token
            FROM users 
            WHERE remember_token = ?
        ''', (token,))
        
        user = cursor.fetchone()
        if user:
            user_dict = dict(user)
            conn.close()
            return user_dict
        else:
            conn.close()
            return None
    except Exception as e:
        conn.close()
        return None

# Session management functions
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
    return session_id, start_time

def end_session_and_summarize(session_id, end_time):
    """End session and create summary from stored metrics"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get start time to calculate duration
    cursor.execute("SELECT start_time FROM sessions WHERE session_id = ?", (session_id,))
    session_data = cursor.fetchone()
    if not session_data:
        conn.close()
        return

    start_time_str = session_data['start_time']
    start_time = datetime.fromisoformat(start_time_str)
    duration_seconds = int((end_time - start_time).total_seconds())

    # Update session end time and duration
    cursor.execute('''
        UPDATE sessions
        SET end_time = ?, duration_seconds = ?
        WHERE session_id = ?
    ''', (end_time, duration_seconds, session_id))

    # Get metrics data for summary
    cursor.execute('''
        SELECT total_predictions, on_target_count, timestamps_data
        FROM session_metrics
        WHERE session_id = ?
    ''', (session_id,))
    metrics_data = cursor.fetchone()

    time_on_target_seconds = 0
    percent_on_target = 0.0
    
    if metrics_data and metrics_data['total_predictions'] > 0:
        total_predictions = metrics_data['total_predictions']
        on_target_count = metrics_data['on_target_count']
        
        percent_on_target = (on_target_count / total_predictions) * 100.0
        
        # Calculate time on target from timestamps
        try:
            timestamps_data = json.loads(metrics_data['timestamps_data'])
            if len(timestamps_data) > 1:
                session_duration = timestamps_data[-1] - timestamps_data[0]
                time_on_target_seconds = int((on_target_count / total_predictions) * session_duration)
        except:
            time_on_target_seconds = on_target_count  # Fallback

    # Insert session summary
    cursor.execute('''
        INSERT OR REPLACE INTO session_summary 
        (session_id, time_on_target_seconds, percent_on_target, average_confidence)
        VALUES (?, ?, ?, ?)
    ''', (session_id, time_on_target_seconds, percent_on_target, 0.0))

    conn.commit()
    conn.close()

def end_session(session_id):
    """Simple wrapper to end a session"""
    end_time = datetime.now()
    return end_session_and_summarize(session_id, end_time)

def get_all_sessions_summary(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.session_id, s.session_type, s.start_time, s.duration_seconds,
               ss.percent_on_target, ss.time_on_target_seconds
        FROM sessions s
        LEFT JOIN session_summary ss ON s.session_id = ss.session_id
        WHERE s.user_id = ?
        ORDER BY s.start_time DESC
    ''', (user_id,))
    sessions = cursor.fetchall()
    conn.close()
    return sessions

def get_session_metrics_data(session_id):
    """Retrieve session metrics data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT predictions_data, on_target_data, timestamps_data
            FROM session_metrics
            WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        if result:
            predictions = json.loads(result['predictions_data'])
            on_target = json.loads(result['on_target_data'])
            timestamps = json.loads(result['timestamps_data'])
            
            return {
                'predictions': predictions,
                'on_target': on_target,
                'timestamps': timestamps
            }
        return None
        
    except Exception as e:
        return None
    finally:
        conn.close()

def get_session_band_data(session_id):
    """Retrieve band power data for a session"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT timestamps_data, alpha_data, beta_data, theta_data, 
                   ab_ratio_data, bt_ratio_data
            FROM session_band_data 
            WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        if result:
            data = {
                'timestamps': json.loads(result['timestamps_data']),
                'alpha': json.loads(result['alpha_data']),
                'beta': json.loads(result['beta_data']),
                'theta': json.loads(result['theta_data']),
                'ab_ratio': json.loads(result['ab_ratio_data']),
                'bt_ratio': json.loads(result['bt_ratio_data'])
            }
            return data
        return None
            
    except Exception as e:
        return None
    finally:
        conn.close()

def get_session_eeg_data(session_id):
    """Retrieve raw EEG data for a session"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT timestamps_data, channel_0_data, channel_1_data, 
                   channel_2_data, channel_3_data, sampling_rate
            FROM session_eeg_data 
            WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        if result:
            data = {
                'timestamps': json.loads(result['timestamps_data']),
                'channel_0': json.loads(result['channel_0_data']),
                'channel_1': json.loads(result['channel_1_data']),
                'channel_2': json.loads(result['channel_2_data']),
                'channel_3': json.loads(result['channel_3_data']),
                'sampling_rate': result['sampling_rate']
            }
            return data
        return None
            
    except Exception as e:
        return None
    finally:
        conn.close()

# Utility functions
def add_session_note(session_id, note):
    """Add a note to a session"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE sessions 
            SET notes = COALESCE(notes, '') || ? || char(10)
            WHERE session_id = ?
        ''', (note, session_id))
        
        conn.commit()
        return True
    except Exception as e:
        return False
    finally:
        if conn:
            conn.close()

# Legacy compatibility functions
def add_session_metric(session_id, prediction_label, is_on_target, raw_score=None):
    """Legacy function - now just logs the call"""
    pass

def get_session_details(session_id):
    """Legacy function - returns metrics data in old format"""
    metrics_data = get_session_metrics_data(session_id)
    if not metrics_data:
        return []
    
    details = []
    for i, prediction in enumerate(metrics_data['predictions']):
        if i < len(metrics_data['on_target']) and i < len(metrics_data['timestamps']):
            details.append({
                'timestamp': metrics_data['timestamps'][i],
                'prediction_label': prediction,
                'is_on_target': metrics_data['on_target'][i],
                'raw_score': None
            })
    
    return details

# Initialize database when module is imported
if __name__ == "__main__":
    initialize_database()
else:
    initialize_database()