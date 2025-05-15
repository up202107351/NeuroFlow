# database_manager.py
import sqlite3
from datetime import datetime
import os
from pathlib import Path

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

    # Session Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_type TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_seconds INTEGER,
            target_metric_name TEXT, -- e.g., 'Relaxation', 'Focus'
            notes TEXT -- Optional user notes
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

# --- Functions to interact with the database ---

def start_new_session(session_type, target_metric_name=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    start_time = datetime.now()
    cursor.execute('''
        INSERT INTO sessions (session_type, start_time, target_metric_name)
        VALUES (?, ?, ?)
    ''', (session_type, start_time, target_metric_name))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"Started new session ID: {session_id} of type {session_type}")
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


def get_all_sessions_summary():
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