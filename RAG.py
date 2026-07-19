import asyncio
import json
import os
import pickle
import time
from datetime import datetime


def generate_rag_query(observation, history_text):
    """
    Creates a search query for the RAG system based on the current observation,
    the long-term summary, and the complete short-term consolidation.
    """

    full_query = (
        f"History: {history_text}"
    )

    return full_query[-1200:].strip()


def format_relative_time(date_str, observation):
    """
    Converts a timestamp or time range into a relative natural language description.
    Handles formats like:
    - "2026-01-07 Wednesday 16:58:43"
    - "2026-01-07 Wednesday 15:13:49 to 2026-01-07 Wednesday 16:49:41"
    """
    try:
        # Reference time for calculation
        now = observation.get("time", "")

        raw_now = observation.get("time", "").replace("   - Current Time: ", "").strip()

        def parse_dt(s):
            s = s.strip()
            formats = [
                "%Y-%m-%d %A %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %A %H:%M",
                "%Y-%m-%d"
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            return datetime.fromisoformat(s.split(' ')[0])

        now = parse_dt(raw_now)

        duration_str = ""
        if " to " in date_str:
            start_str, end_str = date_str.split(" to ")
            dt_start = parse_dt(start_str)
            dt_end = parse_dt(end_str)

            # Calculate duration
            duration = dt_end - dt_start
            d_hours = duration.seconds // 3600
            d_mins = (duration.seconds % 3600) // 60

            if duration.days > 0:
                duration_str = f" for {duration.days} days"
            elif d_hours > 0:
                duration_str = f" for {d_hours} hours"
            elif d_mins > 0:
                duration_str = f" for {d_mins} minutes"

            dt = dt_start  # Use start time for relative calculation
        else:
            dt = parse_dt(date_str)

        diff = now - dt
        days = diff.days
        seconds = diff.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60

        time_str = dt.strftime("%H:%M:%S")

        if days == 0:
            if hours == 0:
                if minutes == 0:
                    return f"Just now{duration_str} at {time_str}"
                return f"{minutes} minutes ago{duration_str} at {time_str}"
            return f"Today, {hours} hours ago{duration_str} at {time_str}"
        elif days == 1:
            return f"Yesterday{duration_str} at {time_str}"
        elif days < 7:
            return f"{days} days ago{duration_str} at {time_str}"
        else:
            weeks = days // 7
            return f"{weeks} weeks ago{duration_str} at {time_str}"

    except Exception:
        return date_str