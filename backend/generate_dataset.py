import csv
import random
from datetime import datetime, timedelta

# 1. Our App Catalog
APPS = ["whatsapp", "maps", "spotify", "chrome", "instagram", "camera", "youtube", "gmail"]

# 2. The "Habits" (Transition Matrix)
# This dictates what app a user is likely to open NEXT based on the CURRENT app.
# THIS is the exact logic our LSTM will learn to predict.
TRANSITIONS = {
    "camera": {"instagram": 0.70, "whatsapp": 0.20, "chrome": 0.10},  # Usually post photos
    "maps": {"spotify": 0.60, "whatsapp": 0.30, "chrome": 0.10},      # Usually play music while driving
    "instagram": {"whatsapp": 0.50, "camera": 0.20, "youtube": 0.30}, 
    "youtube": {"whatsapp": 0.40, "chrome": 0.40, "instagram": 0.20},
    "gmail": {"chrome": 0.60, "whatsapp": 0.20, "youtube": 0.20},     # Usually click links in emails
    "spotify": {"maps": 0.40, "whatsapp": 0.40, "instagram": 0.20},
    "chrome": {"youtube": 0.40, "whatsapp": 0.30, "gmail": 0.30},
    "whatsapp": {"instagram": 0.40, "chrome": 0.30, "youtube": 0.30}
}

def get_next_app(current_app):
    """Picks the next app based on our weighted probabilities."""
    possible_next_apps = list(TRANSITIONS[current_app].keys())
    probabilities = list(TRANSITIONS[current_app].values())
    return random.choices(possible_next_apps, weights=probabilities, k=1)[0]

def generate_synthetic_data(num_users=50, sessions_per_user=20, max_switches_per_session=15):
    """Generates the CSV dataset."""
    data = []
    start_time = datetime.now() - timedelta(days=30) # Start data from 30 days ago
    
    print("Generating synthetic Android usage dataset...")
    
    for user_id in range(1, num_users + 1):
        current_time = start_time + timedelta(hours=random.randint(1, 24))
        
        for session_id in range(1, sessions_per_user + 1):
            # Pick a random starting app for the session
            current_app = random.choice(APPS)
            num_switches = random.randint(5, max_switches_per_session)
            
            for _ in range(num_switches):
                # Calculate duration spent in the app (between 10 seconds and 10 minutes)
                duration_sec = random.randint(10, 600)
                
                # Append the row
                data.append([
                    f"U{user_id:03d}",         # User ID
                    f"S{session_id:03d}",      # Session ID
                    current_time.strftime("%Y-%m-%d %H:%M:%S"), # Timestamp
                    current_app,               # App Name
                    duration_sec               # Time spent
                ])
                
                # Move time forward
                current_time += timedelta(seconds=duration_sec)
                
                # Determine the next app using our Markov chain
                current_app = get_next_app(current_app)
            
            # Add a gap between sessions (1 to 5 hours)
            current_time += timedelta(hours=random.randint(1, 5))
            
    return data

def save_to_csv(data, filename="synthetic_android_usage.csv"):
    headers = ["user_id", "session_id", "timestamp", "app_id", "duration_seconds"]
    
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)
        
    print(f"Success! Generated {len(data)} app switch events and saved to {filename}.")

if __name__ == "__main__":
    # Generates ~10,000 rows of highly structured training data
    dataset = generate_synthetic_data()
    save_to_csv(dataset)