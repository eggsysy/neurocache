import time
from simulator.device import MemorySimulator
from simulator.predictor import PredictionEngine

def run_simulation():
    print("=== STARTING NEUROCACHE SIMULATION ===")
    
    # 1. Initialize our system with 4GB of RAM
    device = MemorySimulator(total_ram_mb=4096)
    
    # 2. Define the apps on our "phone"
    app_catalog = ["whatsapp", "maps", "spotify", "chrome", "instagram", "camera", "youtube", "gmail"]
    predictor = PredictionEngine(app_catalog)

    # 3. Simulate a normal user opening background apps
    print("\n--- PHASE 1: Normal Phone Usage ---")
    device.allocate("whatsapp", 300, "background_cache")
    device.allocate("spotify", 250, "background_cache")
    device.allocate("chrome", 800, "background_cache")
    device.allocate("maps", 400, "background_cache")
    device.allocate("instagram", 600, "background_cache")
    
    # Make WhatsApp and Maps look "old" (last used 2 hours ago)
    device.active_apps["whatsapp"]["last_used"] -= 7200 
    device.active_apps["maps"]["last_used"] -= 7200
    
  # 4. Generate ML Predictions using REAL inference
    print("\n--- PHASE 2: ML Prediction Engine Fires ---")
    
    # We feed the model a sequence of 5 apps. 
    # Based on our Markov chain, Camera -> Instagram is a highly likely habit.
    recent_user_history = ["whatsapp", "chrome", "maps", "camera", "instagram"]
    print(f"User just opened: {recent_user_history}")
    
    # Ask the neural network what happens next
    real_predictions = predictor.get_predictions(recent_user_history)
    
    device.current_predictions = real_predictions
    print(f"Neural Network Output: {real_predictions}")

    # 5. Trigger the Agentic AI (This will cause an Out-Of-Memory crisis!)
    print("\n--- PHASE 3: Agentic AI Triggered (Memory Wall Hit) ---")
    print(f"Current Free RAM: {device.total_ram_mb - device.used_ram_mb}MB")
    print("Attempting to load Llama-3 (KV Cache requires 2500MB)...")
    
    # This allocation will force the P-LRU algorithm to decide who lives and who dies
    device.allocate("llama_3_agent", 2500, "agentic_ai")

    print("\n--- SIMULATION COMPLETE ---")
    print(f"Thrashing Events Prevented/Handled: {device.metrics['thrashing_events']}")
    print("Check the logs above: Notice how 'whatsapp' was killed instead of 'maps', even though both were idle for 2 hours!")

if __name__ == "__main__":
    run_simulation()