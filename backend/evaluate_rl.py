import random
from simulator.device import MemorySimulator
from simulator.predictor import PredictionEngine
from simulator.rl_agent import RLOptimizer

def run_stress_test(use_rl=False):
    app_catalog = ["whatsapp", "maps", "spotify", "chrome", "instagram", "camera", "youtube", "gmail"]
    predictor = PredictionEngine(app_catalog)
    device = MemorySimulator(total_ram_mb=4096)
    
    if use_rl:
        rl_agent = RLOptimizer(state_dim=2, action_dim=3)
        rl_agent.epsilon = 0.0 

    # 1. Fill the RAM almost to the absolute brim (Simulating heavy use)
    for app in app_catalog:
        device.allocate(app, random.randint(400, 700), "background_cache")
        device.active_apps[app]["last_used"] -= random.randint(100, 5000)

    # 2. Fire the LSTM Predictor
    recent_history = ["whatsapp", "chrome", "maps", "camera", "instagram"]
    device.current_predictions = predictor.get_predictions(recent_history)

    # 3. THE TRUE GAUNTLET: Sustained Memory Pressure
    # We load 4 massive background processes WITHOUT freeing them.
    # This forces the OS into a terrifying OOM cascade.
    for i in range(4):
        if use_rl:
            ram_utilization = device.used_ram_mb / device.total_ram_mb
            state = [ram_utilization, device.metrics['thrashing_events']]
            action = rl_agent.act(state)
            
            # The RL Agent senses the system dying and drops the prediction weight
            # to let standard LRU clear out space.
            if action == 0: device.cache_controller.w_pred = max(0, device.cache_controller.w_pred - 1500)
            elif action == 2: device.cache_controller.w_pred += 500

        try:
            # We allocate 900MB, but we DO NOT free it. The pressure stacks.
            device.allocate(f"heavy_ai_task_{i}", 900, "agentic_ai")
        except Exception:
            # If the OS throws an error because it literally cannot find space, massive penalty.
            device.metrics['thrashing_events'] += 10 

    return device.metrics['thrashing_events']

if __name__ == "__main__":
    print("=== COMMENCING SUSTAINED OOM CASCADE TEST ===")
    print("Simulating 4 stacked Agentic AI memory spikes with NO memory release...\n")
    
    static_thrashing = run_stress_test(use_rl=False)
    print(f"System V1 (Static P-LRU): {static_thrashing} thrashing events.")
    
    rl_thrashing = run_stress_test(use_rl=True)
    print(f"System V2 (Dynamic RL + P-LRU): {rl_thrashing} thrashing events.")
    
    print("\n--- RESULTS ---")
    if rl_thrashing < static_thrashing:
        improvement = ((static_thrashing - rl_thrashing) / static_thrashing) * 100
        print(f"[SUCCESS] The RL Agent improved OS stability by {improvement:.1f}%!")
    elif rl_thrashing == static_thrashing:
        print("[STABLE] The RL Agent matched the static baseline.")
    else:
        print("[WARNING] The RL Agent performed worse.")