import time
import random
import torch
from simulator.device import MemorySimulator
from simulator.predictor import PredictionEngine
from simulator.rl_agent import RLOptimizer

def train_rl_agent_cloud():
    print("=== STARTING CLOUD-SCALE REINFORCEMENT LEARNING ===")
    print("Target: 50,000 Episodes | Checkpointing every 5,000 Episodes\n")
    
    # 1. Initialize our components
    app_catalog = ["whatsapp", "maps", "spotify", "chrome", "instagram", "camera", "youtube", "gmail"]
    predictor = PredictionEngine(app_catalog)
    rl_agent = RLOptimizer(state_dim=2, action_dim=3)
    
    # [ENTERPRISE UPGRADE] Cloud-Scale Parameters
    EPISODES = 50000
    BATCH_SIZE = 128

    start_time = time.time()

    for e in range(EPISODES):
        # Reset the OS environment for the new episode
        device = MemorySimulator(total_ram_mb=4096)
        starting_weight = 5000.0
        device.cache_controller.w_pred = starting_weight
        
        # Load up the system to cause pressure
        apps_to_load = random.sample(app_catalog, 5)
        for app in apps_to_load:
            device.allocate(app, random.randint(300, 800), "background_cache")
            device.active_apps[app]["last_used"] -= random.randint(100, 10000)

        # Generate some predictions
        recent_history = random.sample(app_catalog, 5)
        predictions = predictor.get_predictions(recent_history)
        device.current_predictions = predictions

        # Define the RL State: [RAM Utilization %, Thrashing Count]
        ram_utilization = device.used_ram_mb / device.total_ram_mb
        state = [ram_utilization, device.metrics['thrashing_events']]
        
        total_reward = 0
        
        # Simulate 5 distinct memory stress events per episode
        for time_step in range(5):
            action = rl_agent.act(state)
            
            # 0 = Decrease Weight (-1000), 1 = Hold, 2 = Increase Weight (+1000)
            if action == 0: device.cache_controller.w_pred = max(0, device.cache_controller.w_pred - 1000)
            elif action == 2: device.cache_controller.w_pred += 1000
                
            current_weight = device.cache_controller.w_pred
            thrashing_before = device.metrics['thrashing_events']

            # Trigger the stress test
            try:
                device.allocate(f"heavy_workload_{time_step}", 1500, "agentic_ai")
                reward = 10 
            except Exception:
                reward = -100 

            # Calculate penalties
            thrashing_after = device.metrics['thrashing_events']
            if thrashing_after > thrashing_before:
                reward -= 50 

            # Observe new state
            next_ram_utilization = device.used_ram_mb / device.total_ram_mb
            next_state = [next_ram_utilization, thrashing_after]
            
            # Store experience
            done = time_step == 4
            rl_agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # Replay and Learn
        rl_agent.replay(BATCH_SIZE)
        
        # [ENTERPRISE UPGRADE] Sparse Logging for AWS CloudWatch
        if (e + 1) % 500 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"Ep: {e+1:05d}/{EPISODES} | Reward: {total_reward:4d} | W: {current_weight} | Eps: {rl_agent.epsilon:.3f} | Time: {elapsed:.1f}m")

        # [ENTERPRISE UPGRADE] Cloud Checkpointing
        if (e + 1) % 5000 == 0:
            ckpt_path = f'simulator/rl_weights_ckpt_{e+1}.pth'
            torch.save(rl_agent.model.state_dict(), ckpt_path)
            print(f">>> [CHECKPOINT] Brain state securely backed up to {ckpt_path}")

    # Final Save
    rl_agent.save()
    print("\n[SUCCESS] 50k Epoch Cloud Training Complete. Gold weights saved to 'rl_weights.pth'")

if __name__ == "__main__":
    train_rl_agent_cloud()