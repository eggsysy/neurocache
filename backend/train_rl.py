import time
import random
from simulator.device import MemorySimulator
from simulator.predictor import PredictionEngine
from simulator.rl_agent import RLOptimizer

def train_rl_agent():
    print("=== STARTING REINFORCEMENT LEARNING TRAINING ===")
    
    # 1. Initialize our components
    app_catalog = ["whatsapp", "maps", "spotify", "chrome", "instagram", "camera", "youtube", "gmail"]
    predictor = PredictionEngine(app_catalog)
    rl_agent = RLOptimizer(state_dim=2, action_dim=3)
    
    EPISODES = 500
    BATCH_SIZE = 64

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
            # 1. The RL Agent takes an action based on the state
            action = rl_agent.act(state)
            
            # 0 = Decrease Weight (-1000)
            # 1 = Hold Weight (+0)
            # 2 = Increase Weight (+1000)
            if action == 0: device.cache_controller.w_pred = max(0, device.cache_controller.w_pred - 1000)
            elif action == 2: device.cache_controller.w_pred += 1000
                
            current_weight = device.cache_controller.w_pred
            thrashing_before = device.metrics['thrashing_events']

            # 2. Trigger the stress test (Loading heavy KV Cache workloads)
            try:
                device.allocate(f"heavy_workload_{time_step}", 1500, "agentic_ai")
                reward = 10 # Base reward for surviving the allocation
            except Exception as ex:
                reward = -100 # Massive penalty for total OS crash

            # 3. Calculate penalties for bad decisions
            thrashing_after = device.metrics['thrashing_events']
            if thrashing_after > thrashing_before:
                reward -= 50 # Penalty for causing a thrashing event

            # 4. Observe new state
            next_ram_utilization = device.used_ram_mb / device.total_ram_mb
            next_state = [next_ram_utilization, thrashing_after]
            
            # 5. Store experience and calculate total score
            done = time_step == 4
            rl_agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # 6. Replay and Learn at the end of the episode
        rl_agent.replay(BATCH_SIZE)
        
        # Print episode stats
        print(f"Episode: {e+1:03d}/{EPISODES} | Reward: {total_reward:4d} | Final P-LRU Weight: {current_weight} | Epsilon (Exploration): {rl_agent.epsilon:.2f}")

    # Save the optimized brain
    rl_agent.save()
    print("\n[SUCCESS] RL Agent fully trained and weights saved to 'rl_weights.pth'!")

if __name__ == "__main__":
    train_rl_agent()