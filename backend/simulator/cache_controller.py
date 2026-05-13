import time

class SmartCacheController:
    def __init__(self, time_weight=1.0, prediction_weight=5000.0):
        """
        time_weight: How much we care about the app being old (standard LRU).
        prediction_weight: The multiplier for our ML prediction to 'shield' the app.
        """
        self.w_time = time_weight
        self.w_pred = prediction_weight

    def calculate_eviction_score(self, app_data, current_time, prediction_prob=0.0):
        """
        Calculates the P-LRU score. 
        HIGHER SCORE = Gets evicted first.
        Formula: (Time Idle) - (Prediction Probability * Weight)
        """
        time_idle = current_time - app_data["last_used"]
        
        # The higher the ML probability, the larger the negative 'shield'
        # This artificially lowers the eviction score, saving the app from being killed.
        shield = self.w_pred * prediction_prob
        
        return (self.w_time * time_idle) - shield

    def get_app_to_evict(self, background_apps, current_predictions):
        """
        Evaluates all background apps and chooses the one with the highest eviction score.
        """
        current_time = time.time()
        highest_score = -float('inf')
        app_to_kill = None

        for app_id, app_data in background_apps.items():
            # Get the ML prediction for this app (default to 0% if no prediction exists)
            prob = current_predictions.get(app_id, 0.0) 
            
            score = self.calculate_eviction_score(app_data, current_time, prob)
            
            print(f"  [P-LRU] App: {app_id} | Idle: {time_idle:.1f}s | Prob: {prob*100}% | Score: {score:.1f}")
            
            if score > highest_score:
                highest_score = score
                app_to_kill = app_id
                
        return app_to_kill