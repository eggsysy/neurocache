import time
from simulator.cache_controller import SmartCacheController

class MemorySimulator:
    def __init__(self, total_ram_mb=6144): # Defaulting to 6GB RAM
        self.total_ram_mb = total_ram_mb
        self.used_ram_mb = 0
        
        self.cache_controller = SmartCacheController()
        self.current_predictions = {}

        # Our dynamic memory pools
        self.pools = {
            "foreground": 0,
            "background_cache": 0,
            "agentic_ai": 0,
            "system": 512  # OS always reserves some memory
        }
        
        # Tracking active apps in memory: {app_id: {"size": mb, "pool": name, "last_used": timestamp}}
        self.active_apps = {}
        
        # Hackathon KPIs
        self.metrics = {
            "thrashing_events": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    def allocate(self, app_id, size_mb, target_pool="foreground"):
        """Attempts to load an app into memory."""
        if self.used_ram_mb + size_mb > self.total_ram_mb:
            self._handle_oom(size_mb) # Out of Memory! Trigger eviction.
            
        self.active_apps[app_id] = {
            "size": size_mb,
            "pool": target_pool,
            "last_used": time.time()
        }
        self.pools[target_pool] += size_mb
        self.used_ram_mb += size_mb
        print(f"[ALLOCATE] {app_id} loaded into {target_pool} ({size_mb}MB). Free RAM: {self.total_ram_mb - self.used_ram_mb}MB")

    def evict(self, app_id):
        """Removes an app from memory."""
        if app_id in self.active_apps:
            app_data = self.active_apps.pop(app_id)
            self.pools[app_data["pool"]] -= app_data["size"]
            self.used_ram_mb -= app_data["size"]
            print(f"[EVICT] {app_id} cleared from RAM.")

    def _handle_oom(self, required_mb):
        """Emergency memory clearing using P-LRU Logic."""
        print(f"\n[WARNING] Memory Wall Hit! Need {required_mb}MB. Consulting P-LRU...")
        self.metrics["thrashing_events"] += 1
        
        # Gather all apps currently sitting in the background
        background_apps = {k: v for k, v in self.active_apps.items() if v["pool"] == "background_cache"}
        
        if not background_apps:
            raise MemoryError("System Crash: Cannot allocate memory even after attempting to clear background cache.")
            
        # --- THE MAGIC HAPPENS HERE ---
        # Instead of just finding the oldest app, we ask our Smart Controller
        app_to_kill = self.cache_controller.get_app_to_evict(background_apps, self.current_predictions)
        
        self.evict(app_to_kill)
        
        # If we still don't have enough room, run it recursively
        if self.used_ram_mb + required_mb > self.total_ram_mb:
            self._handle_oom(required_mb)

    def get_state(self):
        """Returns the current state for our Next.js Dashboard."""
        return {
            "total": self.total_ram_mb,
            "used": self.used_ram_mb,
            "pools": self.pools,
            "metrics": self.metrics
        }