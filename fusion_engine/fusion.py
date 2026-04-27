from functools import lru_cache
import numpy as np
from collections import defaultdict
import time

class FusionEngine:
    def __init__(self):
        # Weights for the hybrid detection logic
        self.w_supervised = 0.45
        self.w_anomaly = 0.35
        self.w_rule = 0.20

        # Risk thresholds
        self.threshold_yellow = 0.4
        self.threshold_red = 0.75

        # Decision cache to avoid recomputing similar inputs
        self.decision_cache = {}
        self.cache_max_size = 10000
        self.cache_hits = 0
        self.cache_misses = 0

        # Performance tracking
        self.decision_times = []
        self.total_decisions = 0

    @lru_cache(maxsize=5000)
    def _cached_risk_computation(self, supervised_score, anomaly_score, rule_score):
        """
        Cached version of risk score computation for repeated similar inputs.
        """
        risk_score = (
            (self.w_supervised * supervised_score) +
            (self.w_anomaly * anomaly_score) +
            (self.w_rule * rule_score)
        )

        # CRITICAL OVERRIDE: If a specific rule matches with high confidence,
        # it overrides the weighted average. (Deterministic > Probabilistic)
        if rule_score > 0.8:
            risk_score = 1.0

        return round(risk_score, 4)

    def compute_risk_score(self, supervised_score, anomaly_score, rule_score):
        """
        Optimized risk score computation with caching.
        """
        start_time = time.time()

        # Create cache key (rounded to reduce cache misses)
        cache_key = (
            round(supervised_score, 2),
            round(anomaly_score, 2),
            round(rule_score, 2)
        )

        # Check cache
        if cache_key in self.decision_cache:
            self.cache_hits += 1
            risk_score = self.decision_cache[cache_key]
        else:
            self.cache_misses += 1
            risk_score = self._cached_risk_computation(
                supervised_score, anomaly_score, rule_score
            )

            # Cache the result
            if len(self.decision_cache) < self.cache_max_size:
                self.decision_cache[cache_key] = risk_score

        # Track performance
        decision_time = time.time() - start_time
        self.decision_times.append(decision_time)
        self.total_decisions += 1

        # Periodic cache cleanup (every 1000 decisions)
        if self.total_decisions % 1000 == 0:
            self._cleanup_cache()

        return risk_score

    def _cleanup_cache(self):
        """Periodic cache cleanup to maintain performance."""
        # Keep only recent entries if cache is getting full
        if len(self.decision_cache) > self.cache_max_size * 0.8:
            # Remove oldest 20% of entries
            items_to_remove = int(len(self.decision_cache) * 0.2)
            keys_to_remove = list(self.decision_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.decision_cache[key]

    def get_cache_stats(self):
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        avg_decision_time = np.mean(self.decision_times) if self.decision_times else 0

        return {
            'cache_size': len(self.decision_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_decisions': self.total_decisions,
            'avg_decision_time': avg_decision_time
        }

    def determine_alert_level(self, risk_score):
        """
        Determines the alert level (Green/Yellow/Red) based on risk score.
        """
        if risk_score > self.threshold_red:
            return "RED", "Malicious/Attack"
        elif risk_score >= self.threshold_yellow:
            return "YELLOW", "Suspicious"
        else:
            return "GREEN", "Normal"

    def process_flow(self, supervised_score, anomaly_score, rule_score):
        """
        Main function to process a flow's scores and return the final verdict.
        """
        final_score = self.compute_risk_score(supervised_score, anomaly_score, rule_score)
        color, status = self.determine_alert_level(final_score)
        
        return {
            "risk_score": final_score,
            "alert_color": color,
            "status": status,
            "details": {
                "supervised_score": supervised_score,
                "anomaly_score": anomaly_score,
                "rule_score": rule_score
            }
        }

if __name__ == "__main__":
    # Test cases
    engine = FusionEngine()
    
    # Case 1: High Attack Probability
    print("Test Case 1 (Attack):", engine.process_flow(0.9, 0.8, 0.5))
    
    # Case 2: Suspicious
    print("Test Case 2 (Suspicious):", engine.process_flow(0.3, 0.8, 0.1))
    
    # Case 3: Normal
    print("Test Case 3 (Normal):", engine.process_flow(0.1, 0.1, 0.0))
