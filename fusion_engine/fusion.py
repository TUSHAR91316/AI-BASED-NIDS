class FusionEngine:
    def __init__(self):
        # Weights for the hybrid detection logic
        self.w_supervised = 0.45
        self.w_anomaly = 0.35
        self.w_rule = 0.20
        
        # Risk thresholds
        self.threshold_yellow = 0.4
        self.threshold_red = 0.75

    def compute_risk_score(self, supervised_score, anomaly_score, rule_score):
        """
        Combines scores from all layers to calculate a final Risk Score.
        Formula: 0.45 * Supervised + 0.35 * Anomaly + 0.20 * Rule
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
