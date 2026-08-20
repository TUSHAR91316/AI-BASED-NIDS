import streamlit as st
import os

class ThreatSummarizer:
    def __init__(self):
        self.model_name = "moonshotai/Kimi-K2-Instruct-0905"
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        hf_token = os.environ.get("HF_TOKEN")
        
        # Safely check Streamlit secrets without throwing StreamlitSecretNotFoundError
        if not hf_token:
            try:
                if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
                    hf_token = st.secrets["HF_TOKEN"]
            except Exception:
                pass

        if hf_token:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                )
            except Exception:
                self.client = None

    def generate_summary(self, alerts):
        if not alerts:
            return "No incidents detected in the current logging window."

        total_alerts = len(alerts)
        high_critical = len([a for a in alerts if a.get('alert_level') in ['High', 'Critical', 'RED', 'Malicious/Attack']])
        unique_ips = list(set([str(a.get('src_ip')) for a in alerts if a.get('src_ip')]))
        top_tactics = list(set([str(a.get('mitre_tactics')) for a in alerts if a.get('mitre_tactics') and a.get('mitre_tactics') != 'None']))
        attack_types = list(set([str(a.get('rule_match')) for a in alerts if a.get('rule_match')]))

        # If LLM client is configured, request generation via API
        if self.client is not None:
            prompt = f"The network intrusion detection system processed a recent batch of traffic and found {total_alerts} security alerts. "
            prompt += f"There were {high_critical} high or critical risk threats identified. "
            
            if unique_ips:
                prompt += f"The attacks originated from source IPs including {', '.join(unique_ips[:4])}. "
            
            if top_tactics:
                prompt += f"Observed attacker techniques match MITRE ATT&CK tactics: {', '.join(top_tactics)}. "

            if attack_types:
                prompt += f"Detected threat vectors include: {', '.join(attack_types[:3])}. "
            
            prompt += "Provide a concise executive threat intelligence summary focusing on observed impacts and immediate containment actions."

            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=180
                )
                return completion.choices[0].message.content
            except Exception as e:
                pass  # Fallback to local deterministic briefing

        # Local High-Quality Executive Threat Briefing
        top_ip_str = ", ".join(unique_ips[:3]) if unique_ips else "Internal/External hosts"
        tactics_str = ", ".join(top_tactics) if top_tactics else "Network Discovery / Denial of Service"
        patterns_str = ", ".join(attack_types[:3]) if attack_types else "Suspicious Traffic Patterns"

        briefing = (
            f"**Executive Threat Assessment**: Ingested {total_alerts} network flows with **{high_critical} high/critical incidents** "
            f"originating across {len(unique_ips)} distinct endpoint(s) ({top_ip_str}). Primary threat vectors include **{patterns_str}** "
            f"associated with MITRE ATT&CK tactics (*{tactics_str}*).\n\n"
            f"**Recommended Containment Actions**:\n"
            f"1. Implement immediate ingress rate limiting or firewall drop rules for identified attacker IPs (`{top_ip_str}`).\n"
            f"2. Inspect perimeter border routers and verify TCP SYN cookies / volumetric DoS mitigation policies.\n"
            f"3. Validate flagged incidents in the **SOC Feedback Loop** to update active learning retraining datasets."
        )
        return briefing
