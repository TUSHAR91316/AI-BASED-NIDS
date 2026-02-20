import streamlit as st
import os
from openai import OpenAI

class ThreatSummarizer:
    def __init__(self):
        # We use the Hugging Face Router API via the OpenAI client
        self.model_name = "moonshotai/Kimi-K2-Instruct-0905"
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        hf_token = os.environ.get("HF_TOKEN")
        
        # If not in env, check Streamlit secrets mapping
        if not hf_token and hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            hf_token = st.secrets["HF_TOKEN"]

        if hf_token:
            try:
                self.client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_token,
                )
            except Exception as e:
                st.error(f"Failed to initialize OpenAI client: {e}")

    def generate_summary(self, alerts):
        if not alerts:
            return "No threats detected to summarize."

        if self.client is None:
            return "Error: Hugging Face Token (HF_TOKEN) not found. Please set it in your environment variables or Streamlit secrets."

        # Consolidate alerts into a text format that the model can understand
        total_alerts = len(alerts)
        high_critical = len([a for a in alerts if a.get('alert_level') in ['High', 'Critical']])
        unique_ips = list(set([str(a.get('src_ip')) for a in alerts]))
        top_tactics = list(set([str(a.get('mitre_tactics')) for a in alerts if a.get('mitre_tactics') and a.get('mitre_tactics') != 'None']))

        prompt = f"The network intrusion detection system processed a recent batch of traffic and found {total_alerts} security alerts. "
        prompt += f"There were {high_critical} high or critical risk threats identified. "
        
        if unique_ips:
            prompt += f"The attacks originated from various source IPs including {', '.join(unique_ips[:3])}. "
        
        if top_tactics:
            prompt += f"The main attacker techniques observed match MITRE ATT&CK tactics such as {', '.join(top_tactics)}. "

        # Append specific notable events (first few critical ones)
        critical_alerts = [a for a in alerts if a.get('alert_level') in ['Critical', 'High']]
        if critical_alerts:
            prompt += "A notable serious incident includes an attack described as " + str(critical_alerts[0].get('rule_match', 'Unknown Rule Error')) + ". "
        
        prompt += "Please provide a concise, executive-level Threat Intelligence summary of these events. Focus on the impact and recommended immediate actions in 2-3 sentences."

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=150
            )

            # Extract the actual text completion
            return completion.choices[0].message.content
        except Exception as e:
            return f"Failed to generate summary via Hugging Face API: {e}"
