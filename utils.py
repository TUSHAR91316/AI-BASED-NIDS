# utils.py
import ipaddress
import numpy as np
import pandas as pd

def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        return np.nan

def preprocess_dataframe(df):
    """
    Preprocess the dataframe by converting IPs and selecting relevant features.
    """
    # Convert IPs if present
    if 'src_ip' in df.columns:
        df['src_ip'] = df['src_ip'].apply(ip_to_int)
    if 'dst_ip' in df.columns:
        df['dst_ip'] = df['dst_ip'].apply(ip_to_int)

    # Select relevant features
    required_columns = ['dur', 'proto', 'sbytes', 'dbytes', 'sttl', 'ct_state_ttl']
    df = df[required_columns].copy()

    # Encode categorical 'proto'
    df['proto'] = df['proto'].astype('category').cat.codes

    # Ensure numeric values
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    return df
