"""
Utility functions for the Hindi ASR assignment
"""

import json
import logging
import os
from typing import Dict, List, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def create_session_with_retries(retries: int = 3, timeout: float = 10) -> requests.Session:
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def download_with_retries(url: str, output_path: str, timeout: float = 30) -> bool:
    """Download file with retry logic"""
    try:
        session = create_session_with_retries()
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded {url} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def load_json_from_url(url: str) -> Dict[str, Any]:
    """Load JSON from URL"""
    try:
        session = create_session_with_retries()
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to load JSON from {url}: {e}")
        return {}


def save_results(results: Dict, output_path: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved results to {output_path}")


def load_results(input_path: str) -> Dict:
    """Load results from JSON file"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results from {input_path}: {e}")
        return {}


def format_metrics(metrics: Dict) -> str:
    """Format metrics for display"""
    formatted = "Metrics:\n"
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted += f"  {key}: {value:.4f}\n"
        else:
            formatted += f"  {key}: {value}\n"
    return formatted
