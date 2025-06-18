"""
MarketPulse AI Configuration
Centralized configuration for data sources and system settings.
"""

import os
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load all configuration from environment variables."""
    
    config = {
        # Data source configuration
        "data_source": {
            "pathway": {
                "enabled": os.getenv("USE_PATHWAY", "false").lower() == "true",
                "live_feed": os.getenv("USE_LIVE_FEED", "false").lower() == "true",
                "finnhub_token": os.getenv("FINNHUB_TOKEN", ""),
                "symbols": os.getenv("SYMBOLS", "AAPL,TSLA,MSFT,GOOGL,NVDA,META,AMZN").split(","),
                "websocket_url": "wss://ws.finnhub.io",
            },
            "mock": {
                "enabled": True,  # Always available as fallback
                "refresh_interval": int(os.getenv("MOCK_REFRESH_INTERVAL", "900")),  # 15 minutes
                "num_articles": int(os.getenv("MOCK_NUM_ARTICLES", "10")),
            }
        },
        
        # OpenAI configuration
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "mock_mode": os.getenv("MOCK_EMBEDDING", "false").lower() == "true",
            "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            "chat_model": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            "max_tokens": int(os.getenv("MAX_TOKENS", "300")),
            "temperature": float(os.getenv("TEMPERATURE", "0.1")),
        },
        
        # System configuration
        "system": {
            "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.30")),
            "cache_size": int(os.getenv("CACHE_SIZE", "512")),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true",
        },
        
        # Streamlit configuration
        "streamlit": {
            "port": int(os.getenv("STREAMLIT_PORT", "8501")),
            "host": os.getenv("STREAMLIT_HOST", "localhost"),
            "theme": os.getenv("STREAMLIT_THEME", "light"),
        }
    }
    
    return config

# Load configuration
CONFIG = load_config()

def get_config(path: str = None) -> Any:
    """
    Get configuration value by path.
    
    Args:
        path: Dot-separated path (e.g., "data_source.pathway.enabled")
        
    Returns:
        Configuration value or entire config if no path specified
    """
    if path is None:
        return CONFIG
    
    keys = path.split(".")
    value = CONFIG
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    
    return value

def get_active_data_source() -> str:
    """Determine which data source is currently active."""
    if get_config("data_source.pathway.enabled"):
        return "pathway"
    else:
        return "mock"

def is_pathway_configured() -> bool:
    """Check if Pathway is properly configured."""
    pathway_config = get_config("data_source.pathway")
    
    if not pathway_config["enabled"]:
        return False
    
    if pathway_config["live_feed"] and not pathway_config["finnhub_token"]:
        return False
    
    return True

def validate_config() -> Dict[str, bool]:
    """Validate configuration and return status."""
    validation = {
        "openai_key_present": bool(get_config("openai.api_key")),
        "pathway_available": False,
        "pathway_configured": is_pathway_configured(),
        "finnhub_configured": bool(get_config("data_source.pathway.finnhub_token"))
    }
    
    # Check if pathway module is available
    try:
        import pathway_pipeline
        validation["pathway_available"] = True
    except ImportError:
        validation["pathway_available"] = False
    
    return validation