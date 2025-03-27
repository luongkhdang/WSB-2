import logging
import time
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

class DataCache:
    """Cache for financial data to reduce API calls and improve reliability"""
    
    def __init__(self, cache_dir="./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized data cache at {self.cache_dir}")
        
    def get_cache_key(self, func_name, *args, **kwargs):
        """Generate a unique cache key based on function name and arguments"""
        key_parts = [func_name]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, func_name, *args, max_age_hours=24, **kwargs) -> Optional[Any]:
        """Get data from cache if available and not expired"""
        cache_key = self.get_cache_key(func_name, *args, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            # Check if cache is expired
            file_age = time.time() - cache_file.stat().st_mtime
            max_age_seconds = max_age_hours * 3600
            
            if file_age < max_age_seconds:
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"Retrieved {func_name} data from cache")
                    return data
                except Exception as e:
                    logger.warning(f"Failed to load cache for {func_name}: {e}")
        
        return None
    
    def set(self, func_name, data, *args, **kwargs):
        """Save data to cache"""
        if data is None:
            return
            
        cache_key = self.get_cache_key(func_name, *args, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Cached {func_name} data successfully")
        except Exception as e:
            logger.warning(f"Failed to cache {func_name} data: {e}") 