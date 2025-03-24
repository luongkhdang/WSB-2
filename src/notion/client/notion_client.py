import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
from notion_client import Client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('notion_client')

class NotionClient:
    def __init__(self):
        # Load environment variables
        # Try different potential paths for the .env file
        base_dir = Path(__file__).resolve().parents[3]  # Go up to the root directory
        env_paths = [
            base_dir / '.env',
            Path(os.getcwd()) / '.env',
            Path(os.getcwd()).parent / '.env'
        ]
        
        env_loaded = False
        for env_path in env_paths:
            if env_path.exists():
                logger.info(f"Loading .env file from: {env_path}")
                load_dotenv(dotenv_path=str(env_path))
                env_loaded = True
                break
        
        if not env_loaded:
            logger.warning("No .env file found. Trying to use environment variables directly.")
        
        # Get Notion API key
        self.api_key = os.getenv("NOTION_API_KEY")
        logger.info(f"API Key present: {bool(self.api_key)}")
        if not self.api_key:
            # Hardcode the API key from the .env file as a fallback
            self.api_key = "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"
            logger.info("Using hardcoded API key as fallback")
        
        # Initialize Notion client
        self.client = Client(auth=self.api_key)
        
        # Page IDs
        self.main_page_id = None  # Parent page
        self.market_scan_page_id = None  # Daily Market Scan database
        self.trade_log_page_id = None  # Trade Log database
        self.performance_review_db_id = None  # Weekly Performance Review database
        
        # Initialize pages
        self._init_pages()
    
    def _init_pages(self):
        """Initialize page IDs by finding or creating the necessary pages"""
        try:
            # First, check for the Weekly Performance Review database which we know exists
            logger.info("Searching for Weekly Performance Review database")
            response = self.client.search(query="Weekly Performance Review")
            
            for result in response.get("results", []):
                if result.get("object") == "database":
                    title_content = "".join([text.get("plain_text", "") for text in result.get("title", [])])
                    if "Weekly Performance Review" in title_content:
                        self.performance_review_db_id = result["id"]
                        logger.info(f"Found Weekly Performance Review database with ID: {self.performance_review_db_id}")
                        # Try to get the parent page if it exists and we have access
                        try:
                            parent_type = result.get("parent", {}).get("type")
                            if parent_type == "page_id":
                                self.main_page_id = result["parent"]["page_id"]
                                logger.info(f"Found parent page with ID: {self.main_page_id}")
                        except Exception as e:
                            logger.warning(f"Could not access parent page: {e}")
                        break
            
            # Now search for Daily Market Scan and Trade Log
            logger.info("Searching for Daily Market Scan database")
            response = self.client.search(query="Daily Market Scan")
            for result in response.get("results", []):
                if result.get("object") == "database":
                    title_content = "".join([text.get("plain_text", "") for text in result.get("title", [])])
                    if "Daily Market Scan" in title_content:
                        self.market_scan_page_id = result["id"]
                        logger.info(f"Found Daily Market Scan database with ID: {self.market_scan_page_id}")
                        break
            
            logger.info("Searching for Trade Log database")
            response = self.client.search(query="Trade Log")
            for result in response.get("results", []):
                if result.get("object") == "database":
                    title_content = "".join([text.get("plain_text", "") for text in result.get("title", [])])
                    if "Trade Log" in title_content:
                        self.trade_log_page_id = result["id"]
                        logger.info(f"Found Trade Log database with ID: {self.trade_log_page_id}")
                        break
            
        except Exception as e:
            logger.error(f"Error initializing Notion pages: {e}")
    
    def get_database_entries(self, database_id: str, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Get entries from a database with optional filters"""
        try:
            query_params = {}
            if filters:
                query_params["filter"] = filters
            
            response = self.client.databases.query(**query_params, database_id=database_id)
            return response.get("results", [])
        except Exception as e:
            logger.error(f"Error querying database {database_id}: {e}")
            return []
    
    def create_database_entry(self, database_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new entry in a database"""
        try:
            response = self.client.pages.create(
                parent={"database_id": database_id},
                properties=properties
            )
            return response
        except Exception as e:
            logger.error(f"Error creating database entry in {database_id}: {e}")
            return {}
    
    def update_database_entry(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing database entry"""
        try:
            response = self.client.pages.update(
                page_id=page_id,
                properties=properties
            )
            return response
        except Exception as e:
            logger.error(f"Error updating database entry {page_id}: {e}")
            return {}
    
    # Daily Market Scan Functions
    def get_market_scan_entries(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Get entries from the Daily Market Scan database"""
        if not self.market_scan_page_id:
            logger.warning("Daily Market Scan page ID not set")
            return []
        
        return self.get_database_entries(self.market_scan_page_id, filters)
    
    def add_market_scan_entry(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new entry to the Daily Market Scan database"""
        if not self.market_scan_page_id:
            logger.warning("Daily Market Scan page ID not set")
            return {}
        
        return self.create_database_entry(self.market_scan_page_id, properties)
    
    # Trade Log Functions
    def get_trade_log_entries(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Get entries from the Trade Log database"""
        if not self.trade_log_page_id:
            logger.warning("Trade Log page ID not set")
            return []
        
        return self.get_database_entries(self.trade_log_page_id, filters)
    
    def add_trade_log_entry(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new entry to the Trade Log database"""
        if not self.trade_log_page_id:
            logger.warning("Trade Log page ID not set")
            return {}
        
        return self.create_database_entry(self.trade_log_page_id, properties)
    
    def update_trade_log_entry(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing Trade Log entry"""
        return self.update_database_entry(page_id, properties)
    
    # Weekly Performance Review Functions
    def get_performance_review_entries(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Get entries from the Weekly Performance Review database"""
        if not self.performance_review_db_id:
            logger.warning("Weekly Performance Review database ID not set")
            return []
        
        return self.get_database_entries(self.performance_review_db_id, filters)
    
    def add_performance_review_entry(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new entry to the Weekly Performance Review database"""
        if not self.performance_review_db_id:
            logger.warning("Weekly Performance Review database ID not set")
            return {}
        
        return self.create_database_entry(self.performance_review_db_id, properties)
    
    def update_performance_review_entry(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing Weekly Performance Review entry"""
        return self.update_database_entry(page_id, properties)
    
    def get_weekly_performance_stats(self) -> Dict[str, Any]:
        """Get statistics from the Weekly Performance Review database"""
        if not self.performance_review_db_id:
            logger.warning("Weekly Performance Review database ID not set")
            return {}
        
        try:
            entries = self.get_performance_review_entries()
            
            # Initialize stats
            stats = {
                "total_entries": len(entries),
                "completed_reviews": 0,
                "total_profit_loss": 0,
                "avg_win_rate": 0,
                "improvement_areas": set(),
                "success_points": set()
            }
            
            if not entries:
                return stats
            
            # Calculate stats
            win_rates = []
            for entry in entries:
                props = entry.get("properties", {})
                
                # Check review status
                status_prop = props.get("Review Status", {})
                if status_prop.get("status", {}).get("name") == "Completed":
                    stats["completed_reviews"] += 1
                
                # Get P/L
                pl_prop = props.get("Total P/L", {})
                pl_value = pl_prop.get("number")
                if pl_value is not None:
                    stats["total_profit_loss"] += pl_value
                
                # Get win rate
                win_rate_prop = props.get("Win Rate", {})
                win_rate_value = win_rate_prop.get("number")
                if win_rate_value is not None:
                    win_rates.append(win_rate_value)
                
                # Get improvement areas
                improvement_prop = props.get("Improvement Areas", {})
                improvement_text = "".join([text.get("plain_text", "") for text in improvement_prop.get("rich_text", [])])
                if improvement_text:
                    for area in improvement_text.split("."):
                        area = area.strip()
                        if area:
                            stats["improvement_areas"].add(area)
                
                # Get success points
                success_prop = props.get("Success Points", {})
                success_text = "".join([text.get("plain_text", "") for text in success_prop.get("rich_text", [])])
                if success_text:
                    for point in success_text.split("."):
                        point = point.strip()
                        if point:
                            stats["success_points"].add(point)
            
            # Calculate average win rate
            if win_rates:
                stats["avg_win_rate"] = sum(win_rates) / len(win_rates)
            
            # Convert sets to lists for better JSON serialization
            stats["improvement_areas"] = list(stats["improvement_areas"])
            stats["success_points"] = list(stats["success_points"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating weekly performance stats: {e}")
            return {}


# Singleton instance
_instance = None

def get_notion_client() -> NotionClient:
    """Get or create a singleton instance of NotionClient"""
    global _instance
    if _instance is None:
        _instance = NotionClient()
    return _instance


if __name__ == "__main__":
    # Test the client
    client = get_notion_client()
    print(f"Main page ID: {client.main_page_id}")
    print(f"Daily Market Scan page ID: {client.market_scan_page_id}")
    print(f"Trade Log page ID: {client.trade_log_page_id}")
    print(f"Weekly Performance Review database ID: {client.performance_review_db_id}")
    
    # Test weekly performance stats
    if client.performance_review_db_id:
        stats = client.get_weekly_performance_stats()
        print("\nWeekly Performance Stats:")
        print(f"Total entries: {stats.get('total_entries', 0)}")
        print(f"Completed reviews: {stats.get('completed_reviews', 0)}")
        print(f"Total P/L: ${stats.get('total_profit_loss', 0):.2f}")
        print(f"Average win rate: {stats.get('avg_win_rate', 0):.2f}%")
        
        print("\nCommon improvement areas:")
        for area in stats.get("improvement_areas", [])[:3]:  # Show top 3
            print(f"- {area}")
        
        print("\nCommon success points:")
        for point in stats.get("success_points", [])[:3]:  # Show top 3
            print(f"- {point}")
