"""
Create sample news data for testing the pipeline.
"""

import json
from datetime import datetime, timedelta
import random

# Sample news headlines for different tickers
sample_news = [
    {"ticker": "AAPL", "headlines": [
        "Apple reports record quarterly earnings beating analyst expectations",
        "Apple announces new iPhone with revolutionary AI capabilities",
        "Apple stock rises 5% on strong services revenue growth",
        "Apple CEO discusses company's AI strategy in earnings call"
    ]},
    {"ticker": "GOOGL", "headlines": [
        "Google's AI breakthrough drives Alphabet stock surge",
        "Google Cloud revenue jumps 35% year-over-year",
        "Alphabet announces major AI research partnership",
        "Google faces new regulatory challenges in Europe"
    ]},
    {"ticker": "TSLA", "headlines": [
        "Tesla delivers record number of vehicles in Q4",
        "Tesla's Autopilot technology receives new safety updates",
        "Tesla stock volatile amid production concerns",
        "Elon Musk announces Tesla's expansion into new markets"
    ]},
    {"ticker": "MSFT", "headlines": [
        "Microsoft Azure revenue growth accelerates to 30%",
        "Microsoft's AI copilot adoption exceeds expectations",
        "Microsoft stock hits new all-time high",
        "Microsoft announces dividend increase and share buyback"
    ]}
]

def create_sample_news_file():
    """Create a sample news JSONL file."""
    
    with open('data/sample_news.jsonl', 'w') as f:
        news_id = 1
        
        for company in sample_news:
            ticker = company["ticker"]
            
            for i, headline in enumerate(company["headlines"]):
                # Create timestamps spread over the last 24 hours
                hours_ago = random.randint(1, 24)
                timestamp = datetime.now() - timedelta(hours=hours_ago)
                
                news_item = {
                    "id": f"{ticker}_{news_id}",
                    "ticker": ticker,
                    "headline": headline,
                    "timestamp": timestamp.isoformat()
                }
                
                f.write(json.dumps(news_item) + '\n')
                news_id += 1
    
    print(f"Created sample news file with {news_id-1} items")

if __name__ == "__main__":
    create_sample_news_file()