"""
Simple launcher for MarketPulse AI demo.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if basic requirements are met."""
    print("🔍 Checking requirements...")
    
    required_packages = ['streamlit', 'numpy', 'pandas', 'python-dotenv']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    if missing:
        print(f"\n📦 Installing missing packages: {', '.join(missing)}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
    
    return True

def setup_env():
    """Setup environment file if it doesn't exist."""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("📝 Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# OpenAI API Key (optional for demo)\n")
            f.write("OPENAI_API_KEY=your_api_key_here\n")
        print("✅ .env file created (you can add your OpenAI API key)")
    else:
        print("✅ .env file exists")

def run_demo():
    """Run the demo application."""
    print("🚀 Starting MarketPulse AI Demo...")
    print("=" * 40)
    
    if check_requirements():
        setup_env()
        
        print("\n🌐 Launching Streamlit app...")
        print("Open your browser to the URL shown below:")
        print("-" * 40)
        
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        except KeyboardInterrupt:
            print("\n👋 Demo stopped.")
        except Exception as e:
            print(f"❌ Error running demo: {e}")

if __name__ == "__main__":
    run_demo()