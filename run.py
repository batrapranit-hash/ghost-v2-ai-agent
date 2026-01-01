"""
Ghost V2 - Quick Start Script
"""

from main import GhostV2

def main():
    print("🚀 Initializing Ghost V2...\n")
    
    # Initialize (manual=False for auto-approve on tablet)
    ghost = GhostV2(manual=False)
    
    # Show status
    ghost.status()
    
    # Example task
    print("\n" + "="*60)
    print("EXAMPLE: Creating a function")
    print("="*60 + "\n")
    
    ghost.develop_tdd("Create a function to check if a number is prime")
    
    print("\n✅ Done! Check src/main.py for generated code")
    print("📝 Tests saved in tests/ directory")

if __name__ == "__main__":
    main()
