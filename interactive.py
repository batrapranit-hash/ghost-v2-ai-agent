"""
Ghost V2 - Interactive Terminal Interface
"""

from main import GhostV2

def main():
    ghost = GhostV2(manual=False)
    
    commands = """
╔════════════════════════════════════════╗
║      👻 GHOST V2 COMMANDS             ║
╠════════════════════════════════════════╣
║ tdd <task>        - Test-Driven Dev   ║
║ review <task>     - Peer Review Mode  ║
║ agents <task>     - Use Agents        ║
║ file <name> <purpose> - Create File   ║
║ status            - Show Status       ║
║ memory            - View Memory       ║
║ list              - List Agents       ║
║ help              - Show This Menu    ║
║ exit              - Quit Ghost        ║
╚════════════════════════════════════════╝

Examples:
  tdd Create a binary search function
  review Build a web scraper
  agents Make a Pygame space shooter
  file utils.py Helper functions
"""
    
    print(commands)
    
    while True:
        try:
            cmd = input("\n👻 Ghost> ").strip()
            
            if not cmd:
                continue
            
            parts = cmd.split(" ", 1)
            action = parts[0].lower()
            task = parts[1] if len(parts) > 1 else ""
            
            if action == "tdd" and task:
                print()
                ghost.develop_tdd(task)
                
            elif action == "review" and task:
                print()
                ghost.develop_with_review(task)
                
            elif action == "agents" and task:
                print()
                ghost.develop_with_agents(task)
                
            elif action == "file":
                file_parts = task.split(" ", 1)
                if len(file_parts) == 2:
                    ghost.create_file(file_parts[0], file_parts[1])
                else:
                    print("❌ Usage: file <filename> <purpose>")
                    
            elif action == "status":
                ghost.status()
                
            elif action == "memory":
                print(f"\n📊 Successes: {len(ghost.memory['successes'])}")
                print(f"📊 Failures: {len(ghost.memory['failures'])}")
                
                if ghost.memory['successes']:
                    print("\n✅ Recent Successes:")
                    for exp in ghost.memory['successes'][-5:]:
                        print(f"  • {exp['task']}")
                
                if ghost.memory['failures']:
                    print("\n❌ Recent Failures:")
                    for exp in ghost.memory['failures'][-3:]:
                        print(f"  • {exp['task']}: {exp['error'][:50]}")
                        
            elif action == "list":
                ghost.list_agents()
                
            elif action == "help":
                print(commands)
                
            elif action == "exit":
                print("\n👋 Shutting down Ghost V2...")
                break
                
            else:
                print("❌ Unknown command. Type 'help' for commands.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Shutting down...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
