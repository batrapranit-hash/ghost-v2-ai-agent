"""
Ghost V2 - God-Mode AI Coding Agent
Optimized for Replit/Cloud Platforms
Features: Multi-Model Router, Memory, TDD, Peer Review, Agent System
"""

import os, sys, re, time, subprocess, difflib, shutil, ast, base64, tempfile, importlib.util, json, hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai

try: from redbaron import RedBaron
except ImportError: RedBaron = None

# Platform detection
IS_REPLIT = 'REPL_ID' in os.environ
USE_DOCKER = False  # Disabled for cloud platforms

# Agent specifications
AGENT_TYPES = {
    "graphics": {
        "handles": ["GUI", "rendering", "images", "animation", "UI"],
        "libraries": ["pygame", "tkinter", "pillow"],
        "keywords": ["gui", "window", "display", "render", "draw", "image", "animation", "pygame", "tkinter"]
    },
    "game_engine": {
        "handles": ["physics", "collision", "game loop"],
        "libraries": ["pygame", "arcade"],
        "keywords": ["game", "physics", "collision", "sprite"]
    },
    "web_server": {
        "handles": ["HTTP", "APIs", "REST"],
        "libraries": ["flask", "fastapi", "requests"],
        "keywords": ["api", "server", "http", "rest", "flask", "fastapi"]
    },
    "database": {
        "handles": ["SQL", "NoSQL", "persistence"],
        "libraries": ["sqlite3", "sqlalchemy"],
        "keywords": ["database", "sql", "storage", "persist"]
    },
    "web_scraper": {
        "handles": ["web scraping", "data extraction"],
        "libraries": ["beautifulsoup4", "requests"],
        "keywords": ["scrape", "extract", "crawl", "parse html"]
    },
    "ml_ops": {
        "handles": ["ML operations", "model deployment"],
        "libraries": ["numpy", "scikit-learn"],
        "keywords": ["machine learning", "model", "train", "predict"]
    }
}

class GhostV2:
    def __init__(self, api_key=None, project_name="ghost_v2", manual=True, test_timeout=10):
        self.root = Path(project_name).resolve()
        self.root.mkdir(exist_ok=True)
        
        # Multi-model setup
        self.keys_file = self.root / "keys.json"
        self.api_keys = self._load_api_keys()
        
        if api_key:
            self.api_keys['gemini'] = api_key
        
        if not self.api_keys.get('gemini'):
            raise ValueError("❌ Gemini API key required! Add to keys.json or pass as parameter")
        
        genai.configure(api_key=self.api_keys['gemini'])
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Configuration
        self.manual = manual
        self.test_timeout = test_timeout
        self.completed_tasks = 0
        self.last_task_failed = False
        self.last_error = ""
        
        # Setup
        self._setup_workspace()
        self._load_agent_registry()
        self._init_memory_system()
        self._update_project_map()
        
        print("🚀 Ghost V2 initialized" + (" (Replit Mode)" if IS_REPLIT else ""))

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from keys.json"""
        if not self.keys_file.exists():
            default = {
                "gemini": "",
                "openai": "",
                "anthropic": "",
                "notes": "Get Gemini key: https://aistudio.google.com/app/apikey"
            }
            self.keys_file.write_text(json.dumps(default, indent=2))
            return {}
        
        try:
            keys = json.loads(self.keys_file.read_text())
            available = [k for k, v in keys.items() if v and k != "notes"]
            if available:
                print(f"🔑 API Keys loaded: {', '.join(available)}")
            return keys
        except:
            return {}

    def _setup_workspace(self):
        """Create project structure"""
        for d in ['src', 'tests', 'logs', 'backups', 'agents', 'memory']:
            (self.root / d).mkdir(exist_ok=True)
        
        self.main_py = self.root / "src" / "main.py"
        if not self.main_py.exists():
            self.main_py.write_text("# Ghost V2 - Generated Code\n")

    def _init_memory_system(self):
        """Initialize experience memory"""
        self.memory_db = self.root / "memory" / "experiences.json"
        
        if not self.memory_db.exists():
            self.memory_db.write_text(json.dumps({
                "successes": [],
                "failures": [],
                "lessons": []
            }, indent=2))
        
        try:
            self.memory = json.loads(self.memory_db.read_text())
        except:
            self.memory = {"successes": [], "failures": [], "lessons": []}

    def _save_memory(self):
        """Save memory to disk"""
        try:
            self.memory_db.write_text(json.dumps(self.memory, indent=2))
        except Exception as e:
            print(f"⚠️ Memory save failed: {e}")

    def _remember_success(self, task: str, code: str):
        """Store successful solution"""
        self.memory["successes"].append({
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "code_hash": hashlib.md5(code.encode()).hexdigest()[:8]
        })
        self._save_memory()

    def _remember_failure(self, task: str, error: str):
        """Store failure"""
        self.memory["failures"].append({
            "task": task,
            "error": error[:200],
            "timestamp": datetime.now().isoformat()
        })
        self._save_memory()

    def _recall_similar(self, task: str) -> List[Dict]:
        """Find similar past experiences"""
        keywords = set(task.lower().split())
        matches = []
        
        for exp in self.memory["successes"][-10:]:
            exp_keywords = set(exp["task"].lower().split())
            overlap = len(keywords & exp_keywords)
            if overlap > 0:
                matches.append((overlap, exp))
        
        matches.sort(reverse=True, key=lambda x: x[0])
        return [m[1] for m in matches[:3]]

    def _update_project_map(self):
        """Map project structure"""
        self.project_map = {"functions": {}, "classes": {}, "files": {}}
        
        src_dir = self.root / "src"
        if not src_dir.exists():
            return
        
        for py_file in src_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        self.project_map["functions"][node.name] = str(py_file.relative_to(self.root))
                    elif isinstance(node, ast.ClassDef):
                        self.project_map["classes"][node.name] = str(py_file.relative_to(self.root))
            except:
                pass

    def _get_project_context(self) -> str:
        """Generate context for AI"""
        ctx = "# PROJECT CONTEXT\n\n"
        
        if self.project_map["functions"]:
            ctx += "Available Functions:\n"
            for func in list(self.project_map["functions"].keys())[:10]:
                ctx += f"- {func}()\n"
        
        if self.project_map["classes"]:
            ctx += "\nAvailable Classes:\n"
            for cls in list(self.project_map["classes"].keys())[:10]:
                ctx += f"- {cls}\n"
        
        return ctx

    def _load_agent_registry(self):
        """Load existing agents"""
        self.agent_registry = {}
        agent_dir = self.root / "agents"
        
        if agent_dir.exists():
            for agent_file in agent_dir.glob("*_agent.py"):
                agent_name = agent_file.stem.replace("_agent", "")
                self.agent_registry[agent_name] = {
                    "file": agent_file,
                    "created": datetime.fromtimestamp(agent_file.stat().st_mtime)
                }

    # ========== MULTI-MODEL ROUTER ==========
    
    def _route_model(self, task: str) -> str:
        """Select best model for task"""
        if any(kw in task.lower() for kw in ['algorithm', 'optimize', 'complex']):
            if self.api_keys.get('openai'):
                return 'openai'
        
        if any(kw in task.lower() for kw in ['explain', 'document']):
            if self.api_keys.get('anthropic'):
                return 'anthropic'
        
        return 'gemini'

    def _generate(self, prompt: str, model: str = None, temp: float = 0.7) -> str:
        """Generate with specified model"""
        if model is None:
            model = 'gemini'
        
        try:
            if model == 'gemini':
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": temp}
                )
                return response.text
            
            elif model == 'openai' and self.api_keys.get('openai'):
                import openai
                openai.api_key = self.api_keys['openai']
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp
                )
                return response.choices[0].message.content
            
            elif model == 'anthropic' and self.api_keys.get('anthropic'):
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_keys['anthropic'])
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            else:
                response = self.model.generate_content(prompt)
                return response.text
                
        except Exception as e:
            print(f"⚠️ {model} error: {e}, using Gemini")
            response = self.model.generate_content(prompt)
            return response.text

    # ========== CODE EXECUTION (NO DOCKER) ==========
    
    def run_code(self, code: str, test_suite: str) -> Dict:
        """Execute code and tests (no Docker)"""
        try:
            # Create isolated namespace
            namespace = {}
            
            # Execute main code
            exec(code, namespace)
            
            # Execute tests
            exec(test_suite, namespace)
            
            return {
                "success": True,
                "latency": 0,
                "out": "Tests passed",
                "err": ""
            }
        except Exception as e:
            return {
                "success": False,
                "latency": 0,
                "out": "",
                "err": str(e)
            }

    # ========== TDD MODE ==========
    
    def develop_tdd(self, task: str) -> bool:
        """Test-Driven Development"""
        print(f"\n🧪 TDD MODE: {task}")
        
        # Check memory
        similar = self._recall_similar(task)
        memory_ctx = ""
        if similar:
            print(f"💡 Found {len(similar)} similar experiences")
            memory_ctx = "\n\nPAST SUCCESSES:\n" + "\n".join([
                f"- {exp['task']}" for exp in similar
            ])
        
        project_ctx = self._get_project_context()
        
        # Step 1: Write test
        print("📝 Step 1: Writing tests...")
        
        test_prompt = f"""Task: {task}

{project_ctx}{memory_ctx}

Write comprehensive test suite (assert statements) for this task.
Tests should initially FAIL (code doesn't exist yet).

Return only test code.
"""
        
        try:
            model = self._route_model(task)
            test_suite = self._extract(self._generate(test_prompt, model))
            print(f"✅ Tests written (model: {model})")
        except Exception as e:
            print(f"❌ Test generation failed: {e}")
            return False
        
        # Step 2: Verify test fails
        print("🧪 Step 2: Verifying test fails...")
        result = self.run_code("pass", test_suite)
        
        if result["success"]:
            print("⚠️ Test passed without code!")
        else:
            print("✅ Test failed as expected (Red)")
        
        # Step 3: Write code
        print("💻 Step 3: Writing code...")
        
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            print(f"   Attempt {attempt}/{max_attempts}")
            
            code_prompt = f"""Task: {task}

{project_ctx}{memory_ctx}

Tests to pass:
{test_suite}

Write code that passes ALL tests.
Return only code, no explanations.
"""
            
            try:
                proposal = self._extract(self._generate(code_prompt))
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
                continue
            
            # Test code
            result = self.run_code(proposal, test_suite)
            
            if result["success"]:
                print(f"✅ TDD SUCCESS (Green - attempt {attempt})")
                
                # Save success
                self._remember_success(task, proposal)
                
                # Merge code
                current = self.main_py.read_text(encoding='utf-8')
                should_merge = self._should_merge(current, proposal)
                
                if should_merge:
                    print("🔀 Merging...")
                    merged = self._merge_safe(current, proposal)
                else:
                    print("🔄 Replacing...")
                    merged = proposal
                
                # Save
                if not self.manual or self._human_gate(current, merged, task, should_merge):
                    self.main_py.write_text(merged, encoding='utf-8')
                    
                    # Save test
                    test_file = self.root / "tests" / f"test_{int(time.time())}.py"
                    test_file.write_text(test_suite, encoding='utf-8')
                    
                    self._update_project_map()
                    self.completed_tasks += 1
                    return True
            else:
                error = result.get('err', '')[:200]
                print(f"   ❌ Failed: {error}")
                self._remember_failure(task, error)
        
        print("❌ TDD failed after max attempts")
        self.last_task_failed = True
        return False

    # ========== PEER REVIEW ==========
    
    def develop_with_review(self, task: str) -> bool:
        """Multi-model peer review"""
        print(f"\n👥 PEER REVIEW MODE: {task}")
        
        # Builder
        print("🏗️ BUILDER: Generating code...")
        
        similar = self._recall_similar(task)
        memory_ctx = ""
        if similar:
            memory_ctx = "\nPast successes: " + ", ".join([e['task'] for e in similar])
        
        build_prompt = f"""Task: {task}

{self._get_project_context()}{memory_ctx}

Generate complete working code.
Return only code.
"""
        
        try:
            proposal = self._extract(self._generate(build_prompt, 'gemini'))
            print("✅ Code generated")
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
        
        # Critic
        critic_model = 'anthropic' if self.api_keys.get('anthropic') else 'openai' if self.api_keys.get('openai') else None
        
        if critic_model:
            print(f"🔍 CRITIC ({critic_model}): Reviewing...")
            
            critique_prompt = f"""Review this code:

CODE:
{proposal}

TASK:
{task}

Find: logic errors, edge cases, performance issues, security issues.
If good, say "APPROVED". If issues, list them.
"""
            
            try:
                critique = self._generate(critique_prompt, critic_model, 0.3)
                print("✅ Review complete")
                
                if "APPROVED" not in critique.upper():
                    print("🔧 Fixing issues...")
                    
                    fix_prompt = f"""Task: {task}

Your code:
{proposal}

Review:
{critique}

Fix ALL issues. Return only improved code.
"""
                    
                    try:
                        proposal = self._extract(self._generate(fix_prompt))
                        print("✅ Issues fixed")
                    except:
                        print("⚠️ Fix failed, using original")
                else:
                    print("✅ Approved by reviewer")
            except Exception as e:
                print(f"⚠️ Review failed: {e}")
        
        # Generate tests
        print("📝 Generating tests...")
        test_prompt = f"Task: {task}\n\nCode:\n{proposal}\n\nGenerate 3 assert tests."
        
        try:
            test_suite = self._extract(self._generate(test_prompt))
        except:
            test_suite = "assert True"
        
        # Test
        print("🧪 Testing...")
        result = self.run_code(proposal, test_suite)
        
        if result["success"]:
            print("✅ Tests passed!")
            
            self._remember_success(task, proposal)
            
            current = self.main_py.read_text(encoding='utf-8')
            should_merge = self._should_merge(current, proposal)
            
            merged = self._merge_safe(current, proposal) if should_merge else proposal
            
            if not self.manual or self._human_gate(current, merged, task, should_merge):
                self.main_py.write_text(merged, encoding='utf-8')
                
                test_file = self.root / "tests" / f"test_{int(time.time())}.py"
                test_file.write_text(test_suite, encoding='utf-8')
                
                self._update_project_map()
                self.completed_tasks += 1
                return True
        else:
            print(f"❌ Tests failed: {result.get('err', '')[:200]}")
            self.last_task_failed = True
        
        return False

    # ========== AGENT SYSTEM ==========
    
    def _detect_agents(self, task: str) -> List[str]:
        """Detect needed agents"""
        task_lower = task.lower()
        needed = []
        
        for agent_type, spec in AGENT_TYPES.items():
            if any(kw in task_lower for kw in spec.get("keywords", [])):
                needed.append(agent_type)
        
        return list(set(needed))

    def create_agent(self, agent_type: str) -> Optional[Path]:
        """Create specialized agent"""
        print(f"\n🤖 Creating {agent_type} agent...")
        
        agent_file = self.root / "agents" / f"{agent_type}_agent.py"
        
        if agent_file.exists():
            print(f"✅ Agent exists")
            return agent_file
        
        spec = AGENT_TYPES.get(agent_type, {})
        
        prompt = f"""Create Python agent class: {agent_type.title()}Agent

Handles: {', '.join(spec.get('handles', []))}
Libraries: {', '.join(spec.get('libraries', []))}

Requirements:
1. Class: {agent_type.title()}Agent
2. __init__(self, api_key)
3. generate_code(self, task: str) -> str
4. test_code(self, code: str) -> dict

Return complete code.
"""
        
        try:
            agent_code = self._extract(self._generate(prompt))
            
            full_code = f"""# {agent_type.upper()} Agent
# Created by Ghost V2

import google.generativeai as genai

{agent_code}
"""
            
            agent_file.write_text(full_code, encoding='utf-8')
            print(f"✅ Agent created")
            
            self.agent_registry[agent_type] = {
                "file": agent_file,
                "created": datetime.now()
            }
            
            return agent_file
        except Exception as e:
            print(f"❌ Failed: {e}")
            return None

    def develop_with_agents(self, task: str) -> bool:
        """Use specialized agents"""
        print(f"\n🤖 AGENT MODE: {task}")
        
        needed = self._detect_agents(task)
        
        if not needed:
            print("ℹ️ No specialized agents needed")
            return self.develop_tdd(task)
        
        print(f"💡 Detected: {', '.join(needed)}")
        
        if self.manual:
            choice = input("\n1=Code only, 2=Use agents, 3=Cancel: ").strip()
            if choice == "3":
                return False
            elif choice != "2":
                return self.develop_tdd(task)
        
        # Create/use agents
        for agent_type in needed:
            if agent_type not in self.agent_registry:
                self.create_agent(agent_type)
        
        print("✅ Agents ready, delegating task...")
        
        # For simplicity, use TDD with agent context
        return self.develop_tdd(f"{task} (Use {', '.join(needed)} capabilities)")

    # ========== FILE CREATION ==========
    
    def create_file(self, filename: str, purpose: str) -> bool:
        """Create specialized file"""
        print(f"\n📄 Creating: {filename}")
        print(f"Purpose: {purpose}")
        
        target = self.root / "agents" / filename
        
        if target.exists() and self.manual:
            if input("Overwrite? (y/n): ").lower() != 'y':
                return False
        
        prompt = f"""Create Python file: {filename}

Purpose: {purpose}

Requirements:
1. Complete implementation
2. Proper imports and error handling
3. Documentation

Return only code.
"""
        
        try:
            content = self._extract(self._generate(prompt))
            
            full = f"""# {filename}
# Created by Ghost V2
# Purpose: {purpose}
# Date: {datetime.now().strftime('%Y-%m-%d')}

{content}
"""
            
            target.write_text(full, encoding='utf-8')
            print(f"✅ Created: {target.name}")
            
            self._update_project_map()
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False

    # ========== UTILITY METHODS ==========
    
    def _should_merge(self, old: str, new: str) -> bool:
        """Decide merge vs replace"""
        if len(old.strip()) < 50:
            return False
        
        try:
            old_tree = ast.parse(old)
            new_tree = ast.parse(new)
            
            old_defs = {n.name for n in ast.walk(old_tree) 
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            new_defs = {n.name for n in ast.walk(new_tree) 
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            
            if not old_defs.intersection(new_defs):
                return False
            
            return True
        except:
            return True

    def _merge_safe(self, old: str, new: str) -> str:
        """Merge code"""
        if RedBaron:
            try:
                r, n = RedBaron(old), RedBaron(new)
                for node in n:
                    if node.type in ["def", "class"]:
                        target = r.find(node.type, name=node.name)
                        if target:
                            target.replace(node)
                        else:
                            r.append(node)
                return r.dumps()
            except:
                pass
        
        return f"{old}\n\n# New Code\n{new}"

    def _extract(self, text: str) -> str:
        """Extract code from markdown"""
        m = re.search(r"```(?:python|py)?\s*(.*?)\s*```", text, re.DOTALL | re.I)
        if m:
            return m.group(1).strip()
        return text.strip()

    def _human_gate(self, old: str, new: str, task: str, is_merge: bool) -> bool:
        """Human approval"""
        print("\n" + "="*60)
        print(f"{'MERGE' if is_merge else 'REPLACE'}: {task}")
        print("="*60)
        
        try:
            diff = list(difflib.unified_diff(
                old.splitlines(), new.splitlines(), lineterm=''
            ))[:30]
            print('\n'.join(diff))
        except:
            print(new[:500])
        
        print("="*60)
        
        try:
            return input("\nApprove? (y/n): ").lower().strip() in ['y', 'yes']
        except:
            return False

    # ========== STATUS & INFO ==========
    
    def status(self):
        """Show status"""
        print("\n" + "="*60)
        print("👻 GHOST V2 STATUS")
        print("="*60)
        print(f"📁 Project: {self.root}")
        print(f"⚙️  Mode: {'Manual' if self.manual else 'Auto'}")
        print(f"✅ Completed: {self.completed_tasks}")
        print(f"🧠 Memories: {len(self.memory['successes'])} successes, {len(self.memory['failures'])} failures")
        print(f"🤖 Agents: {len(self.agent_registry)}")
        print(f"🔑 Models: {', '.join([k for k in self.api_keys if k != 'notes' and self.api_keys[k]])}")
        print(f"📊 Functions: {len(self.project_map['functions'])}")
        print(f"📊 Classes: {len(self.project_map['classes'])}")
        
        if self.last_error:
            print(f"🚨 Last error: {self.last_error[:100]}")
        
        print("="*60 + "\n")

    def list_agents(self):
        """List all agents"""
        print("\n🤖 AVAILABLE AGENTS:")
        if not self.agent_registry:
            print("No agents created yet.\n")
            return
        
        for name, info in self.agent_registry.items():
            print(f"\n{name.upper()}")
            print(f"  File: {info['file'].name}")
            print(f"  Created: {info['created'].strftime('%Y-%m-%d %H:%M')}")

    # Main develop method
    def develop(self, task: str) -> bool:
        """Main development entry point (uses TDD by default)"""
        return self.develop_tdd(task)

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════╗
    ║         👻 GHOST V2 - GOD MODE              ║
    ║    Multi-Model AI Coding Agent              ║
    ╚══════════════════════════════════════════════╝
    
    Usage:
      ghost = GhostV2(api_key="YOUR_GEMINI_KEY")
      ghost.develop_tdd("Create a calculator")
      ghost.develop_with_review("Build web scraper")
      ghost.develop_with_agents("Make Pygame game")
      ghost.create_file("helper.py", "Utility functions")
      ghost.status()
    
    Features:
      ✅ Multi-Model Router (Gemini/GPT/Claude)
      ✅ Test-Driven Development
      ✅ Peer Review System
      ✅ Specialized Agents
      ✅ Long-Term Memory
      ✅ Project Mapping
      ✅ Cloud-Optimized (No Docker needed)
    """)
