"""
Ghost V2 - Web Interface (Tablet-Optimized)
Run: python web.py
Then open the URL in your tablet browser
"""

from flask import Flask, request, jsonify, render_template_string
from main import GhostV2
import sys, io

app = Flask(__name__)
ghost = GhostV2(manual=False)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ghost V2 Control</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 15px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.98);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        
        h1 {
            text-align: center;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
            font-size: 0.9em;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
            font-size: 0.95em;
        }
        
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-family: inherit;
            resize: vertical;
            transition: all 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .btn-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }
        
        button {
            padding: 16px 20px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .btn-tdd { background: linear-gradient(135deg, #4CAF50, #45a049); }
        .btn-review { background: linear-gradient(135deg, #2196F3, #1976D2); }
        .btn-agents { background: linear-gradient(135deg, #FF9800, #F57C00); }
        .btn-status { background: linear-gradient(135deg, #607D8B, #455A64); }
        
        button:active {
            transform: scale(0.98);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
            background: #f5f5f5;
            border-radius: 12px;
            margin-top: 20px;
        }
        
        .loading.show {
            display: block;
            animation: slideIn 0.3s;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result {
            display: none;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            max-height: 500px;
            overflow-y: auto;
            line-height: 1.6;
        }
        
        .result.show {
            display: block;
            animation: slideIn 0.3s;
        }
        
        .result.success {
            border-left-color: #4CAF50;
            background: #f1f8f4;
        }
        
        .result.error {
            border-left-color: #f44336;
            background: #fef1f0;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.85em;
            opacity: 0.9;
        }
        
        @media (max-width: 600px) {
            h1 { font-size: 2em; }
            .btn-grid { grid-template-columns: 1fr; }
            .stats { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>👻 Ghost V2</h1>
        <div class="subtitle">God-Mode AI Coding Agent</div>
        
        <div class="stats" id="stats" style="display: none;">
            <div class="stat-card">
                <div class="stat-value" id="tasks">0</div>
                <div class="stat-label">Tasks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="agents">0</div>
                <div class="stat-label">Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="memory">0</div>
                <div class="stat-label">Memories</div>
            </div>
        </div>
        
        <div class="input-group">
            <label for="task">What should Ghost build?</label>
            <textarea 
                id="task" 
                placeholder="Example:&#10;• Create a calculator class&#10;• Build a web scraper for news&#10;• Make a Pygame space shooter&#10;• Implement quicksort algorithm"
            ></textarea>
        </div>
        
        <div class="btn-grid">
            <button class="btn-tdd" onclick="execute('tdd')">
                🧪 TDD Mode
            </button>
            <button class="btn-review" onclick="execute('review')">
                👥 Peer Review
            </button>
            <button class="btn-agents" onclick="execute('agents')">
                🤖 Use Agents
            </button>
            <button class="btn-status" onclick="execute('status')">
                📊 Status
            </button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="color: #667eea; font-weight: bold;">Ghost is working...</p>
            <p style="color: #999; font-size: 0.9em; margin-top: 5px;">This may take 10-30 seconds</p>
        </div>
        
        <div class="result" id="result"></div>
    </div>
    
    <script>
        let isProcessing = false;
        
        async function execute(action) {
            if (isProcessing) {
                alert('Ghost is already working on a task!');
                return;
            }
            
            const task = document.getElementById('task').value.trim();
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            if (!task && action !== 'status') {
                alert('Please enter a task first!');
                document.getElementById('task').focus();
                return;
            }
            
            isProcessing = true;
            loading.classList.add('show');
            result.classList.remove('show', 'success', 'error');
            
            // Disable all buttons
            document.querySelectorAll('button').forEach(btn => btn.disabled = true);
            
            try {
                const response = await fetch('/execute', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action, task})
                });
                
                const data = await response.json();
                
                result.textContent = data.result;
                result.classList.add('show', data.success ? 'success' : 'error');
                
                // Update stats if status
                if (action === 'status' && data.stats) {
                    document.getElementById('stats').style.display = 'grid';
                    document.getElementById('tasks').textContent = data.stats.tasks;
                    document.getElementById('agents').textContent = data.stats.agents;
                    document.getElementById('memory').textContent = data.stats.memory;
                }
                
            } catch (error) {
                result.textContent = '❌ Error: ' + error.message + '\n\nMake sure Ghost V2 is properly initialized.';
                result.classList.add('show', 'error');
            } finally {
                loading.classList.remove('show');
                isProcessing = false;
                
                // Re-enable buttons
                document.querySelectorAll('button').forEach(btn => btn.disabled = false);
            }
        }
        
        // Keyboard shortcut: Ctrl+Enter to run TDD
        document.getElementById('task').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                execute('tdd');
            }
        });
        
        // Load initial status on page load
        window.addEventListener('load', function() {
            setTimeout(() => execute('status'), 500);
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/execute", methods=["POST"])
def execute():
    data = request.json
    action = data.get("action")
    task = data.get("task", "")
    
    result = ""
    success = True
    stats = None
    
    try:
        if action == "tdd":
            ghost.develop_tdd(task)
            result = f"✅ TDD Mode Completed!\n\n📝 Task: {task}\n\n✨ Code generated and tested\n📂 Check: src/main.py\n🧪 Tests: tests/ directory"
            
        elif action == "review":
            ghost.develop_with_review(task)
            result = f"✅ Peer Review Completed!\n\n📝 Task: {task}\n\n👥 Reviewed by multiple AI models\n📂 Check: src/main.py"
            
        elif action == "agents":
            ghost.develop_with_agents(task)
            result = f"✅ Agent Mode Completed!\n\n📝 Task: {task}\n\n🤖 Specialized agents handled this task\n📂 Check: src/main.py"
            
        elif action == "status":
            # Capture status output
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            ghost.status()
            result = buffer.getvalue()
            sys.stdout = old_stdout
            
            stats = {
                "tasks": ghost.completed_tasks,
                "agents": len(ghost.agent_registry),
                "memory": len(ghost.memory['successes']) + len(ghost.memory['failures'])
            }
            
    except Exception as e:
        result = f"❌ Error occurred:\n\n{str(e)}\n\nPlease check the console for details."
        success = False
    
    return jsonify({
        "result": result,
        "success": success,
        "stats": stats
    })

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════╗
    ║     👻 GHOST V2 WEB INTERFACE               ║
    ╠══════════════════════════════════════════════╣
    ║  Starting web server...                      ║
    ║  Open the URL shown below in your browser    ║
    ╚══════════════════════════════════════════════╝
    """)
    
    # Install Flask if not available
    try:
        import flask
    except ImportError:
        print("📦 Installing Flask...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
        print("✅ Flask installed!\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False)
