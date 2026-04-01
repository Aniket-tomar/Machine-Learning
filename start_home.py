import os
import subprocess
import sys
import time
from flask import Flask, redirect, render_template_string

app = Flask(__name__)

# Dictionary to hold the state of running models
running_processes = {}

MODELS = {
    'pest': {'dir': 'PestDetection', 'cmd': 'python app.py', 'port': 5001},
    'nlp': {'dir': 'audio-Indian Languages', 'cmd': 'python -m streamlit run app.py --server.port 8501 --server.headless true', 'port': 8501},
    'sentiment': {
        'type': 'multi',  # Needs both backend API + Vite frontend
        'processes': [
            {'name': 'sentiment-api', 'dir': os.path.join('sentimenrtal analysis', 'backend'), 'cmd': 'python api.py', 'port': 8502},
            {'name': 'sentiment-ui', 'dir': os.path.join('sentimenrtal analysis', 'frontend'), 'cmd': 'npm run dev -- --port 5173 --host', 'port': 5173},
        ],
        'port': 5173,  # The port users are redirected to (the Vite frontend)
    },
    'salary': {'dir': 'Numeric Data', 'cmd': 'python app.py', 'port': 5002},
    'deepfake': {'dir': 'deepfake detection', 'cmd': 'python -m streamlit run app3.py --server.port 8503 --server.headless true', 'port': 8503}
}

base_dir = os.path.dirname(os.path.abspath(__file__))

def check_port(port):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(('127.0.0.1', port))
        s.close()
        return True
    except ConnectionRefusedError:
        return False

@app.route("/")
def index():
    # Read the STITCH code.html
    html_path = os.path.join(base_dir, "stitch_machine_learning_project_home", "code.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Replace hrefs to point to our launch route
    html_content = html_content.replace('href="#pest-detection"', 'href="/launch/pest" target="_blank"')
    html_content = html_content.replace('href="#language-detection"', 'href="/launch/nlp" target="_blank"')
    html_content = html_content.replace('href="#sentiment-analysis"', 'href="/launch/sentiment" target="_blank"')
    html_content = html_content.replace('href="#salary-prediction"', 'href="/launch/salary" target="_blank"')
    html_content = html_content.replace('href="#deepfake-detection"', 'href="/launch/deepfake" target="_blank"')
    
    return render_template_string(html_content)

def launch_single(model_id, model_info):
    """Launch a single-process model (Flask or Streamlit)."""
    port = model_info['port']
    model_dir = os.path.join(base_dir, model_info['dir'])
    cmd = model_info['cmd']

    # Check if already running on the port
    if check_port(port):
        return redirect(f"http://127.0.0.1:{port}")

    # Start it if not running
    print(f"Starting {model_id} in {model_dir} on port {port}...")
    import shlex
    proc = subprocess.Popen(shlex.split(cmd), cwd=model_dir)
    running_processes[model_id] = proc

    # Wait until it's ready (up to 30 seconds)
    started = False
    for _ in range(60):
        time.sleep(0.5)
        if check_port(port):
            started = True
            break

    if started:
        time.sleep(1.0)
        return redirect(f"http://127.0.0.1:{port}")
    else:
        return (f"Failed to start {model_id} on port {port} within 30 seconds. "
                "Please try again or check console logs."), 500

def launch_multi(model_id, model_info):
    """Launch a multi-process model (e.g. backend API + frontend dev server)."""
    frontend_port = model_info['port']

    # Check if the frontend is already running
    if check_port(frontend_port):
        return redirect(f"http://127.0.0.1:{frontend_port}")

    # Start each sub-process
    for proc_info in model_info['processes']:
        proc_name = proc_info['name']
        proc_port = proc_info['port']
        proc_dir = os.path.join(base_dir, proc_info['dir'])
        proc_cmd = proc_info['cmd']

        if check_port(proc_port):
            print(f"  {proc_name} already running on port {proc_port}")
            continue

        print(f"  Starting {proc_name} in {proc_dir} on port {proc_port}...")

        # Use shell=True on Windows for npm commands
        use_shell = sys.platform == 'win32' and proc_cmd.startswith('npm')
        if use_shell:
            proc = subprocess.Popen(proc_cmd, cwd=proc_dir, shell=True)
        else:
            import shlex
            proc = subprocess.Popen(shlex.split(proc_cmd), cwd=proc_dir)

        running_processes[proc_name] = proc

    # Wait for the frontend port to become available (up to 45 seconds for npm)
    started = False
    for _ in range(90):
        time.sleep(0.5)
        if check_port(frontend_port):
            started = True
            break

    if started:
        time.sleep(1.5)
        return redirect(f"http://127.0.0.1:{frontend_port}")
    else:
        return (f"Failed to start {model_id} on port {frontend_port} within 45 seconds. "
                "Please try again or check console logs."), 500

@app.route("/launch/<model_id>")
def launch(model_id):
    if model_id not in MODELS:
        return "Model not found", 404

    model_info = MODELS[model_id]

    if model_info.get('type') == 'multi':
        return launch_multi(model_id, model_info)
    else:
        return launch_single(model_id, model_info)

if __name__ == "__main__":
    print("Starting ML Project Hub on http://127.0.0.1:8000...")
    app.run(port=8000, debug=False)
