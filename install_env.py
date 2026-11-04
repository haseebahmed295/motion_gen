import os
import sys
import urllib.request
import zipfile
import subprocess
import platform
import shutil

def check_mac_build_tools():
    """Checks if macOS has the necessary toolchain to build C++ extensions from source."""
    issues = []
    
    # Check Xcode Command Line Tools
    try:
        subprocess.run(["xcode-select", "-p"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("Xcode Command Line Tools missing (run 'xcode-select --install' in Terminal)")
        
    # Check CMake
    if not shutil.which("cmake"):
        issues.append("CMake not found in PATH (install via 'brew install cmake')")
        
    return issues

def reporthook(count, block_size, total_size):
    if count % 1000 == 0:
        percent = min(int(count * block_size * 100 / total_size), 100)
        print(f"\rDownloading... {percent}%", end="", flush=True)

def install_standalone_python():
    addon_dir = os.path.dirname(os.path.realpath(__file__))
    runtime_dir = os.path.join(addon_dir, "runtime")
    
    os_name = platform.system()
    arch = platform.machine().lower()
    
    if os_name == "Windows":
        python_exe = os.path.join(runtime_dir, "python.exe")
        pip_exe = os.path.join(runtime_dir, "Scripts", "pip.exe")
        
        if not os.path.exists(python_exe):
            os.makedirs(runtime_dir, exist_ok=True)
            
            if "arm" in arch or "aarch" in arch:
                py_url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-arm64.zip"
            elif "amd64" in arch or "x86_64" in arch:
                py_url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
            else:
                py_url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-win32.zip"
                
            py_zip_path = os.path.join(addon_dir, "python_embed.zip")
            print(f"Downloading Python 3.11 ({arch}) from {py_url}...")
            urllib.request.urlretrieve(py_url, py_zip_path, reporthook=reporthook)
            print("\nDownload complete. Extracting...")
            with zipfile.ZipFile(py_zip_path, 'r') as zip_ref:
                zip_ref.extractall(runtime_dir)
            os.remove(py_zip_path)
            
            pth_file = os.path.join(runtime_dir, "python311._pth")
            with open(pth_file, 'r') as file:
                pth_data = file.read()
            pth_data = pth_data.replace("#import site", "import site")
            with open(pth_file, 'w') as file:
                file.write(pth_data)
                
        if not os.path.exists(pip_exe):
            print("Bootstrapping PIP...")
            get_pip_path = os.path.join(runtime_dir, "get-pip.py")
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)
            subprocess.run([python_exe, get_pip_path], check=True)
            os.remove(get_pip_path)
    else:
        # macOS / Linux: Download indygreg standalone Python
        import tarfile
        python_exe = os.path.join(runtime_dir, "python", "bin", "python3")
        pip_exe = os.path.join(runtime_dir, "python", "bin", "pip3")
        
        if not os.path.exists(python_exe) and not os.path.exists(os.path.join(runtime_dir, "bin", "python3")):
            os.makedirs(runtime_dir, exist_ok=True)
            if os_name == "Darwin":
                if "arm" in arch or "aarch" in arch:
                    py_url = "https://github.com/indygreg/python-build-standalone/releases/download/20240726/cpython-3.11.9+20240726-aarch64-apple-darwin-install_only.tar.gz"
                else:
                    py_url = "https://github.com/indygreg/python-build-standalone/releases/download/20240726/cpython-3.11.9+20240726-x86_64-apple-darwin-install_only.tar.gz"
            else:
                if "arm" in arch or "aarch" in arch:
                    py_url = "https://github.com/indygreg/python-build-standalone/releases/download/20240726/cpython-3.11.9+20240726-aarch64-unknown-linux-gnu-install_only.tar.gz"
                else:
                    py_url = "https://github.com/indygreg/python-build-standalone/releases/download/20240726/cpython-3.11.9+20240726-x86_64-unknown-linux-gnu-install_only.tar.gz"

            py_tar_path = os.path.join(addon_dir, "python_standalone.tar.gz")
            print(f"Downloading Standalone Python 3.11 for {os_name} ({arch}) from {py_url}...")
            urllib.request.urlretrieve(py_url, py_tar_path, reporthook=reporthook)
            print("\nDownload complete. Extracting...")
            with tarfile.open(py_tar_path, "r:gz") as tar_ref:
                tar_ref.extractall(runtime_dir)
            os.remove(py_tar_path)
            
        # Support both nested (runtime/python/bin/python3) and flat (runtime/bin/python3)
        if not os.path.exists(python_exe):
            alt_py = os.path.join(runtime_dir, "bin", "python3")
            if os.path.exists(alt_py):
                python_exe = alt_py
                pip_exe = os.path.join(runtime_dir, "bin", "pip3")
                
        # Ensure binaries are executable
        if os.path.exists(python_exe):
            os.chmod(python_exe, 0o755)
        if os.path.exists(pip_exe):
            os.chmod(pip_exe, 0o755)
                
        if not os.path.exists(pip_exe):
            print("Bootstrapping PIP...")
            get_pip_path = os.path.join(runtime_dir, "get-pip.py")
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)
            subprocess.run([python_exe, get_pip_path], check=True)
            os.remove(get_pip_path)

    print("\nInstalling PyTorch...")
    torch_deps = ["torch", "torchvision", "torchaudio"]
    torch_cmd = [pip_exe, "install"] + torch_deps + ["--no-warn-script-location"]
    
    if os_name != "Darwin":
        # Force CUDA on Windows/Linux
        torch_cmd.extend(["--index-url", "https://download.pytorch.org/whl/cu130"])
    
    proc1 = subprocess.Popen(torch_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='replace')
    for line in proc1.stdout:
        print(line, end='', flush=True)
    proc1.wait()

    if proc1.returncode != 0:
        print("Error during PyTorch installation.")
        sys.exit(1)

    print("\nInstalling llama-cpp-python...")
    wheels_dir = os.path.join(addon_dir, "wheels")
    
    # First attempt: Check bundled wheels folder and (on macOS) the official Metal wheel index
    llama_cmd = [
        pip_exe, "install", 
        "llama-cpp-python", 
        "--find-links", wheels_dir,
        "--no-warn-script-location"
    ]
    if os_name == "Darwin":
        llama_cmd.extend(["--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/metal"])
    
    proc_llama = subprocess.Popen(llama_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='replace')
    for line in proc_llama.stdout:
        print(line, end='', flush=True)
    proc_llama.wait()

    if proc_llama.returncode != 0:
        if os_name == "Darwin":
            print("\nPre-built wheel installation failed. Checking build tools for compiling from source...")
            tool_issues = check_mac_build_tools()
            if tool_issues:
                print("\n" + "=" * 65)
                print("ERROR: Cannot compile llama-cpp-python on macOS.")
                print("Missing required build tools:")
                for issue in tool_issues:
                    print(f"  * {issue}")
                print("=" * 65 + "\n")
                sys.exit(1)
            else:
                print("\nBuild tools found. Compiling llama-cpp-python from source with Metal...")
                compile_env = os.environ.copy()
                compile_env["CMAKE_ARGS"] = "-DGGML_METAL=on"
                proc_compile = subprocess.Popen(
                    [pip_exe, "install", "llama-cpp-python", "--no-binary", "llama-cpp-python", "--no-warn-script-location"],
                    env=compile_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='replace'
                )
                for line in proc_compile.stdout:
                    print(line, end='', flush=True)
                proc_compile.wait()
                if proc_compile.returncode != 0:
                    print("Error compiling llama-cpp-python from source.")
                    sys.exit(1)
        else:
            print("Error during llama-cpp-python installation.")
            sys.exit(1)

    print("\nInstalling remaining dependencies...")
    dependencies = [
        "numpy<2.0.0",
        "scipy",
        "transformers>=4.38.0",
        "safetensors",
        "huggingface_hub",
        "torchdiffeq",
        "einops"
    ]
    
    install_cmd = [pip_exe, "install"] + dependencies + ["--no-warn-script-location"]
    proc2 = subprocess.Popen(install_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8', errors='replace')
    for line in proc2.stdout:
        print(line, end='', flush=True)
    proc2.wait()

    if proc2.returncode == 0:
        print("Standalone ML Environment built successfully!")
    else:
        print("Error during remaining pip install.")
        sys.exit(1)

    return python_exe

if __name__ == "__main__":
    install_standalone_python()
