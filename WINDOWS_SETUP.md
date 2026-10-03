# Running this project on Windows

There are two ways to run this. **WSL2 is recommended** - the project's
original scripts work completely unmodified inside it, and it's what most
real-world Kubernetes-on-Windows development actually uses. A native
PowerShell path is also provided below if you'd rather not install WSL2.

Note: these Windows instructions are written against standard, documented
tool behavior, but weren't executed on an actual Windows machine while
being written (this project was built in a Linux-only environment) - if
you hit something that doesn't match your setup exactly, it's most likely
a version/PATH difference worth double-checking against the tool's own
`--help` output.

---

## Path A: WSL2 (recommended)

### 1. Install WSL2 + Ubuntu

In an **Administrator PowerShell**:
```powershell
wsl --install
```
Reboot if prompted. This installs WSL2 with Ubuntu by default. Open
"Ubuntu" from the Start menu once it's done and finish the Linux user setup.

### 2. Install Docker Desktop, with WSL2 integration

Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
During/after install:
- Settings -> General -> ensure "Use the WSL 2 based engine" is checked
- Settings -> Resources -> WSL Integration -> enable integration with your
  Ubuntu distro

### 3. Inside the Ubuntu (WSL2) terminal, install the rest

```bash
# minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/kubectl

# python + pip (Ubuntu usually has python3 already; this ensures pip/venv)
sudo apt update && sudo apt install -y python3-pip python3-venv
```

### 4. Unzip the project INSIDE the WSL2 filesystem

Important for performance: don't run this from `/mnt/c/...` (your Windows
drive mounted into WSL2) - copy/unzip it into your Linux home directory
instead:
```bash
cp /mnt/c/Users/<you>/Downloads/k8s-autoscaler-project.zip ~/
cd ~ && unzip k8s-autoscaler-project.zip && cd k8s-autoscaler-project
```

### 5. From here, follow the original README.md exactly

Everything in the main `README.md` (the `bash scripts/01_...` through `08_...`
sequence) now works completely unmodified, because you're in a real Linux
environment. Browser URLs like `http://localhost:9090` still work normally
from Windows - WSL2 forwards localhost automatically.

```bash
minikube start --cpus=4 --memory=6144
eval $(minikube docker-env)
bash scripts/01_build_images.sh
bash scripts/03_deploy_all.sh
bash scripts/04_verify_deployment.sh
# new terminal (wsl -d Ubuntu, or just open another Ubuntu terminal tab):
bash scripts/05_port_forward.sh
# back in the first terminal:
bash scripts/06_train_all_arms.sh
bash scripts/07_run_traffic.sh burst 180 20
bash scripts/08_run_autoscaler.sh --shadow --iterations 20
```

---

## Path B: Native Windows (PowerShell, no WSL2)

### 1. Install prerequisites

| Tool | Install |
|---|---|
| Docker Desktop | [docker.com](https://www.docker.com/products/docker-desktop/) - Windows containers engine is fine, don't need WSL2 backend for this path, though it still works if you have it |
| minikube | `winget install minikube` (or the [installer](https://minikube.sigs.k8s.io/docs/start/)) |
| kubectl | `winget install Kubernetes.kubectl` (or comes bundled with Docker Desktop) |
| Python 3.11+ | [python.org](https://www.python.org/downloads/windows/) - **check "Add python.exe to PATH"** during install |

Verify each:
```powershell
docker --version
minikube version
kubectl version --client
python --version
```

### 2. Unzip the project and set up a virtual environment

```powershell
Expand-Archive k8s-autoscaler-project.zip -DestinationPath .
cd k8s-autoscaler-project

python -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy, run once as Administrator:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Install dependencies:
```powershell
pip install -r autoscaler\requirements.txt
pip install -r traffic-generator\requirements.txt
```

### 3. Start minikube

```powershell
minikube start --cpus=4 --memory=6144
```

### 4. Point Docker at minikube's daemon, then build the image

```powershell
minikube docker-env | Invoke-Expression
.\scripts\windows\01_build_images.ps1
```

(If using kind instead of minikube: build normally with `docker build`,
then run `.\scripts\windows\02_load_image_kind.ps1`.)

### 5. Deploy everything

```powershell
.\scripts\windows\03_deploy_all.ps1
.\scripts\windows\04_verify_deployment.ps1
```

**Troubleshooting `ImagePullBackOff`:** you skipped step 4's
`minikube docker-env | Invoke-Expression`, or ran it in a *different*
PowerShell window than the one you built the image in - that command only
affects the current session. Re-run it in the same window you're using for
`docker build`, then re-apply.

### 6. Open a second PowerShell window for port-forwarding

```powershell
cd k8s-autoscaler-project
.\scripts\windows\05_port_forward.ps1
```
Leave this running. (If PowerShell background jobs act up on your machine,
see the note inside that script for a three-separate-windows alternative -
it's more verbose but bulletproof.)

### 7. Back in your first window: train the models

```powershell
.\scripts\windows\06_train_all_arms.ps1
```
Doesn't need the cluster - pure local training against the calibrated
simulator, same as on Linux.

### 8. Generate real traffic

```powershell
.\scripts\windows\07_run_traffic.ps1 -Pattern burst -Duration 180 -BaseRps 20
```

Open `http://localhost:16686` in your browser to see real traces in the
Jaeger UI while this runs.

### 9. Run the autoscaler

```powershell
# shadow mode first, to see decisions without applying them:
.\scripts\windows\08_run_autoscaler.ps1 --shadow --iterations 20

# then live:
.\scripts\windows\08_run_autoscaler.ps1 --iterations 30
```

Watch it happen in real time, in another window:
```powershell
kubectl get pods -n autoscaler-poc -w
```

---

## Common Windows-specific gotchas

- **`python` vs `python3`**: Windows installs usually give you `python`,
  not `python3`. All the `.ps1` scripts above already use `python`. If
  `python` isn't recognized, re-run the Python installer and check
  "Add to PATH", or use the `py` launcher (`py -m ...`) instead.
- **Line endings**: if you edit any `.sh` file on Windows and later run it
  inside WSL2, save it with LF line endings (not CRLF), or `dos2unix` it -
  otherwise you'll see `$'\r': command not found` errors. The `.ps1` files
  don't have this issue.
- **Docker Desktop must be running** before any `minikube start` or
  `docker build` command - check the whale icon in your system tray.
- **Hyper-V / WSL2 / virtualization conflicts**: if `minikube start` fails
  mentioning virtualization, ensure virtualization is enabled in your
  BIOS/UEFI, and that you're not simultaneously running VirtualBox and
  Hyper-V-based tools in a conflicting configuration.
- **Firewall prompts**: Windows Defender Firewall may prompt to allow
  `kubectl.exe` or `docker.exe` network access the first time you run
  `port-forward` - allow it, or port-forwarding will silently fail to
  connect.
