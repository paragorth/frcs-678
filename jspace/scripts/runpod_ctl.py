"""RunPod lifecycle driver for the ignition run (REST API, no web UI).

Subcommands:
  preflight   check the key, list 4090 availability + price per datacenter
  up          create the network volume (if needed) and the pod, wait for SSH
  push        rsync the repo to the pod
  run         run the pipeline on the pod (streams output)
  pull        rsync $JLENS_OUT back down
  down        terminate the pod and report billed wall time / dollars
  status      show pod state and elapsed cost so far

Key lookup order: $RUNPOD_API_KEY, then ~/.config/runpod/api_key.
State (pod id, start time, price) is kept in out/runpod_state.json so `down`
works even from a fresh shell -- a pod left billing is the expensive failure
mode here.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
import urllib.error, urllib.request

API = "https://rest.runpod.io/v1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
# One state file per concurrent job, so pods can run in parallel and each be
# torn down independently. JLENS_JOB names the job (default "default").
JOB = os.environ.get("JLENS_JOB", "default")
STATE = os.path.join(OUT, f"runpod_state_{JOB}.json")
KEYFILE = os.path.expanduser("~/.config/runpod/api_key")
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_runpod")
GPU_PREF = ["NVIDIA GeForce RTX 4090", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
# Verified against the console's own default template on 2026-08-19.
IMAGE = os.environ.get(
    "RUNPOD_IMAGE",
    "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")
# The REST API exposes no /gputypes endpoint, so datacenters are tried in
# order until one accepts a volume and has capacity for the requested GPU.
DC_CANDIDATES = ["EU-RO-1", "US-KS-2", "EU-CZ-1", "CA-MTL-1", "US-TX-3",
                 "US-GA-1", "EU-NL-1", "US-IL-1", "EU-SE-1", "US-CA-2"]


def key() -> str:
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not k and os.path.exists(KEYFILE):
        k = open(KEYFILE).read().strip()
    if not k:
        sys.exit(f"No RunPod API key. Put it in {KEYFILE} (chmod 600) "
                 f"or export RUNPOD_API_KEY.")
    return k


def api(method: str, path: str, body=None, quiet=False):
    req = urllib.request.Request(
        f"{API}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key()}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:600]
        if quiet:
            return None
        print(f"HTTP {e.code} on {method} {path}: {detail}", file=sys.stderr)
        raise SystemExit(1)


def state(**update):
    s = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if update:
        s.update(update)
        os.makedirs(OUT, exist_ok=True)
        json.dump(s, open(STATE, "w"), indent=1)
    return s


def ensure_ssh_key() -> str:
    if not os.path.exists(SSH_KEY):
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", SSH_KEY,
                        "-C", "jlens-runpod"], check=True)
        print(f"generated {SSH_KEY}")
    return open(SSH_KEY + ".pub").read().strip()


def as_list(resp):
    """The REST API returns either a bare list or {data:[...]}"""
    if isinstance(resp, dict):
        return resp.get("data") or resp.get("pods") or resp.get("networkVolumes") or []
    return resp or []


# --------------------------------------------------------------------------
def cmd_preflight(a):
    pods = as_list(api("GET", "/pods", quiet=True) or [])
    vols = as_list(api("GET", "/networkvolumes", quiet=True) or [])
    print("key OK.")
    print(f"  pods: {[(p.get('id'), p.get('name'), p.get('desiredStatus')) for p in pods]}")
    print(f"  network volumes: {[(v.get('id'), v.get('name'), v.get('size'), v.get('dataCenterId')) for v in vols]}")
    print(f"  gpu preference: {GPU_PREF[0]}")
    print(f"  datacenter candidates: {DC_CANDIDATES}")
    print(f"  image: {IMAGE}")


def cmd_up(a):
    pub = ensure_ssh_key()
    gpus = [a.gpu] if a.gpu else GPU_PREF

    vol, vol_dc = None, None
    if not a.no_volume:
        vols = as_list(api("GET", "/networkvolumes", quiet=True) or [])
        vol = next((v for v in vols if v.get("name") == a.volume_name), None)
        if vol is None:
            for dc in DC_CANDIDATES:
                vol = api("POST", "/networkvolumes",
                          {"name": a.volume_name, "size": a.volume_gb,
                           "dataCenterId": dc}, quiet=True)
                if vol and vol.get("id"):
                    break
        vol_dc = vol.get("dataCenterId") if vol else None

    def body_for(gpu, cloud, dcs):
        b = {"name": a.name, "imageName": a.image, "gpuTypeIds": [gpu],
             "gpuCount": a.gpu_count, "cloudType": cloud,
             "containerDiskInGb": a.disk_gb, "ports": ["22/tcp"],
             "supportPublicIp": True,
             "env": {"PUBLIC_KEY": pub, "JLENS_OUT": "/workspace/out"}}
        if vol:
            b["networkVolumeId"] = vol["id"]; b["volumeMountPath"] = "/workspace"
        else:
            b["volumeInGb"] = a.volume_gb; b["volumeMountPath"] = "/workspace"
        if dcs:
            b["dataCenterIds"] = dcs
        return b

    # Capacity is the binding constraint, not price. Try each GPU across each
    # datacenter and both clouds rather than failing on the first miss.
    pod = None
    dc_list = [vol_dc] if vol_dc else DC_CANDIDATES
    for gpu in gpus:
        clouds = [a.cloud] if a.cloud else ["COMMUNITY", "SECURE"]
        for cloud in clouds:
            for dc in dc_list:
                pod = api("POST", "/pods", body_for(gpu, cloud, [dc]), quiet=True)
                if pod and pod.get("id"):
                    print(f"got {gpu} in {dc} ({cloud})")
                    break
            if pod and pod.get("id"):
                break
        if pod and pod.get("id"):
            break
    if not pod or not pod.get("id"):
        sys.exit(f"no capacity for any of {gpus} in any datacenter")

    pid = pod.get("id"); price = pod.get("costPerHr") or 0
    state(pod_id=pid, gpu=pod.get("machine", {}).get("gpuTypeId") or gpus[0],
          price_per_hr=price, started=time.time(),
          volume_id=(vol or {}).get("id"), image=a.image)
    print(f"pod {pid} creating (${price}/hr). waiting for SSH...")
    for _ in range(a.wait):
        p = api("GET", f"/pods/{pid}", quiet=True) or {}
        ip, port = ssh_target(p)
        if ip and port:
            state(ssh_ip=ip, ssh_port=port)
            print(f"pod up  ssh root@{ip} -p {port}")
            if wait_ssh(ip, port, a.wait):
                return
            sys.exit("pod is running but SSH never came up")
        time.sleep(5)
    sys.exit("timed out waiting for pod; run `status`, and `down` if it is billing")


def ssh_target(p):
    for pm in (p.get("portMappings") or []):
        if isinstance(pm, dict) and str(pm.get("privatePort")) == "22":
            return p.get("publicIp"), pm.get("publicPort")
    pm = p.get("portMappings")
    if isinstance(pm, dict) and "22" in {str(k) for k in pm}:
        return p.get("publicIp"), pm.get("22") or pm.get(22)
    return p.get("publicIp"), None


def ssh_args(s):
    return ["-i", SSH_KEY, "-p", str(s["ssh_port"]), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]


def wait_ssh(ip, port, tries=120):
    s = {"ssh_ip": ip, "ssh_port": port}
    for i in range(tries):
        r = subprocess.run(["ssh", *ssh_args(s), "-o", "ConnectTimeout=8",
                            f"root@{ip}", "true"], capture_output=True)
        if r.returncode == 0:
            print(f"SSH up after {i*5}s")
            return True
        time.sleep(5)
    return False


def cmd_push(a):
    s = state()
    subprocess.run(["ssh", *ssh_args(s), f"root@{s['ssh_ip']}",
                    "mkdir -p /root/jlens /workspace/out"], check=True)
    rsh = "ssh " + " ".join(ssh_args(s))
    for src in ("scripts", "jacobian-lens"):
        subprocess.run(["rsync", "-az", "--delete", "-e", rsh,
                        "--exclude", "__pycache__", "--exclude", ".git",
                        "--exclude", ".venv",
                        f"{ROOT}/{src}/", f"root@{s['ssh_ip']}:/root/jlens/{src}/"],
                       check=True)
    # seed the network volume with the frozen inputs (never overwrite results)
    for f in ("sweep_prompts.json", "fit_prompts.json", "tracked_tokens.json"):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            subprocess.run(["rsync", "-az", "-e", rsh, p,
                            f"root@{s['ssh_ip']}:/workspace/out/"], check=True)
    print("pushed code + frozen inputs")


REMOTE_SETUP = """set -e
cd /root/jlens
pip -q install --upgrade pip
pip -q install './jacobian-lens' 'transformers>=5.5' datasets numpy
python3 -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"
"""


def cmd_run(a):
    s = state()
    script = REMOTE_SETUP if a.setup_only else (
        REMOTE_SETUP + f"\nMODEL={a.model} NP={a.n_prompts} "
                       f"JLENS_OUT=/workspace/out bash scripts/run_all.sh\n")
    r = subprocess.run(["ssh", *ssh_args(s), f"root@{s['ssh_ip']}",
                        "bash -s"], input=script.encode())
    print(f"remote exit code {r.returncode}")
    return r.returncode


def cmd_pull(a):
    s = state()
    rsh = "ssh " + " ".join(ssh_args(s))
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["rsync", "-azP", "-e", rsh,
                    f"root@{s['ssh_ip']}:/workspace/out/", f"{OUT}/"], check=True)
    print(f"pulled results -> {OUT}")


def elapsed_cost(s):
    hrs = (time.time() - s.get("started", time.time())) / 3600
    return hrs, hrs * float(s.get("price_per_hr") or 0)


def cmd_status(a):
    s = state()
    if not s.get("pod_id"):
        print("no pod recorded"); return
    p = api("GET", f"/pods/{s['pod_id']}", quiet=True) or {}
    hrs, cost = elapsed_cost(s)
    print(json.dumps({"pod_id": s["pod_id"], "status": p.get("desiredStatus"),
                      "gpu": s.get("gpu"), "price_per_hr": s.get("price_per_hr"),
                      "elapsed_hours": round(hrs, 3),
                      "cost_so_far_usd": round(cost, 3)}, indent=1))


def cmd_down(a):
    s = state()
    pid = s.get("pod_id")
    if not pid:
        print("no pod recorded"); return
    hrs, cost = elapsed_cost(s)
    api("DELETE", f"/pods/{pid}", quiet=True)
    time.sleep(4)
    left = [p.get("id") for p in as_list(api("GET", "/pods", quiet=True) or [])]
    ok = pid not in left
    state(destroyed=time.time(), billed_hours=round(hrs, 4),
          cost_usd=round(cost, 3), destroy_confirmed=ok)
    print(f"pod {pid} terminated={ok}; billed ~{hrs:.3f} h ~= ${cost:.2f}")
    if not ok:
        print("!!! pod still listed -- CHECK THE CONSOLE, IT MAY STILL BE BILLING")
    if a.keep_volume:
        print(f"network volume {s.get('volume_id')} kept (still charged for storage)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight").set_defaults(fn=cmd_preflight)
    u = sub.add_parser("up"); u.set_defaults(fn=cmd_up)
    u.add_argument("--gpu", default=None)
    u.add_argument("--no-volume", action="store_true")
    u.add_argument("--gpu-count", type=int, default=1)
    u.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    u.add_argument("--name", default="jlens-ignition")
    u.add_argument("--volume-name", default="jlens-out")
    u.add_argument("--volume-gb", type=int, default=50)
    u.add_argument("--disk-gb", type=int, default=60)
    u.add_argument("--image", default=IMAGE)
    u.add_argument("--wait", type=int, default=180)
    sub.add_parser("push").set_defaults(fn=cmd_push)
    r = sub.add_parser("run"); r.set_defaults(fn=cmd_run)
    r.add_argument("--model", default="1.5b")
    r.add_argument("--n-prompts", type=int, default=500)
    r.add_argument("--setup-only", action="store_true")
    sub.add_parser("pull").set_defaults(fn=cmd_pull)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    d = sub.add_parser("down"); d.set_defaults(fn=cmd_down)
    d.add_argument("--keep-volume", action="store_true", default=True)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
