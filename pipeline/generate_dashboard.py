# pipeline/generate_dashboard.py
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import KMeans
import yaml

DOCS_DIR = "docs"
CONFIG_PATH = "config/settings.yaml"

def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)

def ensure_docs_dir():
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(os.path.join(DOCS_DIR, ".nojekyll"), "w") as f:
        f.write("")

def load_config():
    defaults = {
        "dashboard_title": "AI Pipeline Demo",
        "clusters": 4,
        "mesh": {"type": "torus", "R": 3.0, "r": 1.0, "nu": 100, "nv": 80, "file_path": ""},
        "metrics_days": 30,
        "seed": 42,
    }
    if not os.path.exists(CONFIG_PATH):
        log(f"Config not found at {CONFIG_PATH}, using defaults.")
        return defaults
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # merge shallowly
    out = defaults | {k: cfg.get(k, defaults[k]) for k in defaults}
    out["mesh"] = defaults["mesh"] | (cfg.get("mesh") or {})
    return out

def torus(R=3.0, r=1.0, nu=100, nv=80):
    u = np.linspace(0, 2*np.pi, nu, endpoint=False)
    v = np.linspace(0, 2*np.pi, nv, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = (R + r*np.cos(vv)) * np.cos(uu)
    y = (R + r*np.cos(vv)) * np.sin(uu)
    z = r * np.sin(vv)
    def idx(i, j): return (i % nu) * nv + (j % nv)
    I, J, K = [], [], []
    for i in range(nu):
        for j in range(nv):
            a = idx(i, j); b = idx(i+1, j); c = idx(i+1, j+1); d = idx(i, j+1)
            I.extend([a, a]); J.extend([b, c]); K.extend([c, d])
    return x.ravel(), y.ravel(), z.ravel(), np.array(I), np.array(J), np.array(K)

def sphere(r=1.5, nu=100, nv=80):
    u = np.linspace(0, np.pi, nu, endpoint=False)
    v = np.linspace(0, 2*np.pi, nv, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = r*np.sin(uu)*np.cos(vv)
    y = r*np.sin(uu)*np.sin(vv)
    z = r*np.cos(uu)
    def idx(i, j): return (i % nu) * nv + (j % nv)
    I, J, K = [], [], []
    for i in range(nu-1):
        for j in range(nv):
            a = idx(i, j); b = idx(i+1, j); c = idx(i+1, j+1); d = idx(i, j+1)
            I.extend([a, a]); J.extend([b, c]); K.extend([c, d])
    return x.ravel(), y.ravel(), z.ravel(), np.array(I), np.array(J), np.array(K)

def load_mesh_from_config(mesh_cfg):
    kind = mesh_cfg.get("type", "torus")
    if kind == "torus":
        return torus(mesh_cfg["R"], mesh_cfg["r"], mesh_cfg["nu"], mesh_cfg["nv"])
    if kind == "sphere":
        return sphere(r=mesh_cfg.get("R", 1.5), nu=mesh_cfg.get("nu", 100), nv=mesh_cfg.get("nv", 80))
    if kind == "file":
        path = mesh_cfg.get("file_path", "")
        if not path:
            raise ValueError("mesh.type is 'file' but mesh.file_path is empty.")
        try:
            import trimesh  # optional; not in requirements by default
        except ImportError as e:
            raise RuntimeError("trimesh not installed. Add 'trimesh>=4.4' to requirements.txt") from e
        m = trimesh.load(path, force='mesh')
        if m.faces.ndim != 2 or m.faces.shape[1] != 3:
            m = m.triangles
        x, y, z = m.vertices[:,0], m.vertices[:,1], m.vertices[:,2]
        I, J, K = m.faces[:,0], m.faces[:,1], m.faces[:,2]
        return x, y, z, I, J, K
    raise ValueError(f"Unknown mesh type: {kind}")

def cluster_vertices_kmeans(x, y, z, n_clusters=4, seed=42):
    pts = np.column_stack([x, y, z])
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    labels = km.fit_predict(pts)
    return labels

def make_metrics(n_days=30):
    dates = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=n_days, freq="D")
    values = pd.Series(np.cumsum(np.random.randn(n_days)) + 50, index=dates).clip(lower=0)
    return pd.DataFrame({"date": dates, "value": values})

def build_dashboard_html(cfg, out_path="docs/index.html"):
    log(f"Config: {json.dumps(cfg, indent=2)}")
    x, y, z, I, J, K = load_mesh_from_config(cfg["mesh"])

    labels = cluster_vertices_kmeans(x, y, z, n_clusters=cfg["clusters"], seed=cfg["seed"])
    intensity = labels.astype(float)

    metrics = make_metrics(cfg["metrics_days"])
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    cluster_df = pd.DataFrame({"cluster": cluster_counts.index.astype(str),
                               "count": cluster_counts.values})

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        column_widths=[0.6, 0.4],
        row_heights=[0.55, 0.45],
        subplot_titles=("3D Model (clusters)", "Metric (last 30d)", "Cluster sizes")
    )

    fig.add_trace(
        go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, intensity=intensity, showscale=False, name="mesh"),
        row=1, col=1
    )
    fig.update_scenes(aspectmode="data", camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=metrics["date"], y=metrics["value"], mode="lines+markers", name="metric"),
        row=1, col=2
    )

    fig.add_trace(go.Bar(x=cluster_df["cluster"], y=cluster_df["count"], name="clusters"), row=2, col=2)

    fig.update_layout(
        title=f"{cfg['dashboard_title']} — last run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} • K-Means on vertices",
        height=850, margin=dict(l=10, r=10, t=60, b=10),
    )

    pio.write_html(fig, file=out_path, include_plotlyjs="cdn", full_html=True)
    log(f"Wrote {out_path}")

def main():
    try:
        ensure_docs_dir()
        cfg = load_config()
        build_dashboard_html(cfg, out_path=os.path.join(DOCS_DIR, "index.html"))
        log("Done.")
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
