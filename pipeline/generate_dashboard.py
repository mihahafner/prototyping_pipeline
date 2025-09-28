# pipeline/generate_dashboard.py
import os
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timezone

DOCS_DIR = "docs"
os.makedirs(DOCS_DIR, exist_ok=True)

# Make sure GitHub Pages doesn't try to process with Jekyll
with open(os.path.join(DOCS_DIR, ".nojekyll"), "w") as f:
    f.write("")

def torus(R=3.0, r=1.0, nu=100, nv=80):
    """Generate a torus mesh and return flattened x,y,z and triangle indices i,j,k."""
    u = np.linspace(0, 2*np.pi, nu, endpoint=False)
    v = np.linspace(0, 2*np.pi, nv, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    x = (R + r*np.cos(vv)) * np.cos(uu)
    y = (R + r*np.cos(vv)) * np.sin(uu)
    z = r * np.sin(vv)

    # Triangulate the wrap-around grid
    def idx(i, j): return (i % nu) * nv + (j % nv)
    I, J, K = [], [], []
    for i in range(nu):
        for j in range(nv):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i + 1, j + 1)
            d = idx(i, j + 1)
            # two triangles: (a,b,c) and (a,c,d)
            I.extend([a, a])
            J.extend([b, c])
            K.extend([c, d])

    x, y, z = x.ravel(), y.ravel(), z.ravel()
    I, J, K = np.array(I), np.array(J), np.array(K)
    intensity = (z - z.min()) / (z.ptp() + 1e-9)  # color by height
    return x, y, z, I, J, K, intensity

def make_metrics():
    """Fake metric series that updates each pipeline run."""
    dates = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=30, freq="D")
    values = pd.Series(np.cumsum(np.random.randn(30)) + 50, index=dates).clip(lower=0)
    return pd.DataFrame({"date": dates, "value": values})

def build_dashboard_html(out_path="docs/index.html"):
    x, y, z, I, J, K, c = torus()
    metrics = make_metrics()

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.6, 0.4],
        subplot_titles=("3D Model (torus)", "Pipeline Metric")
    )

    fig.add_trace(
        go.Mesh3d(
            x=x, y=y, z=z,
            i=I, j=J, k=K,
            intensity=c,
            showscale=True,
            opacity=1.0
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=metrics["date"], y=metrics["value"],
            mode="lines+markers", name="metric"
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=f"AI Pipeline Demo — last run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        height=700,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    pio.write_html(fig, file=out_path, include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    build_dashboard_html()
