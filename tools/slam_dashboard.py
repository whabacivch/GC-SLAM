#!/usr/bin/env python3
"""
Golden Child SLAM v2 Debugging Dashboard.

Interactive Plotly dashboard for visualizing per-scan diagnostics.

Four panels:
- Panel A: Timeline scrubber with diagnostic scalar plots
- Panel B: Evidence inspector heatmap (22x22 L matrix)
- Panel C: 3D trajectory view with local evidence field glyphs
- Panel D: Excitation & Fusion diagnostics

Usage:
    # Auto-open in browser (Wayland and X11 compatible)
    python tools/slam_dashboard.py /tmp/gc_slam_diagnostics.npz
    
    # Save to file and open manually
    python tools/slam_dashboard.py /tmp/gc_slam_diagnostics.npz --output dashboard.html
    
    # Start at specific scan
    python tools/slam_dashboard.py /tmp/gc_slam_diagnostics.npz --scan 50
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

import numpy as np

# Add the package to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
FL_WS_SRC = PROJECT_ROOT / "fl_ws" / "src" / "fl_slam_poc"
sys.path.insert(0, str(FL_WS_SRC))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: plotly not installed. Install with: pip install plotly")
    sys.exit(1)

try:
    from fl_slam_poc.backend.diagnostics import DiagnosticsLog
except ImportError:
    print("Warning: Could not import DiagnosticsLog, using standalone loader")
    DiagnosticsLog = None


def load_diagnostics_npz(path: str) -> dict:
    """Load diagnostics from NPZ file into a dictionary."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def numpy_to_json(obj):
    """Convert numpy arrays to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_json(v) for v in obj]
    return obj


def open_browser_wayland_compatible(file_path: str) -> bool:
    """
    Open a file in the default browser, compatible with Wayland and X11.
    
    Tries multiple methods:
    1. xdg-open (works on both Wayland and X11)
    2. $BROWSER environment variable
    3. webbrowser module (fallback)
    
    Returns True if successful, False otherwise.
    """
    file_path = os.path.abspath(file_path)
    file_url = f"file://{urllib.parse.quote(file_path, safe='/')}"
    
    # Method 1: Try xdg-open (works on Wayland and X11)
    try:
        subprocess.Popen(
            ["xdg-open", file_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        return True
    except (FileNotFoundError, OSError):
        pass
    
    # Method 2: Try $BROWSER environment variable
    browser = os.environ.get("BROWSER")
    if browser:
        try:
            # Handle browsers that need the URL as an argument
            if "%s" in browser or "%u" in browser:
                cmd = browser.replace("%s", file_url).replace("%u", file_url).split()
            else:
                cmd = [browser, file_url]
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            return True
        except (FileNotFoundError, OSError):
            pass
    
    # Method 3: Fallback to webbrowser module
    try:
        import webbrowser
        webbrowser.open(file_url)
        return True
    except Exception:
        pass
    
    return False


def create_full_dashboard(data: dict, selected_scan: int = 0, output_path: str = None) -> str:
    """
    Create and display the fully interactive dashboard.

    All panels update dynamically when the slider is moved.
    
    Args:
        data: Diagnostics data dictionary
        selected_scan: Initial scan index to display
        output_path: Optional path to save HTML file. If None, uses temp file.
    
    Returns:
        Path to the created HTML file
    """
    n_scans = int(data.get("n_scans", 0))
    if n_scans == 0:
        print("No scan data found in file")
        return None

    # Prepare all data for JavaScript
    scan_idx = list(range(n_scans))

    # Timeline data - handle missing keys gracefully
    def safe_list(key, default_val=0.0):
        if key in data:
            arr = np.array(data[key])
            return arr.tolist() if hasattr(arr, 'tolist') else list(arr)
        return [default_val] * n_scans

    timeline_data = {
        "n_scans": n_scans,
        "scan_idx": scan_idx,
        "logdet_L_total": safe_list("logdet_L_total"),
        "trace_L_total": safe_list("trace_L_total"),
        "L_dt": safe_list("L_dt"),
        "trace_L_ex": safe_list("trace_L_ex"),
        "psd_delta_fro": safe_list("psd_delta_fro"),
        "psd_min_eig_after": safe_list("psd_min_eig_after"),
        "trace_Q_mode": safe_list("trace_Q_mode"),
        "trace_Sigma_lidar_mode": safe_list("trace_Sigma_lidar_mode"),
        "s_dt": safe_list("s_dt"),
        "s_ex": safe_list("s_ex"),
        "fusion_alpha": safe_list("fusion_alpha"),
        "dt_secs": safe_list("dt_secs", 0.1),
        "wahba_cost": safe_list("wahba_cost"),
        "translation_residual_norm": safe_list("translation_residual_norm"),
    }

    # Bin statistics
    N_bins = data.get("N_bins", np.zeros((n_scans, 48)))
    kappa_bins = data.get("kappa_bins", np.zeros((n_scans, 48)))
    if hasattr(N_bins, 'tolist'):
        N_bins = np.array(N_bins)
    if hasattr(kappa_bins, 'tolist'):
        kappa_bins = np.array(kappa_bins)

    timeline_data["sum_N"] = np.sum(N_bins, axis=1).tolist()
    timeline_data["mean_kappa"] = np.mean(kappa_bins, axis=1).tolist()

    # Trajectory data
    p_W = data.get("p_W", np.zeros((n_scans, 3)))
    if hasattr(p_W, 'tolist'):
        p_W = np.array(p_W)
    trajectory_data = {
        "x": p_W[:, 0].tolist(),
        "y": p_W[:, 1].tolist(),
        "z": p_W[:, 2].tolist(),
        "logdet": timeline_data["logdet_L_total"],
    }

    # L matrices for heatmap (all scans)
    L_total = data.get("L_total", np.zeros((n_scans, 22, 22)))
    if hasattr(L_total, 'tolist'):
        L_total = np.array(L_total)

    # S_bins and R_WL for direction glyphs
    S_bins = data.get("S_bins", np.zeros((n_scans, 48, 3)))
    R_WL = data.get("R_WL", np.tile(np.eye(3), (n_scans, 1, 1)))
    if hasattr(S_bins, 'tolist'):
        S_bins = np.array(S_bins)
    if hasattr(R_WL, 'tolist'):
        R_WL = np.array(R_WL)

    # Create HTML with embedded data and interactive JavaScript
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GC SLAM v2 Diagnostics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{
            text-align: center;
            color: #00d4ff;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 20px;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            max-width: 1900px;
            margin: 0 auto;
        }}
        .panel {{
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .controls {{
            text-align: center;
            margin: 15px 0;
            padding: 15px;
            background: #16213e;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }}
        .controls input[type="range"] {{
            width: 400px;
            height: 8px;
            -webkit-appearance: none;
            background: #0f3460;
            border-radius: 4px;
            outline: none;
        }}
        .controls input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #00d4ff;
            border-radius: 50%;
            cursor: pointer;
        }}
        .controls label {{
            font-weight: bold;
            font-size: 16px;
        }}
        .scan-info {{
            background: #0f3460;
            padding: 8px 15px;
            border-radius: 5px;
            font-family: monospace;
        }}
        .scan-info span {{
            color: #00d4ff;
            font-weight: bold;
        }}
        #scan-display {{
            font-size: 24px;
            color: #00d4ff;
            min-width: 60px;
            display: inline-block;
            text-align: center;
        }}
        .panel-title {{
            font-size: 14px;
            color: #888;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stats-row {{
            display: flex;
            gap: 15px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        .stat-box {{
            background: #0f3460;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 12px;
        }}
        .stat-box .label {{
            color: #888;
        }}
        .stat-box .value {{
            color: #00d4ff;
            font-weight: bold;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <h1>Golden Child SLAM v2 Diagnostics Dashboard</h1>
    <p class="subtitle">Interactive per-scan pipeline diagnostics | {n_scans} scans loaded</p>

    <div class="controls">
        <label>Scan:</label>
        <span id="scan-display">{selected_scan}</span>
        <input type="range" id="scan-slider" min="0" max="{n_scans - 1}" value="{selected_scan}">
        <div class="scan-info">
            dt: <span id="info-dt">--</span>s |
            α: <span id="info-alpha">--</span> |
            log|L|: <span id="info-logdet">--</span>
        </div>
    </div>

    <div class="dashboard-grid">
        <div class="panel full-width">
            <div class="panel-title">Panel A: Timeline Diagnostics</div>
            <div id="timeline"></div>
        </div>
        <div class="panel">
            <div class="panel-title">Panel B: Evidence Matrix L (22×22)</div>
            <div id="heatmap"></div>
        </div>
        <div class="panel">
            <div class="panel-title">Panel C: 3D Trajectory</div>
            <div id="trajectory"></div>
        </div>
        <div class="panel full-width">
            <div class="panel-title">Panel D: Excitation & Fusion Diagnostics</div>
            <div id="excitation"></div>
        </div>
    </div>

    <script>
    // Embedded data
    const timelineData = {json.dumps(numpy_to_json(timeline_data))};
    const trajectoryData = {json.dumps(numpy_to_json(trajectory_data))};
    const L_matrices = {json.dumps(L_total.tolist())};
    const S_bins = {json.dumps(S_bins.tolist())};
    const R_WL = {json.dumps(R_WL.tolist())};
    const kappa_bins = {json.dumps(kappa_bins.tolist())};

    let currentScan = {selected_scan};
    const nScans = {n_scans};

    // Block labels for heatmap
    const blockBoundaries = [0, 3, 6, 9, 12, 15, 16, 22];
    const blockLabels = ['pos', 'rot', 'vel', 'bg', 'ba', 'dt', 'ex'];

    // Dark theme layout
    const darkLayout = {{
        paper_bgcolor: '#16213e',
        plot_bgcolor: '#0f3460',
        font: {{ color: '#eee' }},
        xaxis: {{ gridcolor: '#1a1a2e', zerolinecolor: '#1a1a2e' }},
        yaxis: {{ gridcolor: '#1a1a2e', zerolinecolor: '#1a1a2e' }},
    }};

    // =====================================================================
    // Panel A: Timeline
    // =====================================================================
    function createTimeline() {{
        const traces = [
            // Row 1: Evidence strength
            {{ x: timelineData.scan_idx, y: timelineData.logdet_L_total, name: 'log|L|', line: {{color: '#00d4ff'}}, xaxis: 'x', yaxis: 'y' }},
            {{ x: timelineData.scan_idx, y: timelineData.trace_L_total, name: 'tr(L)', line: {{color: '#ff6b6b'}}, xaxis: 'x', yaxis: 'y2' }},
            // Row 2: Observability
            {{ x: timelineData.scan_idx, y: timelineData.L_dt, name: 'L[dt,dt]', line: {{color: '#4ecdc4'}}, xaxis: 'x2', yaxis: 'y3' }},
            {{ x: timelineData.scan_idx, y: timelineData.trace_L_ex, name: 'tr(L_ex)', line: {{color: '#f7b731'}}, xaxis: 'x2', yaxis: 'y3' }},
            // Row 3: Numerical health
            {{ x: timelineData.scan_idx, y: timelineData.psd_delta_fro, name: 'PSD Δ', line: {{color: '#a55eea'}}, xaxis: 'x3', yaxis: 'y4' }},
            {{ x: timelineData.scan_idx, y: timelineData.psd_min_eig_after, name: 'min λ', line: {{color: '#26de81'}}, xaxis: 'x3', yaxis: 'y4' }},
            // Row 4: Noise adaptation
            {{ x: timelineData.scan_idx, y: timelineData.trace_Q_mode, name: 'tr(Q)', line: {{color: '#fd9644'}}, xaxis: 'x4', yaxis: 'y5' }},
            {{ x: timelineData.scan_idx, y: timelineData.trace_Sigma_lidar_mode, name: 'tr(Σ_lidar)', line: {{color: '#fc5c65'}}, xaxis: 'x4', yaxis: 'y5' }},
            // Row 5: Bin statistics
            {{ x: timelineData.scan_idx, y: timelineData.sum_N, name: 'Σ N', line: {{color: '#778ca3'}}, xaxis: 'x5', yaxis: 'y6' }},
            {{ x: timelineData.scan_idx, y: timelineData.mean_kappa, name: 'mean(κ)', line: {{color: '#a5b1c2'}}, xaxis: 'x5', yaxis: 'y6' }},
        ];

        const layout = {{
            ...darkLayout,
            height: 500,
            showlegend: true,
            legend: {{ orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center' }},
            grid: {{ rows: 5, columns: 1, pattern: 'independent', roworder: 'top to bottom' }},
            xaxis: {{ ...darkLayout.xaxis, anchor: 'y', domain: [0, 1], showticklabels: false }},
            xaxis2: {{ ...darkLayout.xaxis, anchor: 'y3', domain: [0, 1], showticklabels: false }},
            xaxis3: {{ ...darkLayout.xaxis, anchor: 'y4', domain: [0, 1], showticklabels: false }},
            xaxis4: {{ ...darkLayout.xaxis, anchor: 'y5', domain: [0, 1], showticklabels: false }},
            xaxis5: {{ ...darkLayout.xaxis, anchor: 'y6', domain: [0, 1], title: 'Scan Index' }},
            yaxis: {{ ...darkLayout.yaxis, domain: [0.84, 1], title: 'Evidence' }},
            yaxis2: {{ ...darkLayout.yaxis, overlaying: 'y', side: 'right' }},
            yaxis3: {{ ...darkLayout.yaxis, domain: [0.63, 0.79], title: 'Observability' }},
            yaxis4: {{ ...darkLayout.yaxis, domain: [0.42, 0.58], title: 'PSD Health' }},
            yaxis5: {{ ...darkLayout.yaxis, domain: [0.21, 0.37], title: 'Noise' }},
            yaxis6: {{ ...darkLayout.yaxis, domain: [0, 0.16], title: 'Bins' }},
            margin: {{ t: 60, b: 40, l: 60, r: 40 }},
            // Shapes for vertical line (will be updated)
            shapes: createVerticalLines(currentScan, 5),
        }};

        Plotly.newPlot('timeline', traces, layout, {{responsive: true}});
    }}

    function createVerticalLines(scanIdx, numRows) {{
        const shapes = [];
        const yDomains = [[0.84, 1], [0.63, 0.79], [0.42, 0.58], [0.21, 0.37], [0, 0.16]];
        for (let i = 0; i < numRows; i++) {{
            shapes.push({{
                type: 'line',
                x0: scanIdx, x1: scanIdx,
                y0: 0, y1: 1,
                xref: 'x' + (i === 0 ? '' : (i + 1)),
                yref: 'paper',
                line: {{ color: '#ff6b6b', width: 2, dash: 'dash' }}
            }});
        }}
        return shapes;
    }}

    // =====================================================================
    // Panel B: Heatmap
    // =====================================================================
    function createHeatmap(scanIdx) {{
        const L = L_matrices[scanIdx];

        // Create annotations for block labels
        const annotations = [];
        for (let i = 0; i < blockLabels.length; i++) {{
            const mid = (blockBoundaries[i] + blockBoundaries[i+1]) / 2 - 0.5;
            annotations.push({{
                x: mid, y: 22.5, text: blockLabels[i], showarrow: false, font: {{size: 10, color: '#888'}}
            }});
            annotations.push({{
                x: -1.5, y: mid, text: blockLabels[i], showarrow: false, font: {{size: 10, color: '#888'}}
            }});
        }}

        // Create shapes for block boundaries
        const shapes = [];
        for (let i = 1; i < blockBoundaries.length - 1; i++) {{
            const b = blockBoundaries[i] - 0.5;
            shapes.push({{ type: 'line', x0: b, x1: b, y0: -0.5, y1: 21.5, line: {{color: '#333', width: 1, dash: 'dot'}} }});
            shapes.push({{ type: 'line', x0: -0.5, x1: 21.5, y0: b, y1: b, line: {{color: '#333', width: 1, dash: 'dot'}} }});
        }}

        const trace = {{
            z: L,
            type: 'heatmap',
            colorscale: 'RdBu',
            zmid: 0,
            colorbar: {{ title: 'Value', tickfont: {{color: '#eee'}} }}
        }};

        const layout = {{
            ...darkLayout,
            height: 450,
            title: {{ text: `L_total (Scan ${{scanIdx}})`, font: {{color: '#00d4ff'}} }},
            xaxis: {{ ...darkLayout.xaxis, title: 'Column', scaleanchor: 'y', constrain: 'domain' }},
            yaxis: {{ ...darkLayout.yaxis, title: 'Row', autorange: 'reversed', constrain: 'domain' }},
            annotations: annotations,
            shapes: shapes,
            margin: {{ t: 50, b: 50, l: 50, r: 30 }},
        }};

        Plotly.react('heatmap', [trace], layout, {{responsive: true}});
    }}

    // =====================================================================
    // Panel C: 3D Trajectory
    // =====================================================================
    function createTrajectory(scanIdx) {{
        const traces = [
            // Main trajectory
            {{
                x: trajectoryData.x, y: trajectoryData.y, z: trajectoryData.z,
                mode: 'lines+markers',
                type: 'scatter3d',
                marker: {{
                    size: 3,
                    color: trajectoryData.logdet,
                    colorscale: 'Viridis',
                    colorbar: {{ title: 'log|L|', x: 1.02, tickfont: {{color: '#eee'}} }},
                    showscale: true
                }},
                line: {{ color: '#555', width: 2 }},
                name: 'Trajectory',
                hovertemplate: 'Scan %{{text}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>',
                text: timelineData.scan_idx.map(i => i.toString())
            }},
            // Selected point marker
            {{
                x: [trajectoryData.x[scanIdx]],
                y: [trajectoryData.y[scanIdx]],
                z: [trajectoryData.z[scanIdx]],
                mode: 'markers',
                type: 'scatter3d',
                marker: {{ size: 10, color: '#ff6b6b', symbol: 'diamond' }},
                name: `Scan ${{scanIdx}}`
            }}
        ];

        // Add direction glyphs for top-6 bins by kappa
        const kappaArr = kappa_bins[scanIdx];
        const indices = [...Array(48).keys()].sort((a, b) => kappaArr[b] - kappaArr[a]).slice(0, 6);
        const S_scan = S_bins[scanIdx];
        const R_scan = R_WL[scanIdx];
        const p_scan = [trajectoryData.x[scanIdx], trajectoryData.y[scanIdx], trajectoryData.z[scanIdx]];

        for (const b of indices) {{
            const S_b = S_scan[b];
            const norm = Math.sqrt(S_b[0]*S_b[0] + S_b[1]*S_b[1] + S_b[2]*S_b[2]);
            if (norm < 1e-6) continue;

            const d_body = [S_b[0]/norm, S_b[1]/norm, S_b[2]/norm];
            // Rotate to world frame: d_world = R_scan @ d_body
            const d_world = [
                R_scan[0][0]*d_body[0] + R_scan[0][1]*d_body[1] + R_scan[0][2]*d_body[2],
                R_scan[1][0]*d_body[0] + R_scan[1][1]*d_body[1] + R_scan[1][2]*d_body[2],
                R_scan[2][0]*d_body[0] + R_scan[2][1]*d_body[1] + R_scan[2][2]*d_body[2]
            ];

            const length = 0.3 * Math.log(1 + kappaArr[b]);
            const endX = p_scan[0] + length * d_world[0];
            const endY = p_scan[1] + length * d_world[1];
            const endZ = p_scan[2] + length * d_world[2];

            traces.push({{
                x: [p_scan[0], endX],
                y: [p_scan[1], endY],
                z: [p_scan[2], endZ],
                mode: 'lines',
                type: 'scatter3d',
                line: {{ color: '#f7b731', width: 4 }},
                showlegend: false,
                hoverinfo: 'skip'
            }});
        }}

        const layout = {{
            ...darkLayout,
            height: 450,
            title: {{ text: `3D Trajectory (Scan ${{scanIdx}} selected)`, font: {{color: '#00d4ff'}} }},
            scene: {{
                xaxis: {{ title: 'X (m)', gridcolor: '#1a1a2e', backgroundcolor: '#0f3460' }},
                yaxis: {{ title: 'Y (m)', gridcolor: '#1a1a2e', backgroundcolor: '#0f3460' }},
                zaxis: {{ title: 'Z (m)', gridcolor: '#1a1a2e', backgroundcolor: '#0f3460' }},
                bgcolor: '#0f3460',
                aspectmode: 'data'
            }},
            margin: {{ t: 50, b: 30, l: 30, r: 30 }},
            legend: {{ x: 0, y: 1, bgcolor: 'rgba(22, 33, 62, 0.8)' }}
        }};

        Plotly.react('trajectory', traces, layout, {{responsive: true}});
    }}

    // =====================================================================
    // Panel D: Excitation
    // =====================================================================
    function createExcitation() {{
        const traces = [
            // Row 1: Excitation scales
            {{ x: timelineData.scan_idx, y: timelineData.s_dt, name: 's_dt', line: {{color: '#00d4ff'}}, xaxis: 'x', yaxis: 'y' }},
            {{ x: timelineData.scan_idx, y: timelineData.s_ex, name: 's_ex', line: {{color: '#ff6b6b'}}, xaxis: 'x', yaxis: 'y' }},
            // Row 2: Fusion alpha
            {{ x: timelineData.scan_idx, y: timelineData.fusion_alpha, name: 'α', line: {{color: '#4ecdc4'}}, xaxis: 'x2', yaxis: 'y2' }},
            // Row 3: dt and diagnostics
            {{ x: timelineData.scan_idx, y: timelineData.dt_secs, name: 'dt (s)', line: {{color: '#a55eea'}}, xaxis: 'x3', yaxis: 'y3' }},
            {{ x: timelineData.scan_idx, y: timelineData.wahba_cost, name: 'Wahba', line: {{color: '#f7b731'}}, xaxis: 'x3', yaxis: 'y3' }},
            {{ x: timelineData.scan_idx, y: timelineData.translation_residual_norm, name: 'Trans', line: {{color: '#fd9644'}}, xaxis: 'x3', yaxis: 'y3' }},
        ];

        const layout = {{
            ...darkLayout,
            height: 350,
            showlegend: true,
            legend: {{ orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center' }},
            grid: {{ rows: 3, columns: 1, pattern: 'independent', roworder: 'top to bottom' }},
            xaxis: {{ ...darkLayout.xaxis, anchor: 'y', domain: [0, 1], showticklabels: false }},
            xaxis2: {{ ...darkLayout.xaxis, anchor: 'y2', domain: [0, 1], showticklabels: false }},
            xaxis3: {{ ...darkLayout.xaxis, anchor: 'y3', domain: [0, 1], title: 'Scan Index' }},
            yaxis: {{ ...darkLayout.yaxis, domain: [0.72, 1], title: 'Excitation' }},
            yaxis2: {{ ...darkLayout.yaxis, domain: [0.38, 0.66], title: 'Fusion α' }},
            yaxis3: {{ ...darkLayout.yaxis, domain: [0, 0.32], title: 'Diagnostics' }},
            margin: {{ t: 50, b: 40, l: 60, r: 40 }},
            shapes: createExcitationVerticalLines(currentScan),
        }};

        Plotly.newPlot('excitation', traces, layout, {{responsive: true}});
    }}

    function createExcitationVerticalLines(scanIdx) {{
        const shapes = [];
        for (let i = 0; i < 3; i++) {{
            shapes.push({{
                type: 'line',
                x0: scanIdx, x1: scanIdx,
                y0: 0, y1: 1,
                xref: 'x' + (i === 0 ? '' : (i + 1)),
                yref: 'paper',
                line: {{ color: '#ff6b6b', width: 2, dash: 'dash' }}
            }});
        }}
        return shapes;
    }}

    // =====================================================================
    // Update all panels when slider changes
    // =====================================================================
    function updateAllPanels(scanIdx) {{
        currentScan = scanIdx;

        // Update info display
        document.getElementById('scan-display').textContent = scanIdx;
        document.getElementById('info-dt').textContent = timelineData.dt_secs[scanIdx].toFixed(3);
        document.getElementById('info-alpha').textContent = timelineData.fusion_alpha[scanIdx].toFixed(3);
        document.getElementById('info-logdet').textContent = timelineData.logdet_L_total[scanIdx].toFixed(1);

        // Update timeline vertical lines
        Plotly.relayout('timeline', {{ shapes: createVerticalLines(scanIdx, 5) }});

        // Update excitation vertical lines
        Plotly.relayout('excitation', {{ shapes: createExcitationVerticalLines(scanIdx) }});

        // Update heatmap
        createHeatmap(scanIdx);

        // Update trajectory
        createTrajectory(scanIdx);
    }}

    // =====================================================================
    // Initialize
    // =====================================================================
    document.addEventListener('DOMContentLoaded', function() {{
        createTimeline();
        createHeatmap(currentScan);
        createTrajectory(currentScan);
        createExcitation();

        // Update info display
        document.getElementById('info-dt').textContent = timelineData.dt_secs[currentScan].toFixed(3);
        document.getElementById('info-alpha').textContent = timelineData.fusion_alpha[currentScan].toFixed(3);
        document.getElementById('info-logdet').textContent = timelineData.logdet_L_total[currentScan].toFixed(1);

        // Slider event
        const slider = document.getElementById('scan-slider');
        slider.addEventListener('input', function(e) {{
            updateAllPanels(parseInt(e.target.value));
        }});

        // Click on timeline to select scan
        document.getElementById('timeline').on('plotly_click', function(data) {{
            if (data.points && data.points.length > 0) {{
                const scanIdx = data.points[0].x;
                slider.value = scanIdx;
                updateAllPanels(scanIdx);
            }}
        }});

        // Click on excitation panel too
        document.getElementById('excitation').on('plotly_click', function(data) {{
            if (data.points && data.points.length > 0) {{
                const scanIdx = data.points[0].x;
                slider.value = scanIdx;
                updateAllPanels(scanIdx);
            }}
        }});
    }});
    </script>
</body>
</html>
"""

    # Write HTML file
    if output_path:
        html_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(html_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        # Use temp file
        temp_fd, html_path = tempfile.mkstemp(suffix=".html", text=True)
        os.close(temp_fd)  # Close the file descriptor, we'll open it for writing below
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Dashboard saved to: {html_path}")
    
    # Try to open in browser (only if not explicitly saving to file)
    if not output_path:
        if open_browser_wayland_compatible(html_path):
            print("Dashboard opened in browser")
        else:
            print("")
            print("Could not automatically open browser.")
            print(f"Please manually open: {html_path}")
            print("")
            print("Or use --output to save to a specific location:")
            print(f"  python tools/slam_dashboard.py <diagnostics.npz> --output dashboard.html")
    
    return html_path


def main():
    parser = argparse.ArgumentParser(
        description="Golden Child SLAM v2 Debugging Dashboard"
    )
    parser.add_argument(
        "diagnostics_file",
        type=str,
        help="Path to diagnostics NPZ file (e.g., /tmp/gc_slam_diagnostics.npz)",
    )
    parser.add_argument(
        "--scan",
        type=int,
        default=0,
        help="Initial selected scan index (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save HTML file to specified path (does not auto-open browser)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.diagnostics_file):
        print(f"Error: File not found: {args.diagnostics_file}")
        print("")
        print("Expected file: diagnostics NPZ file from a SLAM run")
        print("This file is created by the backend node during operation.")
        print("")
        print("To generate this file, run the SLAM pipeline:")
        print("  ./tools/run_and_evaluate_gc.sh")
        print("")
        print("Or check if a previous run created it at:")
        print("  /tmp/gc_slam_diagnostics.npz")
        print("  results/gc_*/diagnostics.npz")
        sys.exit(1)

    print(f"Loading diagnostics from: {args.diagnostics_file}")
    data = load_diagnostics_npz(args.diagnostics_file)

    n_scans = int(data.get("n_scans", 0))
    print(f"Loaded {n_scans} scans")

    # Print available keys for debugging
    print(f"Available data keys: {sorted(data.keys())}")

    if n_scans == 0:
        print("")
        print("ERROR: No scan data found in file!")
        print("The diagnostics file exists but contains 0 scans.")
        print("")
        print("This could mean:")
        print("  1. The SLAM run didn't complete successfully")
        print("  2. No LiDAR scans were processed")
        print("  3. The diagnostics weren't saved properly")
        print("")
        print("Check the SLAM log for errors.")
        sys.exit(1)

    # Validate selected scan
    if args.scan >= n_scans:
        print(f"Warning: Requested scan {args.scan} >= n_scans {n_scans}, using scan 0")
        args.scan = 0

    # Create dashboard
    html_path = create_full_dashboard(data, args.scan, output_path=args.output)
    
    if html_path:
        print(f"\n✓ Dashboard ready at: {html_path}")
        if args.output:
            print(f"  Saved to: {os.path.abspath(args.output)}")
            print(f"  Open manually in your browser or use: xdg-open {html_path}")


if __name__ == "__main__":
    main()
