#!/usr/bin/env python3
"""
ply_viewer_web.py

Web-based PLY viewer with 3D visualization and Z-axis colormap.
Uses Dash/Plotly for interactive 3D visualization in the browser.
"""
import argparse
import os
import sys
from pathlib import Path
import glob

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc


class PLYViewerWeb:
    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.points = None
        self.current_file = None
        self.file_cache = {}  # Cache loaded files
        self.current_params = {}  # Current processing parameters
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("PLY Viewer - 3D Point Cloud Visualization", className="mb-4"),
                    
                    # File browser
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("File Browser", className="card-title"),
                            dcc.Dropdown(
                                id="file-dropdown",
                                placeholder="Select a PLY/PCD file...",
                                style={"marginBottom": "10px"}
                            ),
                            dbc.Button("Refresh File List", id="refresh-btn", color="secondary", size="sm")
                        ])
                    ], className="mb-3"),
                    
                    # File info
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("File Information", className="card-title"),
                            html.Div(id="file-info", children="No file loaded")
                        ])
                    ], className="mb-3"),
                    
                    # Controls
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Visualization Controls", className="card-title"),
                            
                            # Z-axis filtering
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Z Min Threshold (mm):"),
                                    dbc.Input(
                                        id="z-min-input",
                                        type="number",
                                        value=0,
                                        step=0.1,
                                        placeholder="0"
                                    )
                                ], width=6),
                                
                                dbc.Col([
                                    dbc.Label("Z Max Threshold (mm):"),
                                    dbc.Input(
                                        id="z-max-input", 
                                        type="number",
                                        value=35,
                                        step=0.1,
                                        placeholder="35"
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            
                            # Angle adjustment controls
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("X-Angle Adjustment (°):"),
                                    dbc.Input(
                                        id="x-angle-input",
                                        type="number",
                                        value=0,
                                        step=0.1,
                                        placeholder="0",
                                        min=-10,
                                        max=10
                                    )
                                ], width=6),
                                
                                dbc.Col([
                                    dbc.Label("Y-Angle Adjustment (°):"),
                                    dbc.Input(
                                        id="y-angle-input",
                                        type="number", 
                                        value=0,
                                        step=0.1,
                                        placeholder="0",
                                        min=-10,
                                        max=10
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            
                            # Save button
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "💾 Save Cleaned PLY",
                                        id="save-btn",
                                        color="success",
                                        size="lg",
                                        className="w-100",
                                        disabled=True
                                    ),
                                    html.Div(id="save-status", className="mt-2")
                                ], width=12)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Colormap:"),
                                    dcc.Dropdown(
                                        id="colormap-dropdown",
                                        options=[
                                            {"label": "Viridis", "value": "Viridis"},
                                            {"label": "Plasma", "value": "Plasma"},
                                            {"label": "Jet", "value": "Jet"},
                                            {"label": "Rainbow", "value": "Rainbow"},
                                            {"label": "Turbo", "value": "Turbo"},
                                        ],
                                        value="Viridis"
                                    )
                                ], width=6),
                                
                                dbc.Col([
                                    dbc.Label("Point Size:"),
                                    dcc.Slider(
                                        id="point-size-slider",
                                        min=1,
                                        max=10,
                                        step=1,
                                        value=3,
                                        marks={i: str(i) for i in range(1, 11)}
                                    )
                                ], width=6)
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Max Points (for performance):"),
                                    dcc.Slider(
                                        id="max-points-slider",
                                        min=10000,
                                        max=500000,
                                        step=10000,
                                        value=100000,
                                        marks={i: f"{i//1000}K" for i in range(50000, 550000, 100000)}
                                    )
                                ], width=12)
                            ], className="mt-3")
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    # 3D Plot
                    dcc.Graph(
                        id="3d-plot",
                        style={"height": "700px"},
                        config={"displayModeBar": True}
                    )
                ], width=8)
            ])
        ], fluid=True)
    
    def find_ply_files(self):
        """Find all PLY/PCD files in current directory and subdirectories."""
        patterns = ["**/*.ply", "**/*.pcd", "**/*.xyz", "**/*.pts"]
        files = []
        for pattern in patterns:
            found = glob.glob(pattern, recursive=True)
            files.extend(found)
        
        # Sort and create dropdown options
        files.sort()
        options = [{"label": f"{f} ({os.path.getsize(f)//1024}KB)", "value": f} for f in files if os.path.isfile(f)]
        print(f"Found {len(options)} PLY/PCD files")
        return options
    
    def setup_callbacks(self):
        @self.app.callback(
            Output("file-dropdown", "options"),
            Input("refresh-btn", "n_clicks"),
            prevent_initial_call=False
        )
        def refresh_file_list(n_clicks):
            options = self.find_ply_files()
            print(f"Refresh callback: found {len(options)} files")
            return options
        
        @self.app.callback(
            [Output("3d-plot", "figure"), Output("file-info", "children"), Output("save-btn", "disabled")],
            [Input("file-dropdown", "value"),
             Input("colormap-dropdown", "value"),
             Input("point-size-slider", "value"),
             Input("max-points-slider", "value"),
             Input("z-min-input", "value"),
             Input("z-max-input", "value"),
             Input("x-angle-input", "value"),
             Input("y-angle-input", "value")]
        )
        def update_plot(selected_file, colormap, point_size, max_points, z_min, z_max, x_angle, y_angle):
            # Load file if selected and different from current
            if selected_file and selected_file != self.current_file:
                if not self.load_file_cached(selected_file):
                    # Failed to load
                    fig = go.Figure()
                    fig.add_trace(go.Scatter3d(
                        x=[0], y=[0], z=[0],
                        mode="text",
                        text=[f"Failed to load: {selected_file}"],
                        textposition="middle center"
                    ))
                    return fig, f"Error loading: {selected_file}", True
            
            if self.points is None:
                # Empty plot
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode="text",
                    text=["Load a PLY file using command line:<br>python ply_viewer_web.py file.ply"],
                    textposition="middle center"
                ))
                fig.update_layout(
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z"
                    ),
                    title="PLY Viewer - No file loaded"
                )
                return fig, "No file loaded", True
            
            # Apply angle adjustments first
            points = self.points.copy()
            
            # Handle None values for angles (use defaults)
            if x_angle is None:
                x_angle = 0
            if y_angle is None:
                y_angle = 0
                
            # Apply rotations if angles are non-zero
            if x_angle != 0 or y_angle != 0:
                # Convert angles to radians
                x_rad = np.radians(x_angle)
                y_rad = np.radians(y_angle)
                
                # Create rotation matrices
                # Rotation around X-axis
                rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(x_rad), -np.sin(x_rad)],
                    [0, np.sin(x_rad), np.cos(x_rad)]
                ])
                
                # Rotation around Y-axis
                ry = np.array([
                    [np.cos(y_rad), 0, np.sin(y_rad)],
                    [0, 1, 0],
                    [-np.sin(y_rad), 0, np.cos(y_rad)]
                ])
                
                # Apply combined rotation (Y then X)
                rotation_matrix = np.dot(rx, ry)
                points = np.dot(points, rotation_matrix.T)
            
            # Handle None values for thresholds (use defaults)
            if z_min is None:
                z_min = 0
            if z_max is None:
                z_max = 35
                
            # Filter points by Z-axis thresholds
            z_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
            points_filtered = points[z_mask]
            
            if len(points_filtered) == 0:
                # No points in range
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode="text",
                    text=[f"No points in Z range [{z_min:.1f}, {z_max:.1f}]"],
                    textposition="middle center"
                ))
                fig.update_layout(
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z"
                    ),
                    title=f"No points in Z range [{z_min:.1f}, {z_max:.1f}]"
                )
                return fig, f"No points in Z range [{z_min:.1f}, {z_max:.1f}]", True
            
            # Downsample if needed (after filtering)
            points = points_filtered
            if len(points) > max_points:
                idx = np.random.choice(len(points), max_points, replace=False)
                points = points[idx]
            
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            # Create 3D scatter plot
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=z,  # Color by Z-axis
                    colorscale=colormap,
                    opacity=0.8,
                    colorbar=dict(title="Z-axis")
                )
            )])
            
            # Layout
            fig.update_layout(
                title=f"{os.path.basename(self.current_file)} - {len(points_filtered):,}/{len(self.points):,} points (Z: [{z_min:.1f}, {z_max:.1f}])",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y", 
                    zaxis_title="Z",
                    aspectmode="data"
                ),
                width=800,
                height=700
            )
            
            # File info
            n_points_total = len(self.points)
            n_points_filtered = len(points_filtered)
            n_points_displayed = len(points)
            
            # Original file stats
            original_z_min, original_z_max = self.points[:, 2].min(), self.points[:, 2].max()
            original_z_span = original_z_max - original_z_min
            original_x_span = self.points[:, 0].max() - self.points[:, 0].min()
            original_y_span = self.points[:, 1].max() - self.points[:, 1].min()
            
            # Filtered stats
            if len(points_filtered) > 0:
                filtered_z_min, filtered_z_max = points_filtered[:, 2].min(), points_filtered[:, 2].max()
                filtered_z_span = filtered_z_max - filtered_z_min
            else:
                filtered_z_min = filtered_z_max = filtered_z_span = 0
            
            filter_percentage = (n_points_filtered / n_points_total) * 100 if n_points_total > 0 else 0
            
            info = html.Div([
                html.P(f"📁 File: {os.path.basename(self.current_file)}"),
                html.P(f"📊 Total Points: {n_points_total:,}"),
                html.P(f"🎯 Z Filter: [{z_min:.1f}, {z_max:.1f}] mm"),
                html.P(f"✅ Filtered Points: {n_points_filtered:,} ({filter_percentage:.1f}%)"),
                html.P(f"👁️ Displayed Points: {n_points_displayed:,}"),
                html.P(f"📏 Original Dimensions:"),
                html.P(f"  X: {original_x_span:.1f} units"),
                html.P(f"  Y: {original_y_span:.1f} units"), 
                html.P(f"  Z: {original_z_span:.1f} units"),
                html.P(f"🎯 Original Z Range: [{original_z_min:.1f}, {original_z_max:.1f}]"),
                html.P(f"🎯 Filtered Z Range: [{filtered_z_min:.1f}, {filtered_z_max:.1f}]")
            ])
            
            # Enable save button if we have filtered data
            save_disabled = len(points_filtered) == 0
            
            # Store current processing parameters for save function
            self.current_params = {
                'points_filtered': points_filtered,
                'z_min': z_min,
                'z_max': z_max,
                'x_angle': x_angle,
                'y_angle': y_angle,
                'filename': self.current_file
            }
            
            return fig, info, save_disabled
        
        @self.app.callback(
            Output("save-status", "children"),
            Input("save-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def save_cleaned_ply(n_clicks):
            if not n_clicks or not hasattr(self, 'current_params') or not self.current_params:
                return ""
            
            try:
                params = self.current_params
                points_filtered = params['points_filtered']
                
                if len(points_filtered) == 0:
                    return dbc.Alert("❌ No points to save!", color="danger", dismissable=True)
                
                # Generate output filename
                import os
                from pathlib import Path
                
                base_name = Path(params['filename']).stem if params['filename'] else "cleaned"
                output_dir = "outputs/fine_tuned"
                os.makedirs(output_dir, exist_ok=True)
                
                # Create descriptive filename
                filename_parts = [base_name]
                if params['z_min'] != 0 or params['z_max'] != 35:
                    filename_parts.append(f"Z{params['z_min']:.1f}-{params['z_max']:.1f}mm")
                if params['x_angle'] != 0:
                    filename_parts.append(f"Xrot{params['x_angle']:.1f}deg")
                if params['y_angle'] != 0:
                    filename_parts.append(f"Yrot{params['y_angle']:.1f}deg")
                
                output_filename = "_".join(filename_parts) + "_fine_tuned.ply"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save the filtered points
                import open3d as o3d
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_filtered))
                success = o3d.io.write_point_cloud(output_path, pcd)
                
                if success:
                    return dbc.Alert(
                        [
                            html.Strong("✅ Success! "),
                            f"Saved {len(points_filtered):,} points to:",
                            html.Br(),
                            html.Code(output_path)
                        ],
                        color="success",
                        dismissable=True
                    )
                else:
                    return dbc.Alert("❌ Failed to save file!", color="danger", dismissable=True)
                    
            except Exception as e:
                return dbc.Alert(f"❌ Error: {str(e)}", color="danger", dismissable=True)
    
    def load_file_cached(self, filepath):
        """Load file with caching."""
        if filepath in self.file_cache:
            self.points = self.file_cache[filepath]
            self.current_file = filepath
            return True
        
        try:
            pcd = o3d.io.read_point_cloud(filepath)
            if pcd.is_empty():
                print(f"Error: Failed to load or empty file: {filepath}")
                return False
                
            points = np.asarray(pcd.points)
            self.file_cache[filepath] = points
            self.points = points
            self.current_file = filepath
            print(f"Loaded: {filepath} ({len(self.points):,} points)")
            return True
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def load_file_from_path(self, filepath):
        """Load file programmatically (for startup)."""
        return self.load_file_cached(filepath)
    
    def run(self, host="127.0.0.1", port=8050, debug=False):
        """Start the web server."""
        print(f"Starting PLY Viewer at http://{host}:{port}")
        print("Press Ctrl+C to stop")
        self.app.run(host=host, port=port, debug=debug)


def main():
    parser = argparse.ArgumentParser(description="Web-based PLY viewer with Z-axis colormap.")
    parser.add_argument("file", nargs="?", help="PLY/PCD file to load initially.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8050, help="Port number (default: 8050).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()
    
    viewer = PLYViewerWeb()
    
    if args.file:
        if not os.path.isfile(args.file):
            print(f"File not found: {args.file}")
            return 1
        if not viewer.load_file_from_path(args.file):
            return 1
    
    try:
        viewer.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
