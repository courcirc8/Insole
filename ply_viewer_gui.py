#!/usr/bin/env python3
"""
ply_viewer_gui.py

GUI application for viewing PLY files with 3D visualization and Z-axis colormap.
Features:
- File browser for PLY/PCD selection
- 3D scatter plot with Z-axis coloring
- Interactive rotation and zoom
- Colormap selection
- Point size adjustment
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


class PLYViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("PLY Viewer - 3D Point Cloud Visualization")
        self.root.geometry("1200x800")
        
        # Current data
        self.points = None
        self.current_file = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # File selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(file_frame, text="Browse PLY/PCD", command=self.browse_file).pack(side=tk.LEFT)
        self.file_label = ttk.Label(file_frame, text="No file selected", width=40)
        self.file_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Visualization controls
        viz_frame = ttk.Frame(control_frame)
        viz_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(viz_frame, text="Colormap:").pack(side=tk.LEFT)
        self.colormap_var = tk.StringVar(value="viridis")
        colormap_combo = ttk.Combobox(viz_frame, textvariable=self.colormap_var, 
                                     values=["viridis", "plasma", "coolwarm", "jet", "rainbow", "terrain"], 
                                     width=10)
        colormap_combo.pack(side=tk.LEFT, padx=(5, 10))
        colormap_combo.bind("<<ComboboxSelected>>", self.update_plot)
        
        ttk.Label(viz_frame, text="Point Size:").pack(side=tk.LEFT)
        self.point_size_var = tk.DoubleVar(value=1.0)
        point_size_scale = ttk.Scale(viz_frame, from_=0.1, to=5.0, variable=self.point_size_var, 
                                    orient=tk.HORIZONTAL, length=100)
        point_size_scale.pack(side=tk.LEFT, padx=(5, 10))
        point_size_scale.bind("<Motion>", self.update_plot)
        
        # Info panel
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(side=tk.RIGHT)
        
        self.info_label = ttk.Label(info_frame, text="Load a PLY file to see info", width=30)
        self.info_label.pack()
        
        # Matplotlib figure
        self.fig = plt.Figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()
        
        # Initial empty plot
        self.setup_empty_plot()
        
    def setup_empty_plot(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, 0.5, "Load a PLY file to visualize", 
                    transform=self.ax.transAxes, ha='center', va='center', fontsize=14)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.canvas.draw()
        
    def browse_file(self):
        filetypes = [
            ("Point clouds", "*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.pts"),
            ("PLY files", "*.ply"),
            ("PCD files", "*.pcd"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select PLY/PCD file",
            filetypes=filetypes
        )
        
        if filename:
            self.load_file(filename)
            
    def load_file(self, filepath):
        try:
            # Load point cloud
            pcd = o3d.io.read_point_cloud(filepath)
            if pcd.is_empty():
                messagebox.showerror("Error", f"Failed to load or empty file: {filepath}")
                return
                
            self.points = np.asarray(pcd.points)
            self.current_file = filepath
            
            # Update file label
            filename = os.path.basename(filepath)
            self.file_label.config(text=f"Loaded: {filename}")
            
            # Update info
            n_points = self.points.shape[0]
            z_min, z_max = self.points[:, 2].min(), self.points[:, 2].max()
            z_span = z_max - z_min
            self.info_label.config(text=f"Points: {n_points:,} | Z: [{z_min:.1f}, {z_max:.1f}] (span: {z_span:.1f})")
            
            # Plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            
    def update_plot(self, event=None):
        if self.points is None:
            return
            
        self.ax.clear()
        
        # Downsample for performance if too many points
        points = self.points
        max_points = 100000
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            
        # 3D scatter with Z colormap
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        scatter = self.ax.scatter(
            x, y, z,
            c=z,  # Color by Z-axis
            cmap=self.colormap_var.get(),
            s=self.point_size_var.get(),
            alpha=0.6
        )
        
        # Colorbar
        if hasattr(self, 'colorbar'):
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(scatter, ax=self.ax, shrink=0.8, label='Z-axis')
        
        # Labels and title
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y") 
        self.ax.set_zlabel("Z")
        
        if self.current_file:
            filename = os.path.basename(self.current_file)
            self.ax.set_title(f"{filename} ({len(self.points):,} points)")
        
        # Equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.canvas.draw()


def main():
    # Check if file provided as argument
    if len(sys.argv) > 1:
        initial_file = sys.argv[1]
        if not os.path.isfile(initial_file):
            print(f"File not found: {initial_file}")
            return 1
    else:
        initial_file = None
    
    # Create GUI
    root = tk.Tk()
    app = PLYViewer(root)
    
    # Load initial file if provided
    if initial_file:
        app.load_file(initial_file)
    
    # Start GUI loop
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
