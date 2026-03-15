"""
view_scan.py

Display 3D scans (point clouds/meshes) with GUI file picker or direct path.
Supports PLY, PCD, STL, OBJ, and other common 3D formats.
"""
import argparse
import os
import sys

import open3d as o3d

try:
	import tkinter as tk
	from tkinter import filedialog
	_HAS_TK = True
except Exception:
	_HAS_TK = False


POINT_CLOUD_EXTS = {
	".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"
}
MESH_EXTS = {
	".stl", ".obj", ".off", ".gltf", ".glb", ".fbx", ".dae"
}


def load_scan(path: str):
	"""Load a scan file as an Open3D geometry (point cloud or mesh)."""
	ext = os.path.splitext(path)[1].lower()
	if ext in POINT_CLOUD_EXTS:
		pcd = o3d.io.read_point_cloud(path)
		if pcd.is_empty():
			raise ValueError(f"Loaded point cloud is empty: {path}")
		return pcd
	elif ext in MESH_EXTS:
		mesh = o3d.io.read_triangle_mesh(path)
		if mesh.is_empty():
			raise ValueError(f"Loaded mesh is empty: {path}")
		# Ensure normals for better shading when visualizing
		if not mesh.has_vertex_normals():
			mesh.compute_vertex_normals()
		return mesh
	else:
		raise ValueError(
			f"Unsupported file extension '{ext}'.\n"
			f"Supported point clouds: {sorted(POINT_CLOUD_EXTS)}\n"
			f"Supported meshes: {sorted(MESH_EXTS)}"
		)


def pick_file_interactive() -> str:
	"""Open a native file dialog to pick a supported file. Re-prompts until a file is selected."""
	if not _HAS_TK:
		return ""
	root = tk.Tk()
	root.withdraw()
	filetypes = [
		("3D files", "*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.pts *.stl *.obj *.off *.gltf *.glb *.fbx *.dae"),
		("Point clouds", "*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.pts"),
		("Meshes", "*.stl *.obj *.off *.gltf *.glb *.fbx *.dae"),
		("All files", "*.*"),
	]
	while True:
		path = filedialog.askopenfilename(title="Select a 3D scan", filetypes=filetypes)
		if path:
			return os.path.abspath(path)
		# If user cancels, keep prompting; allow exit via closing terminal/interrupt
		# Short sleep yields back to UI loop
		root.update()


def main() -> int:
	parser = argparse.ArgumentParser(description="Load and display a 3D scan.")
	parser.add_argument(
		"--path",
		"-p",
		type=str,
		default="",
		help="Path to scan file (PLY/PCD/OBJ/STL/...). If omitted, a file picker is shown.",
	)
	args = parser.parse_args()

	path = args.path
	if not path:
		path = pick_file_interactive()
	elif not os.path.isabs(path):
		path = os.path.join(os.getcwd(), path)

	if not os.path.isfile(path):
		print(f"File not found: {path}", file=sys.stderr)
		return 1

	print(f"Loading: {path}")
	geometry = load_scan(path)

	print("Displaying... Close window to exit.")
	o3d.visualization.draw_geometries([geometry])
	return 0


if __name__ == "__main__":
	sys.exit(main()) 