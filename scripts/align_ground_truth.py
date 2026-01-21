#!/usr/bin/env python3
"""
Align M3DGR ground truth timestamps with SLAM trajectory.

M3DGR ground truth timestamps are in absolute UNIX time.
SLAM trajectory uses ROS2 simulation time starting near 0.

This script finds the time offset and creates an aligned ground truth file.
"""
import sys

def align_timestamps(gt_file, est_file, output_file):
    # Read ground truth
    gt_lines = []
    with open(gt_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                gt_lines.append([float(parts[0])] + [float(p) for p in parts[1:8]])
    
    # Read estimated trajectory
    est_lines = []
    with open(est_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                est_lines.append([float(parts[0])] + [float(p) for p in parts[1:8]])
    
    if not gt_lines or not est_lines:
        print("ERROR: Empty trajectory file(s)")
        return False
    
    # Compute time offset (ground truth is absolute UNIX, estimate starts near 0)
    time_offset = gt_lines[0][0] - est_lines[0][0]
    print(f"Time offset: {time_offset:.6f} seconds")
    
    # Align ground truth timestamps
    with open(output_file, 'w') as f:
        f.write("# timestamp x y z qx qy qz qw\n")
        for gt_line in gt_lines:
            aligned_time = gt_line[0] - time_offset
            f.write(f"{aligned_time:.6f} {' '.join(f'{v:.6f}' for v in gt_line[1:])}\n")
    
    print(f"Aligned ground truth saved to: {output_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: align_ground_truth.py <gt_file> <est_file> <output_file>")
        sys.exit(1)
    
    align_timestamps(sys.argv[1], sys.argv[2], sys.argv[3])
