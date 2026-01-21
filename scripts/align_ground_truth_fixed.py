#!/usr/bin/env python3
"""
Fix alignment: Find the correct time offset by looking at overlapping timestamp ranges.
"""
import sys

def align_timestamps_fixed(gt_file, est_file, output_file):
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

    # Find the actual time ranges
    gt_start = min(line[0] for line in gt_lines)
    gt_end = max(line[0] for line in gt_lines)
    est_start = min(line[0] for line in est_lines if line[0] > 1000000000)  # Skip timestamp 0 entries
    est_end = max(line[0] for line in est_lines)

    print(f"Ground truth range: {gt_start:.6f} to {gt_end:.6f}")
    print(f"Estimated range: {est_start:.6f} to {est_end:.6f}")

    # The offset should be: est_start - gt_start
    # This aligns the ground truth start time with the SLAM start time
    time_offset = est_start - gt_start
    print(f"Calculated offset: {time_offset:.6f} seconds")

    # Apply the offset to ground truth
    with open(output_file, 'w') as f:
        f.write("# timestamp x y z qx qy qz qw\n")
        for gt_line in gt_lines:
            aligned_time = gt_line[0] + time_offset
            f.write(f"{aligned_time:.6f} {' '.join(f'{v:.6f}' for v in gt_line[1:])}\n")

    print(f"Fixed aligned ground truth saved to: {output_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: align_ground_truth_fixed.py <gt_file> <est_file> <output_file>")
        sys.exit(1)

    align_timestamps_fixed(sys.argv[1], sys.argv[2], sys.argv[3])
