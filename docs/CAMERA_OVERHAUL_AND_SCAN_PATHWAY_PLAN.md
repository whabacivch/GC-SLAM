Camera Overhaul + JAX Extraction Plan
Goal
Single-path camera I/O (ring buffer, pick by scan time) + clean JAX boundaries with MA HEX 3D for LiDAR extraction.

Principle
One conversion boundary per path. No JAX↔NumPy round-trips.

Part 1: Camera I/O Overhaul (from existing plan)
1.1 New RGBDImage.msg
std_msgs/Header header
sensor_msgs/Image rgb
sensor_msgs/Image depth
File: fl_ws/src/fl_slam_poc/msg/RGBDImage.msg (new)

1.2 Unified C++ node: camera_rgbd_node
Replace image_decompress_cpp + depth_passthrough with single node:

Subscribe: compressed RGB + raw depth
Decode RGB (OpenCV), scale depth (16UC1→32FC1 if needed)
Pair by timestamp (max_dt param), publish single RGBDImage
File: fl_ws/src/fl_slam_poc/src/camera_rgbd_node.cpp (new or replace image_decompress_node.cpp)

1.3 Backend: one subscription, ring buffer
Remove: _latest_rgb, _latest_depth, dual subscriptions
Add: _rgbd_buf ring buffer, single _on_camera_rgbd callback
On LiDAR: pick frame by argmin |t_frame - t_scan|
File: fl_ws/src/fl_slam_poc/fl_slam_poc/backend/backend_node.py

1.4 Launch cleanup
Remove: image_decompress_cpp, depth_passthrough nodes
Add: camera_rgbd_node with params
File: fl_ws/src/fl_slam_poc/launch/gc_rosbag.launch.py

Part 2: Clean Conversion Boundaries
2.1 Camera path (stays NumPy, one JAX boundary)

[NumPy]
RGB/Depth → ORB detect → depth sample → backproject → Feature3D list
                                                          ↓
[JAX]                                    feature_list_to_camera_batch()
                                         → MeasurementBatch
ORB is OpenCV, can't JAXify - this is fine
Verify no hidden JAX in visual_feature_extractor.py (currently clean)
Single conversion happens in feature_list_to_camera_batch()
Files to verify:

visual_feature_extractor.py
camera_batch_utils.py
2.2 LiDAR path (full JAX with MA HEX 3D)

[JAX - stays JAX throughout]
points_jax
  → MA HEX 3D binning (JAX, fixed buckets)
  → batched plane fit (jax.vmap)
  → batched covariance (jax.vmap)
  → MeasurementBatch (already JAX)
No conversions. Points enter as JAX, exit as JAX.

Part 3: MA HEX 3D for LiDAR Extraction
3.1 Extend MA HEX to 3D
Current: 2D hex grid (x,y BEV only), NumPy
Target: 3D hex grid (x,y hex + z linear), JAX-compatible

Changes to ma_hex_web.py:


@dataclass
class MAHex3DConfig:
    num_cells_1: int = 128      # hex grid x
    num_cells_2: int = 128      # hex grid y
    num_cells_z: int = 64       # linear z bins
    max_occupants: int = 32     # points per cell
    voxel_size: float = 0.1     # cell size (meters)

def hex_cell_3d_batch(points: jnp.ndarray, h: float) -> jnp.ndarray:
    """Vectorized 3D cell assignment. Returns (N, 3) cell indices."""
    # Hex for x,y
    s1 = points[:, 0]  # a1 = (1, 0)
    s2 = points[:, 0] * 0.5 + points[:, 1] * (jnp.sqrt(3) / 2)  # a2
    cell_1 = jnp.floor(s1 / h).astype(jnp.int32)
    cell_2 = jnp.floor(s2 / h).astype(jnp.int32)
    # Linear for z
    cell_z = jnp.floor(points[:, 2] / h).astype(jnp.int32)
    return jnp.stack([cell_1, cell_2, cell_z], axis=1)
Bucket structure (JAX-compatible):


@dataclass
class MAHex3DBucket:
    # Fixed-size arrays, JAX-friendly
    bucket: jnp.ndarray   # (n1*n2*nz, max_occupants) point indices, -1 = empty
    count: jnp.ndarray    # (n1*n2*nz,) occupancy per cell

    @staticmethod
    def create(config: MAHex3DConfig) -> 'MAHex3DBucket':
        n_cells = config.num_cells_1 * config.num_cells_2 * config.num_cells_z
        return MAHex3DBucket(
            bucket=jnp.full((n_cells, config.max_occupants), -1, dtype=jnp.int32),
            count=jnp.zeros(n_cells, dtype=jnp.int32),
        )
Binning (scatter-add):


def bin_points_3d(points: jnp.ndarray, config: MAHex3DConfig) -> MAHex3DBucket:
    """Bin N points into 3D hex grid. Fully vectorized."""
    cells = hex_cell_3d_batch(points, config.voxel_size)  # (N, 3)
    # Wrap to grid bounds
    cells = cells % jnp.array([config.num_cells_1, config.num_cells_2, config.num_cells_z])
    # Linear index
    linear = (cells[:, 0] * config.num_cells_2 * config.num_cells_z
              + cells[:, 1] * config.num_cells_z
              + cells[:, 2])
    # Build bucket using segment_sum or scatter
    # ... (use jax.ops.segment_sum for counts, custom scatter for indices)
3.2 Batched plane fit (JAX vmap)
Current: Per-voxel Python loop → NumPy SVD
Target: Batch all occupied cells → jax.vmap → batched eigensolver


@partial(jax.jit, static_argnums=(1,))
def fit_planes_batched(
    points: jnp.ndarray,           # (N, 3) all points
    bucket: MAHex3DBucket,         # binning result
    min_points: int = 5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns:
        centroids: (V, 3) voxel centroids
        normals: (V, 3) plane normals
        valid: (V,) bool mask for cells with enough points
    """
    # For each occupied cell, gather points (padded to max_occupants)
    # Compute centroid, covariance, smallest eigenvector

    def fit_one_cell(cell_indices, cell_count):
        # Gather points for this cell (masked by count)
        pts = points[cell_indices]  # (max_occupants, 3)
        mask = jnp.arange(len(cell_indices)) < cell_count

        # Centroid
        centroid = jnp.sum(pts * mask[:, None], axis=0) / jnp.maximum(cell_count, 1)

        # Covariance
        centered = (pts - centroid) * mask[:, None]
        cov = centered.T @ centered / jnp.maximum(cell_count - 1, 1)

        # Normal = smallest eigenvector
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # smallest eigenvalue

        return centroid, normal, cell_count >= min_points

    # vmap over all cells
    centroids, normals, valid = jax.vmap(fit_one_cell)(bucket.bucket, bucket.count)
    return centroids, normals, valid
3.3 Batched covariance (JAX vmap)

@jax.jit
def compute_surfel_covariances(
    centroids: jnp.ndarray,   # (V, 3)
    normals: jnp.ndarray,     # (V, 3)
    sigma_along: float,       # uncertainty along plane
    sigma_normal: float,      # uncertainty along normal
) -> jnp.ndarray:
    """Compute surfel covariances from plane fits. Returns (V, 3, 3)."""

    def cov_one(normal):
        # Build covariance: small along normal, larger in plane
        # Σ = R @ diag(σ_along², σ_along², σ_normal²) @ R.T
        # where R rotates z-axis to normal
        # ... (standard rotation matrix construction)
        pass

    return jax.vmap(cov_one)(normals)
3.4 Replace lidar_surfel_extraction.py
File: lidar_surfel_extraction.py


def extract_lidar_surfels_jax(
    points: jnp.ndarray,      # (N, 3) already JAX
    timestamps: jnp.ndarray,  # (N,)
    config: MAHex3DConfig,
) -> MeasurementBatch:
    """Full JAX LiDAR extraction. No NumPy conversions."""

    # 1. Bin points into 3D hex grid
    bucket = bin_points_3d(points, config)

    # 2. Fit planes (batched)
    centroids, normals, valid = fit_planes_batched(points, bucket)

    # 3. Compute covariances (batched)
    covariances = compute_surfel_covariances(centroids, normals, ...)

    # 4. Filter to valid, truncate/pad to N_SURFEL budget
    # 5. Build MeasurementBatch directly (already JAX)

    return measurement_batch
Part 4: Implementation Order
#	Task	Files
1	RGBDImage.msg + CMake/package.xml	msg/, CMakeLists.txt, package.xml
2	camera_rgbd_node C++	src/camera_rgbd_node.cpp
3	Backend ring buffer + single sub	backend_node.py
4	Launch cleanup	gc_rosbag.launch.py
5	MA HEX 3D (JAX binning)	ma_hex_web.py (new functions)
6	Batched plane fit (JAX vmap)	lidar_surfel_extraction.py
7	Batched covariance (JAX vmap)	lidar_surfel_extraction.py
8	Wire up new extraction, remove old	lidar_surfel_extraction.py, lidar_surfels.py
9	Verify camera path has one boundary	visual_feature_extractor.py, camera_batch_utils.py
Part 5: Verification
Build: colcon build --packages-select fl_slam_poc
Run with bag: ros2 launch fl_slam_poc gc_rosbag.launch.py bag_path:=...
Check topics:
/gc/sensors/camera_rgbd publishes RGBDImage
No /gc/sensors/camera_rgb or /gc/sensors/camera_depth (old topics gone)
Check extraction timing: Log extraction time per scan, should be comparable or faster
Check output: Surfels and camera features in MeasurementBatch, same downstream behavior
Summary
Camera I/O: Single node, single topic, ring buffer, pick by scan time
Camera extraction: NumPy (ORB is OpenCV), one JAX boundary at batch construction
LiDAR extraction: Full JAX with MA HEX 3D, no conversions
Principle: Each path has exactly one representation boundary