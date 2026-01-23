# FL-SLAM Engineering Discipline Cleanup Plan

**Date**: 2026-01-23
**Goal**: Reduce accidental complexity, eliminate dead code, leverage Pydantic

---

## Phase 1: Dead Code Detection (MANUAL)

### 1.1 Find Unused Variables
```bash
# Pattern: Variables assigned but never read
grep -rn "^\s*[a-z_]\+ = " --include="*.py" | grep -v "__pycache__"
```

### 1.2 Find Unused Imports
```bash
# Check for imports never referenced
# Requires manual inspection
```

### 1.3 Find Unused Functions/Methods
```bash
# Private methods (_method) that are never called
# Public methods with zero references
```

---

## Phase 2: Duplicate Validation Removal

### Problem: Validation scattered across codebase

**Current State**:
- `frontend_node.py`: Lines 888-1014 (ICP + loop validation, ~126 lines)
- `backend/factors/imu.py`: Lines 129-185 (IMU validation, ~56 lines)
- `common/validation.py`: Utility functions

**Solution**: Create validation decorators

### 2.1 Create Validation Decorators

**File**: `fl_slam_poc/common/validation_decorators.py`

```python
"""Validation decorators to reduce boilerplate."""
from functools import wraps
from typing import Callable
from fl_slam_poc.common.validation import (
    validate_timestamp,
    validate_covariance,
    detect_hardcoded_value,
    ContractViolation
)
from fl_slam_poc.common.op_report import OpReport

def validate_imu_segment(func: Callable) -> Callable:
    """Decorator to validate IMU segment inputs."""
    @wraps(func)
    def wrapper(backend, msg, *args, **kwargs):
        try:
            # Extract data from message
            t_i = msg.t_i
            t_j = msg.t_j
            accel = np.array(msg.accel).reshape(-1, 3)
            gyro = np.array(msg.gyro).reshape(-1, 3)
            stamps = np.array(msg.stamp)

            # Validate timestamps
            validate_timestamp(t_i, "imu_segment.t_i")
            validate_timestamp(t_j, "imu_segment.t_j")

            if t_j <= t_i:
                raise ContractViolation(f"t_j ({t_j:.6f}) <= t_i ({t_i:.6f})")

            # Validate measurements are finite
            if not np.all(np.isfinite(accel)):
                raise ContractViolation("imu_segment.accel: Contains inf/nan")
            if not np.all(np.isfinite(gyro)):
                raise ContractViolation("imu_segment.gyro: Contains inf/nan")
            if not np.all(np.isfinite(stamps)):
                raise ContractViolation("imu_segment.stamps: Contains inf/nan")

            # Call original function
            return func(backend, msg, *args, **kwargs)

        except ContractViolation as e:
            backend.get_logger().error(f"IMU segment contract violation: {e}")
            # Publish error report
            from fl_slam_poc.backend.diagnostics import publish_report
            publish_report(backend, OpReport(
                name="IMUSegmentContractViolation",
                exact=False,
                family_in="IMU",
                family_out="None",
                closed_form=False,
                domain_projection=False,
                metrics={"error": str(e)},
                notes=f"Contract violation: {e}",
            ), backend.pub_report)
            return  # Skip processing

    return wrapper

def validate_icp_result(func: Callable) -> Callable:
    """Decorator to validate ICP result."""
    @wraps(func)
    def wrapper(self, icp_result, *args, **kwargs):
        if icp_result is None:
            return None

        try:
            # Validate transformation is finite
            if not np.all(np.isfinite(icp_result.transformation)):
                raise ContractViolation(
                    f"icp_result.transformation: Contains inf/nan"
                )

            # Validate MSE is finite and non-negative
            if not np.isfinite(icp_result.mse) or icp_result.mse < 0:
                raise ContractViolation(
                    f"icp_result.mse: Invalid value: {icp_result.mse}"
                )

            return func(self, icp_result, *args, **kwargs)

        except ContractViolation as e:
            self.get_logger().error(f"ICP result contract violation: {e}")
            return None

    return wrapper
```

**Savings**: ~100 lines removed from frontend/backend

---

## Phase 3: Pydantic-Driven Parameter Management

### Problem: 104 lines of ROS parameter boilerplate in frontend_node.py

**Current State** (Lines 96-199):
```python
def _declare_parameters(self):
    self.declare_parameter("scan_topic", "/scan")
    self.declare_parameter("odom_topic", "/odom")
    # ... 70+ more parameters
```

**Solution**: Auto-generate from Pydantic models

### 3.1 Enhance FrontendParams Pydantic Model

**File**: `fl_slam_poc/common/param_models.py` (ALREADY EXISTS)

Add ROS metadata:

```python
from pydantic import BaseModel, Field
from typing import Literal

class FrontendParams(BaseModel):
    """Frontend parameters with ROS metadata."""

    # Topics
    scan_topic: str = Field(
        default="/scan",
        description="LaserScan topic",
        ros_param=True  # Auto-generate ROS parameter
    )

    odom_topic: str = Field(
        default="/odom",
        description="Odometry topic",
        ros_param=True
    )

    # ... all other parameters
```

### 3.2 Create ROS Parameter Generator

**File**: `fl_slam_poc/common/ros_param_generator.py` (NEW)

```python
"""Auto-generate ROS parameters from Pydantic models."""
from typing import Type
from pydantic import BaseModel

def declare_parameters_from_model(node, model: Type[BaseModel]):
    """Auto-declare ROS parameters from Pydantic model."""
    for field_name, field_info in model.model_fields.items():
        if field_info.json_schema_extra and field_info.json_schema_extra.get("ros_param"):
            default_value = field_info.default
            node.declare_parameter(field_name, default_value)

def load_parameters_to_model(node, model: Type[BaseModel]) -> BaseModel:
    """Load ROS parameters into Pydantic model instance."""
    values = {}
    for field_name in model.model_fields:
        values[field_name] = node.get_parameter(field_name).value
    return model(**values)
```

### 3.3 Refactor Frontend Node

**Before** (1,412 lines):
```python
def _declare_parameters(self):
    self.declare_parameter("scan_topic", "/scan")
    # ... 104 lines

def _validate_params(self) -> FrontendParams:
    values: dict[str, object] = {}
    for name in FrontendParams.model_fields:
        values[name] = self.get_parameter(name).value
    return FrontendParams(**values)
```

**After** (~1,300 lines):
```python
from fl_slam_poc.common.ros_param_generator import (
    declare_parameters_from_model,
    load_parameters_to_model
)

def _declare_parameters(self):
    declare_parameters_from_model(self, FrontendParams)

def _validate_params(self) -> FrontendParams:
    return load_parameters_to_model(self, FrontendParams)
```

**Savings**: ~104 lines removed, DRY principle enforced

---

## Phase 4: Extract RGB-D Evidence Publishing

### Problem: 118-line method in frontend_node.py doing actual processing

**File**: `frontend_node.py`, lines 1263-1380

**Solution**: Move to `frontend/sensors/rgbd_processor.py`

### 4.1 Create Extraction Method

**File**: `fl_slam_poc/frontend/sensors/rgbd_processor.py` (ALREADY EXISTS)

Add method:
```python
def create_rgbd_evidence_message(
    rgb: np.ndarray,
    depth: np.ndarray,
    pose_odom_base: np.ndarray,
    camera_frame: str,
    stamp,
    sensor_io: SensorIO,
    max_points: int = 500
) -> Optional[String]:
    """
    Create RGB-D evidence message.

    Returns String message with JSON payload or None if failed.
    """
    # Move all logic from _publish_rgbd_evidence here
    # ...
```

### 4.2 Simplify Frontend Node

**Before**:
```python
def _publish_rgbd_evidence(self, rgb, depth, pose_odom_base, camera_frame, stamp):
    # 118 lines of processing logic
```

**After**:
```python
def _publish_rgbd_evidence(self, rgb, depth, pose_odom_base, camera_frame, stamp):
    from fl_slam_poc.frontend.sensors.rgbd_processor import create_rgbd_evidence_message

    msg = create_rgbd_evidence_message(
        rgb, depth, pose_odom_base, camera_frame, stamp, self.sensor_io,
        max_points=int(self.get_parameter("rgbd_max_points_per_msg").value)
    )

    if msg and self.pub_rgbd:
        self.pub_rgbd.publish(msg)
```

**Savings**: ~110 lines moved to specialized module

---

## Phase 5: Consolidate Validation Helpers

### Problem: Validation logic in `validation.py` could have helpers for common patterns

### 5.1 Add High-Level Validators

**File**: `fl_slam_poc/common/validation.py`

Add:
```python
def validate_imu_segment_data(
    t_i: float,
    t_j: float,
    accel: np.ndarray,
    gyro: np.ndarray,
    stamps: np.ndarray,
    covariance: np.ndarray
) -> None:
    """Validate complete IMU segment (Contract B)."""
    validate_imu_factor(
        delta_p=np.zeros(3),  # Not yet computed
        delta_v=np.zeros(3),
        delta_theta=np.zeros(3),
        covariance=covariance,
        timestamp_start=t_i,
        timestamp_end=t_j
    )

    # Additional raw measurement validation
    if not np.all(np.isfinite(accel)):
        raise ContractViolation("accel: Contains inf/nan")
    if not np.all(np.isfinite(gyro)):
        raise ContractViolation("gyro: Contains inf/nan")
    if not np.all(np.isfinite(stamps)):
        raise ContractViolation("stamps: Contains inf/nan")
```

---

## Summary of Savings

| Phase | Location | Lines Saved | Method |
|-------|----------|-------------|--------|
| 2 | Frontend validation | ~100 | Decorators |
| 3 | Frontend ROS params | ~104 | Pydantic codegen |
| 4 | RGB-D evidence | ~110 | Extract to module |
| **TOTAL** | | **~314 lines** | |

**Result**: Frontend node: 1,412 → ~1,098 lines (~22% reduction)

---

## Implementation Order

1. ✅ Phase 1: Manual dead code detection
2. **Phase 2**: Create validation decorators (NEXT)
3. **Phase 3**: Pydantic parameter codegen
4. **Phase 4**: Extract RGB-D publishing
5. **Phase 5**: Consolidate validation helpers

---

## Additional Cleanup Opportunities

### 6.1 Message Creation Boilerplate

**Problem**: ROS message creation is verbose

**Example** (frontend_node.py:1086-1111):
```python
loop = LoopFactor()
loop.header.stamp = stamp
loop.header.frame_id = self.odom_frame
loop.anchor_id = int(anchor_id)
# ... 20 more lines
```

**Solution**: Create message builder helpers

**File**: `fl_slam_poc/common/message_builders.py` (NEW)

```python
def create_loop_factor_msg(
    anchor_id: int,
    rel_pose: np.ndarray,
    covariance: np.ndarray,
    weight: float,
    stamp,
    frame_id: str,
    **kwargs
) -> LoopFactor:
    """Create LoopFactor message from data."""
    from fl_slam_poc.common.geometry.se3_numpy import rotvec_to_rotmat, rotmat_to_quat

    msg = LoopFactor()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.anchor_id = int(anchor_id)
    msg.weight = float(weight)

    # Position
    msg.rel_pose.position.x = float(rel_pose[0])
    msg.rel_pose.position.y = float(rel_pose[1])
    msg.rel_pose.position.z = float(rel_pose[2])

    # Orientation (rotvec → rotmat → quat)
    R = rotvec_to_rotmat(rel_pose[3:6])
    qx, qy, qz, qw = rotmat_to_quat(R)
    msg.rel_pose.orientation.x = qx
    msg.rel_pose.orientation.y = qy
    msg.rel_pose.orientation.z = qz
    msg.rel_pose.orientation.w = qw

    # Covariance
    msg.covariance = covariance.reshape(-1).tolist()

    # Optional fields
    for key, value in kwargs.items():
        setattr(msg, key, value)

    return msg
```

**Usage**:
```python
loop = create_loop_factor_msg(
    anchor_id, rel_pose, cov_transported, final_weight,
    stamp, self.odom_frame,
    approximation_triggers=["Linearization"],
    solver_name="ICP",
    # ... other optional fields
)
self.pub_loop.publish(loop)
```

**Savings**: ~20 lines per message creation site

---

## Long-Term: Functional Core, Imperative Shell

**Goal**: Separate pure functions (testable) from side effects (ROS/IO)

**Pattern**:
```
frontend_node.py (Shell: ROS callbacks, publish)
   ↓ calls
frontend/core.py (Core: pure data transformation)
   ↓ calls
common/geometry/*.py (Core: pure math)
```

**Benefits**:
- Pure functions are easily testable (no mocking)
- Side effects are isolated to shell layer
- Core logic can be reused outside ROS

---

## Tooling Recommendations

1. **ruff**: Fast Python linter (replaces flake8, isort, black)
   ```bash
   pip install ruff
   ruff check fl_ws/src/fl_slam_poc --fix
   ruff format fl_ws/src/fl_slam_poc
   ```

2. **mypy**: Static type checking (leverage Pydantic models)
   ```bash
   pip install mypy
   mypy fl_ws/src/fl_slam_poc/fl_slam_poc
   ```

3. **pytest-cov**: Test coverage to find dead code
   ```bash
   pytest --cov=fl_slam_poc test/
   ```

---

## Success Metrics

- ✅ Frontend node: <1,200 lines
- ✅ No duplicate validation logic (DRY)
- ✅ Pydantic models drive ROS parameters
- ✅ All validation uses decorators or helpers
- ✅ Message creation uses builder functions
- ✅ Dead code removed (coverage >80%)
