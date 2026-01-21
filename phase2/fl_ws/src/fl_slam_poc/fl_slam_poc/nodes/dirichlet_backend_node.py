"""
EXPERIMENTAL: Dirichlet semantic SLAM backend node.

This module is EXPERIMENTAL and not part of the main FL-SLAM pipeline.
It demonstrates Dirichlet-based semantic category fusion but is not
integrated with the primary frontend/backend architecture.

Status: Experimental - kept for reference and future development.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

from fl_slam_poc.operators.dirichlet_geom import iproject_dirichlet_from_mixture


class DirichletBackend(Node):
    def __init__(self):
        super().__init__("dirichlet_backend")

        self.declare_parameter("use_third_order", True)
        self.declare_parameter("allow_ablation", False)

        use_third = bool(self.get_parameter("use_third_order").value)
        allow_ablation = bool(self.get_parameter("allow_ablation").value)
        if not use_third and not allow_ablation:
            self.get_logger().warning(
                "Frobenius correction is mandatory for approximation; "
                "set allow_ablation:=true to disable for baseline runs."
            )
            use_third = True

        self.use_third = use_third
        self.allow_ablation = allow_ablation

        self.sub = self.create_subscription(
            Float32MultiArray, "/sim/dirichlet_mixture", self.on_mix, 10
        )
        self.pub = self.create_publisher(Float32MultiArray, "/cdwm/dirichlet_alpha", 10)
        self.dbg = self.create_publisher(String, "/cdwm/dirichlet_debug", 10)
        self.pub_report = self.create_publisher(String, "/cdwm/op_report", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/cdwm/dirichlet_markers", 10)

        self.alpha = None
        self.ablation_logged = False
        self.frame_id = "map"

    def on_mix(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=float)
        if data.size < 4:
            self.get_logger().warning("Dirichlet mixture message too short.")
            return

        w1, w2 = data[0], data[1]
        rest = data[2:]
        if rest.size % 2 != 0:
            self.get_logger().warning("Dirichlet mixture payload has odd length.")
            return

        K = rest.size // 2
        a1 = rest[:K]
        a2 = rest[K:]

        alphas = np.stack([a1, a2], axis=0)
        weights = np.array([w1, w2], dtype=float)

        if self.alpha is None:
            self.alpha = np.mean(alphas, axis=0)

        alpha_new, report = iproject_dirichlet_from_mixture(
            alphas=alphas,
            weights=weights,
            alpha_init=self.alpha,
            max_iter=4,
            tol=1e-10,
            use_third_order=self.use_third,
        )
        self.alpha = alpha_new

        out = Float32MultiArray()
        out.data = self.alpha.astype(float).tolist()
        self.pub.publish(out)

        if self.allow_ablation and not self.use_third and not self.ablation_logged:
            self.get_logger().warning("Running ablation: Frobenius correction disabled.")
            self.ablation_logged = True

        try:
            report.validate(allow_ablation=self.allow_ablation)
        except ValueError as exc:
            self.get_logger().error(f"OpReport validation failed: {exc}")
            raise

        report_msg = String()
        report_msg.data = report.to_json()
        self.pub_report.publish(report_msg)

        iters = report.metrics.get("iters", 0)
        final_norm = report.metrics.get("final_norm", 0.0)

        approx_str = ",".join(report.approximation_triggers) or "none"

        s = String()
        s.data = (
            f"approx={approx_str} solver={report.solver_used} "
            f"third_order={report.frobenius_applied} "
            f"iters={iters} resid={final_norm:.3e} "
            f"domain_proj={report.domain_projection}"
        )
        self.dbg.publish(s)

        self.publish_markers()

    def publish_markers(self):
        if self.alpha is None:
            return

        alpha = np.asarray(self.alpha, dtype=float).reshape(-1)
        total = float(np.sum(alpha))
        if total <= 0.0:
            return
        mean = alpha / total

        ma = MarkerArray()
        spacing = 0.3
        width = 0.2
        for idx, val in enumerate(mean.tolist()):
            m = Marker()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self.frame_id
            m.ns = "dirichlet"
            m.id = idx
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(idx) * spacing
            m.pose.position.y = 0.0
            m.pose.position.z = float(val) * 0.5
            m.scale.x = width
            m.scale.y = width
            m.scale.z = max(float(val), 1e-3)
            m.color.a = 0.8
            m.color.r = 0.2
            m.color.g = 0.6
            m.color.b = 0.9
            ma.markers.append(m)

        self.pub_markers.publish(ma)


def main():
    rclpy.init()
    node = DirichletBackend()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
