"""
EXPERIMENTAL: Simulated semantic category publisher for Dirichlet testing.

This module is EXPERIMENTAL and not part of the main FL-SLAM pipeline.
It publishes simulated Dirichlet mixture distributions for testing
the experimental dirichlet_backend_node.

Status: Experimental - kept for reference and future development.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class SimSemantics(Node):
    def __init__(self):
        super().__init__("sim_semantics")
        self.pub = self.create_publisher(Float32MultiArray, "/sim/dirichlet_mixture", 10)
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.step)

        self.K = 5
        self.t = 0.0

        self.a1 = np.ones(self.K) * 2.0
        self.a2 = np.ones(self.K) * 2.0

    def step(self):
        self.t += self.dt

        k1 = int((self.t * 0.2) % self.K)
        k2 = int((self.t * 0.2 + 2.0) % self.K)

        self.a1 = np.ones(self.K) * 1.5
        self.a2 = np.ones(self.K) * 1.5
        self.a1[k1] = 15.0
        self.a2[k2] = 15.0

        w1 = 0.5 + 0.45 * np.sin(0.2 * self.t)
        w2 = 1.0 - w1

        msg = Float32MultiArray()
        msg.data = [float(w1), float(w2)] + self.a1.astype(float).tolist() + self.a2.astype(float).tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SimSemantics()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
