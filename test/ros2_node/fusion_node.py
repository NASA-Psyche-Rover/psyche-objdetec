"""
ROS2 node template -- this is the part that needs the real rover to finish.

It assumes:
  - depthai-ros (or your own DepthAI pipeline) is publishing YOLOv6 detections
    as a vision_msgs/Detection2DArray on some topic (adjust DETECTIONS_TOPIC)
  - livox_ros_driver2 is publishing the point cloud as sensor_msgs/PointCloud2
    on /livox/lidar (the default topic name)

What it does each time a detection array + point cloud pair arrive close
enough together in time:
  1. pulls the raw points out of the PointCloud2 message
  2. runs the full fusion_lib pipeline (project -> ground removal ->
     matching -> clustering -> occupancy grid)
  3. publishes a nav_msgs/OccupancyGrid your path planner can consume, and
     logs each fused detection's distance

Things you'll need to fill in once you're actually running this:
  - CALIBRATION paths pointing at your real oak_intrinsics.yaml / extrinsics.yaml
  - DETECTIONS_TOPIC matching whatever your DepthAI pipeline actually publishes
  - grid extent / cell size tuned to your test course dimensions
"""
import numpy as np
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import OccupancyGrid

from fusion_lib.calibration import load_camera_intrinsics, load_extrinsics
from fusion_lib.projection import project_lidar_to_image
from fusion_lib.matching import Detection2D, match_points_to_detections
from fusion_lib.ground_removal import remove_ground_plane
from fusion_lib.clustering import cluster_obstacle_points
from fusion_lib.occupancy_grid import build_occupancy_grid

DETECTIONS_TOPIC = "/yolov6/detections"   # <-- update to match your real topic
POINTCLOUD_TOPIC = "/livox/lidar"
GRID_SIZE = 0.1
X_RANGE = (-3.0, 3.0)
Y_RANGE = (0.0, 5.0)


class FusionNode(Node):
    def __init__(self):
        super().__init__("psyche_fusion_node")

        self.intrinsics = load_camera_intrinsics("calibration/oak_intrinsics.yaml")
        self.extrinsics = load_extrinsics("calibration/extrinsics.yaml")

        det_sub = message_filters.Subscriber(self, Detection2DArray, DETECTIONS_TOPIC)
        cloud_sub = message_filters.Subscriber(self, PointCloud2, POINTCLOUD_TOPIC)
        # 0.1s slop covers typical camera/LiDAR timestamp drift without ROS2 hardware sync
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [det_sub, cloud_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.on_synced_msgs)

        self.grid_pub = self.create_publisher(OccupancyGrid, "/psyche/occupancy_grid", 10)
        self.get_logger().info("Fusion node up, waiting for synced detections + point cloud")

    def on_synced_msgs(self, detections_msg: Detection2DArray, cloud_msg: PointCloud2):
        points = np.array(list(point_cloud2.read_points(
            cloud_msg, field_names=("x", "y", "z"), skip_nans=True
        )))
        if points.size == 0:
            return

        detections = [
            Detection2D(
                x1=d.bbox.center.position.x - d.bbox.size_x / 2,
                y1=d.bbox.center.position.y - d.bbox.size_y / 2,
                x2=d.bbox.center.position.x + d.bbox.size_x / 2,
                y2=d.bbox.center.position.y + d.bbox.size_y / 2,
                cls=d.results[0].hypothesis.class_id if d.results else "unknown",
                confidence=d.results[0].hypothesis.score if d.results else 0.0,
            )
            for d in detections_msg.detections
        ]

        pixels, valid, _ = project_lidar_to_image(
            points, self.extrinsics.R, self.extrinsics.t,
            self.intrinsics.K, self.intrinsics.dist,
            self.intrinsics.width, self.intrinsics.height,
        )

        # Ground removal BEFORE matching -- see scripts/test_synthetic_fusion.py's
        # notes for why matching against raw points can pull in background points.
        obstacle_points, ground_mask = remove_ground_plane(points)
        obstacle_valid = valid[~ground_mask]
        obstacle_pixels = pixels[~ground_mask]

        fused = match_points_to_detections(obstacle_points, obstacle_pixels, obstacle_valid, detections)
        for f in fused:
            self.get_logger().info(
                f"[{f.detection.cls}] {f.distance_m:.2f}m "
                f"(std={f.distance_std:.2f}, n={f.num_points})"
            )

        labels = cluster_obstacle_points(obstacle_points, eps=0.2, min_samples=5)
        grid = build_occupancy_grid(
            obstacle_points[:, [0, 2]], grid_size=GRID_SIZE, x_range=X_RANGE, y_range=Y_RANGE
        )
        self.publish_grid(grid, cloud_msg.header)

    def publish_grid(self, grid: np.ndarray, header):
        msg = OccupancyGrid()
        msg.header = header
        msg.info.resolution = GRID_SIZE
        msg.info.width = grid.shape[1]
        msg.info.height = grid.shape[0]
        msg.info.origin.position.x = X_RANGE[0]
        msg.info.origin.position.y = Y_RANGE[0]
        msg.data = grid.flatten().astype(int).tolist()
        self.grid_pub.publish(msg)


def main():
    rclpy.init()
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
