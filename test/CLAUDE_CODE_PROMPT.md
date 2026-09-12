Paste this into Claude Code once you're at the rover with the OAK-D-Lite and
Livox Mid-360S actually connected to the Jetson Orin Nano. It gives Claude
Code the context it needs to pick up where this was left off.

---

I'm working on a LiDAR-camera fusion pipeline for an ASU EPICS NASA Psyche
rover project. The math and structure are already implemented and validated
against a synthetic scene -- see README.md for the full layout. What's NOT
done yet is anything that touches real hardware, because it was built
without the sensors physically present.

Hardware:
- Camera: OAK-D-Lite (fixed-focus), running YOLOv6 for object detection,
  connected via USB-C
- LiDAR: Livox Mid-360S, 360deg x 59deg FOV, connected via Ethernet to the
  Jetson Orin Nano, driven by livox_ros_driver2
- Compute: Nvidia Jetson Orin Nano

What's already implemented and tested (run
`python scripts/test_synthetic_fusion.py` to see it work against a fake
scene):
- fusion_lib/projection.py -- projects LiDAR points onto the camera image
  using cv2.projectPoints, given R, t, K, distortion
- fusion_lib/matching.py -- matches projected points to a YOLOv6 box,
  returns median distance + a std-based confidence signal
- fusion_lib/ground_removal.py -- RANSAC ground plane removal, numpy-only
- fusion_lib/clustering.py -- DBSCAN over non-ground obstacle points
- fusion_lib/bbox3d.py -- axis-aligned and PCA-oriented 3D box fitting per cluster
- fusion_lib/occupancy_grid.py -- points -> 2D occupancy grid

What needs to happen with real hardware:
1. Run scripts/get_oak_calibration.py on the Jetson to pull real camera
   intrinsics into calibration/oak_intrinsics.yaml
2. Run a real LiDAR<->camera extrinsic calibration (targetless:
   hku-mars/livox_camera_calib, or the official board-based Livox method)
   with the sensors in their FINAL mounted position, and save R/t into
   calibration/extrinsics.yaml
3. Finish ros2_node/fusion_node.py -- it's a template assuming YOLOv6
   detections publish as vision_msgs/Detection2DArray on some topic;
   confirm what our actual DepthAI pipeline publishes and update
   DETECTIONS_TOPIC accordingly
4. Tune eps in cluster_obstacle_points and distance_threshold in
   remove_ground_plane against our actual mock-Psyche terrain and obstacles
   (see the design doc's Test 1-4 movement/obstacle testing procedures)
5. Once calibration is real, re-run scripts/test_synthetic_fusion.py's
   approach but with a REAL captured frame + point cloud pair instead of the
   synthetic scene, to sanity-check the calibration before trusting it live
   (overlay_points_on_image in fusion_lib/projection.py is built for exactly
   this -- if the projected dots don't line up with a real object's edges in
   the image, the calibration needs redoing)

Please help me get this running end to end on the real rover, starting with
step 1.
