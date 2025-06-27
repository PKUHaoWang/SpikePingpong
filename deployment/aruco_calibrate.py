import pyrealsense2 as rs
import numpy as np
import cv2

def get_device_ids():
    context = rs.context()
    devices = context.query_devices()
    device_ids = [device.get_info(rs.camera_info.serial_number) for device in devices]
    return device_ids

calibrate_cam = 0 
device_ids = get_device_ids()

# 初始化RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(device_ids[calibrate_cam])
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
pipeline.start(config)

# 获取相机内参
profile = pipeline.get_active_profile()
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
np.save(f'camera_depthscale_{str(device_ids[calibrate_cam])}.npy', depth_scale)

camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                         [0, intr.fy, intr.ppy],
                         [0, 0, 1]])
print("Camera Matrix:")
print(camera_matrix)
np.save(f'camera_intrin_{str(device_ids[calibrate_cam])}.npy', camera_matrix)
dist_coeffs = np.array([intr.coeffs])

# ArUco参数设置
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
parameters = cv2.aruco.DetectorParameters()

# 创建ArUco检测器
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# 标记板尺寸(米)
marker_length = 0.2

try:
    while True:
        # 获取图像帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
            
        color_image = np.asanyarray(color_frame.get_data())
        
        # 检测ArUco码
        corners, ids, rejected = detector.detectMarkers(color_image)
        
        if ids is not None:
            # 绘制检测到的标记
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            
            # 创建对象点
            objPoints = np.array([[-marker_length/2, marker_length/2, 0],
                                [marker_length/2, marker_length/2, 0],
                                [marker_length/2, -marker_length/2, 0],
                                [-marker_length/2, -marker_length/2, 0]], dtype=np.float32)
            
            # 使用solvePnP获取相机到ArUco码的变换
            success, rvec, tvec = cv2.solvePnP(objPoints, 
                                                corners[0].reshape(-1, 2), 
                                                camera_matrix, 
                                                dist_coeffs)
            
            # 将旋转向量转换为旋转矩阵
            R, _ = cv2.Rodrigues(rvec)
            
            aruco2cam = np.eye(4)
            aruco2cam[:3, :3] = R
            aruco2cam[:3, 3] = tvec.flatten()
            cam2aruco = np.linalg.inv(aruco2cam)
            print(cam2aruco)
            np.save(f'camera_extrin_{str(device_ids[calibrate_cam])}.npy', cam2aruco)

            # 绘制坐标轴
            cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
        
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
