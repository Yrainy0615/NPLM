import numpy as np
import xml.etree.ElementTree as ET

def read_camera_xml(camera_file):
    tree = ET.parse(camera_file)
    root = tree.getroot()
    sensors = root.find('.//sensors')
    for sensor in sensors.findall('sensor'):
        calibration = sensor.find('calibration')
        resolution = calibration.find('resolution')
        w, h = int(resolution.get('width')), int(resolution.get('height'))
        f = float(calibration.find('f').text)
        cx = float(calibration.find('cx').text)
        cy = float(calibration.find('cy').text)
        k1 = float(calibration.find('k1').text)
        k2 = float(calibration.find('k2').text)
        k3 = float(calibration.find('k3').text)
        p1 = float(calibration.find('p1').text)
        p2 = float(calibration.find('p2').text)
        K = np.array([[f, 0, w/2 - abs(cx)], [0, f, h/2+cy], [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, p1, p2, k3])

    # 读取相机的外参
    camera_poses = []
    cameras = root.find('.//cameras')
    for camera in cameras.findall('camera'):
        label = camera.get('label')
        transform_text = camera.find('transform').text
        transform = np.fromstring(transform_text, sep=' ').reshape(4, 4)
        camera_pose = {
            'label': label,
            'transform': transform
        }
        camera_poses.append(camera_pose)
    return w, h, K, dist_coeffs, camera_poses

if __name__ == '__main__':
    mesh_path = 'test_result'
    camera_file = 'test_data/camera.xml'
    w, h, K, dist_coeffs, camera_poses = read_camera_xml(camera_file)
    pass
