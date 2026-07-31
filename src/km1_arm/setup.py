from glob import glob

from setuptools import setup
package_name = 'km1_arm'
setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.json')),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'serial_driver = km1_arm.serial_driver:main',
            'keyboard_teleop = km1_arm.keyboard_teleop:main',
            'arm_controller = km1_arm.arm_controller:main',
            'control_test = km1_arm.control_test:main',
            'yaw_calibration = km1_arm.yaw_calibration:main',
            'vision_bridge = km1_arm.vision_bridge:main',
        ],
    },
)
