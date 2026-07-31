#!/usr/bin/env python3
"""Km1 Arm Serial Driver - ROS2 Node
Keeps serial port open for lifetime. Never closes (prevents DTR reset loop).
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, String, Bool
import serial
import time
import threading


class Km1SerialDriver(Node):
    """KM1机械臂串口驱动节点
    
    负责通过串口与Arduino通信，控制6个舵机。
    保持串口在整个节点生命周期内打开，避免DTR复位循环。
    """
    
    def __init__(self):
        """初始化串口驱动节点"""
        super().__init__('km1_serial_driver')

        # 声明ROS2参数
        self.declare_parameter('port', '/dev/ttyCH341USB0')  # 串口设备路径
        self.declare_parameter('baud_rate', 115200)          # 波特率
        self.declare_parameter('boot_wait', 5.0)             # Arduino启动等待时间(秒)
        self.declare_parameter('default_time_ms', 500)       # 默认舵机运动时间(毫秒)
        self.declare_parameter('release_on_shutdown', False)

        # 获取参数值
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud_rate').value
        boot_wait = self.get_parameter('boot_wait').value
        self.default_time = self.get_parameter('default_time_ms').value
        self.release_on_shutdown = bool(
            self.get_parameter('release_on_shutdown').value
        )

        # 状态变量：每个舵机的当前PWM值（0-5号舵机）
        self.current_pwm = [1500] * 6  # 初始值为1500（中位）
        self.servo_moving = [False] * 6  # 舵机运动状态标志
        self.lock = threading.Lock()  # 线程锁，保护串口写入

        # 打开串口
        self.get_logger().info(f'Opening {port} at {baud} baud...')
        try:
            # dsrdtr=False, rtscts=False 避免硬件流控制导致的复位
            self.ser = serial.Serial(port, baud, timeout=0.1, dsrdtr=False, rtscts=False)
            self.get_logger().info(f'Port opened. Waiting {boot_wait}s for Arduino boot...')
            time.sleep(boot_wait)
            # 清空Arduino启动时的启动消息
            drain = self.ser.read(self.ser.in_waiting or 1000)
            if drain:
                self.get_logger().info(f'Drained {len(drain)} boot bytes')
            self.get_logger().info('Arduino ready. Driver active.')
            self.port_open = True
        except Exception as e:
            self.get_logger().error(f'Failed to open port: {e}')
            self.port_open = False
            self.ser = None

        # 创建订阅者
        # 关节命令订阅：接收舵机控制指令
        self.create_subscription(
            Int32MultiArray, '/km1/joint_command',
            self.cmd_callback, 10)
        # 原始命令订阅：直接透传协议字符串
        self.create_subscription(
            String, '/km1/raw_command',
            self.raw_callback, 10)

        # 创建发布者
        # 发布关节状态：当前所有舵机的PWM值
        self.state_pub = self.create_publisher(Int32MultiArray, '/km1/joint_states', 10)

        # 创建定时器：以10Hz频率发布关节状态
        self.create_timer(0.1, self.publish_state)

    def cmd_callback(self, msg):
        """关节命令回调函数
        
        支持三种格式：
        - 单舵机带时间: [id, pwm, time_ms]
        - 单舵机默认时间: [id, pwm]
        - 多舵机: [id0, pwm0, t0, id1, pwm1, t1, ...]
        
        Args:
            msg: Int32MultiArray类型的消息
        """
        if not self.port_open:
            return
        data = msg.data
        if len(data) == 3:
            # 单舵机带时间: [id, pwm, time_ms]
            sid, pwm, t = data[0], data[1], data[2]
            self.move_servo(sid, pwm, t)
        elif len(data) == 2:
            # 单舵机简写: [id, pwm] 使用默认时间
            self.move_servo(data[0], data[1], self.default_time)
        elif len(data) >= 4 and len(data) % 3 == 0:
            # 多舵机: [id0, pwm0, t0, id1, pwm1, t1, ...]
            self.move_multi(data)

    def raw_callback(self, msg):
        """原始命令回调函数
        
        直接透传协议字符串到串口，用于发送自定义命令。
        
        Args:
            msg: String类型的消息，包含要发送的协议字符串
        """
        if not self.port_open:
            return
        with self.lock:
            self.ser.write(msg.data.encode('ascii'))

    def move_servo(self, servo_id, pwm, time_ms):
        """发送单舵机控制命令
        
        Args:
            servo_id: 舵机ID (0-5)
            pwm: PWM脉宽值 (500-2500)
            time_ms: 运动时间 (0-9999毫秒)
        """
        if servo_id < 0 or servo_id > 5:
            return
        # 限制PWM范围在500-2500之间
        pwm = max(500, min(2500, pwm))
        # 限制时间范围在0-9999毫秒之间
        time_ms = max(0, min(9999, time_ms))
        # 构造协议命令: #XXXPXXXXTXXXX!
        cmd = f'#{servo_id:03d}P{pwm:04d}T{time_ms:04d}!'
        with self.lock:
            self.ser.write(cmd.encode('ascii'))
        self.current_pwm[servo_id] = pwm

    def move_multi(self, data):
        """发送多舵机控制命令
        
        格式: {#id0P...!#id1P...!...}
        
        Args:
            data: Int32MultiArray，格式为[id0, pwm0, t0, id1, pwm1, t1, ...]
        """
        frame = '{'
        for i in range(0, len(data), 3):
            sid = data[i]
            pwm = max(500, min(2500, data[i+1]))
            t = max(0, min(9999, data[i+2]))
            frame += f'#{sid:03d}P{pwm:04d}T{t:04d}!'
            if sid <= 5:
                self.current_pwm[sid] = pwm
        frame += '}'
        with self.lock:
            self.ser.write(frame.encode('ascii'))

    def publish_state(self):
        """发布当前关节状态
        
        以10Hz频率发布所有舵机的当前PWM值到/km1/joint_states话题。
        """
        msg = Int32MultiArray()
        msg.data = list(self.current_pwm)
        self.state_pub.publish(msg)

    def release_all(self):
        """释放所有舵机的扭矩
        
        发送PULK命令释放所有舵机，使舵机进入自由状态。
        """
        if not self.port_open:
            return
        with self.lock:
            for i in range(6):
                self.ser.write(f'#{i:03d}PULK!'.encode('ascii'))
                time.sleep(0.05)
        self.get_logger().info('All servos released')

    def stop_all(self):
        """停止所有舵机运动
        
        发送$DST!命令立即停止所有舵机的运动。
        """
        if not self.port_open:
            return
        with self.lock:
            self.ser.write(b'$DST!')
        self.get_logger().info('All servos stopped')

    def destroy_node(self):
        """节点销毁时的清理工作
        
        默认保持舵机力矩，避免退出launch时机械臂突然下坠。
        """
        if self.ser and self.port_open and self.release_on_shutdown:
            self.release_all()
        super().destroy_node()


def main():
    """主函数：初始化并运行串口驱动节点"""
    rclpy.init()
    node = Km1SerialDriver()
    try:
        rclpy.spin(node)  # 进入ROS2事件循环
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()  # 销毁节点
        rclpy.shutdown()  # 关闭ROS2


if __name__ == '__main__':
    main()
