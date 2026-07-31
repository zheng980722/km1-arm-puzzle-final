#!/usr/bin/env python3
"""Km1 Arm Serial Driver - ROS2 Node
Keeps serial port open for lifetime. Never closes (prevents DTR reset loop).
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, String, Bool
import serial
import json
import re
import time
import threading


SERVO_FRAME_RE = re.compile(
    r'#(?P<id>\d{3})P(?P<pwm>\d{4})T(?P<time_ms>\d{4})!'
)


def external_wire_frame(payload):
    """Encode an internal KM1 command for the external USB UART parser.

    OpenMV-side examples omit the leading ``$`` because they use the
    controller's internal path.  The Orin talks through the external USB
    serialEvent parser, which requires ``$`` to reset its receive buffer and
    requires ``G0000`` inside a synchronous multi-servo frame.
    """

    frame = str(payload).strip()
    if not frame or frame.startswith('$'):
        return frame
    if frame.startswith('{#'):
        return '${G0000' + frame[1:]
    if frame.startswith('{G'):
        return '$' + frame
    if frame.startswith('#'):
        return '$' + frame
    return frame


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

        # 这些数值只是最后一次发送给舵机的命令，不是编码器反馈。
        # 保留 current_pwm 名称以兼容原有代码，并另行发布语义
        # 准确的 /km1/commanded_pwm 与 /km1/driver_status。
        self.current_pwm = [1500] * 6
        self.command_known = [False] * 6
        self.tx_sequence = 0
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

        # 创建发布者。/joint_states 是原项目兼容话题，其内容也只是
        # 命令值；新代码应订阅 /commanded_pwm 和 /driver_status。
        self.state_pub = self.create_publisher(Int32MultiArray, '/km1/joint_states', 10)
        self.commanded_pub = self.create_publisher(
            Int32MultiArray, '/km1/commanded_pwm', 10)
        self.status_pub = self.create_publisher(
            String, '/km1/driver_status', 20)
        self.serial_rx_pub = self.create_publisher(
            String, '/km1/serial_rx', 20)

        # 创建定时器：以10Hz频率发布关节状态
        self.create_timer(0.1, self.publish_state)
        # 只在有字节可读时才读取，不阻塞 ROS 回调。
        self.create_timer(0.02, self.poll_serial_rx)

    def publish_driver_status(self, payload):
        """发布可机器解析的驱动状态。"""
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def update_commanded_pwm(self, commands):
        """根据已发送的 (id, pwm, time_ms) 列表更新命令状态。"""
        for servo_id, pwm, _ in commands:
            if 0 <= servo_id <= 5:
                self.current_pwm[servo_id] = int(pwm)
                self.command_known[servo_id] = True

    def write_serial(self, payload, commands=None, source='unknown'):
        """串行写入完整帧，并记录实际写入字节数。"""
        if not self.port_open:
            self.publish_driver_status({
                'event': 'tx_rejected',
                'reason': 'serial_port_closed',
                'source': source,
            })
            return False

        if isinstance(payload, str):
            requested_frame = payload
            printable = external_wire_frame(payload)
            encoded = printable.encode('ascii')
        else:
            encoded = bytes(payload)
            printable = encoded.decode('ascii', errors='replace')
            requested_frame = printable

        try:
            with self.lock:
                written = self.ser.write(encoded)
                self.ser.flush()
        except Exception as exc:
            self.get_logger().error(f'Serial write failed: {exc}')
            self.publish_driver_status({
                'event': 'tx_error',
                'source': source,
                'error': str(exc),
                'frame': printable,
            })
            return False

        command_list = list(commands or [])
        if written == len(encoded):
            self.update_commanded_pwm(command_list)
        self.tx_sequence += 1
        status = {
            'event': 'tx',
            'sequence': self.tx_sequence,
            'source': source,
            'frame': printable,
            'requested_frame': requested_frame,
            'bytes_requested': len(encoded),
            'bytes_written': int(written),
            'write_complete': written == len(encoded),
            'commands': [
                {
                    'servo_id': int(servo_id),
                    'pwm_us': int(pwm),
                    'time_ms': int(time_ms),
                }
                for servo_id, pwm, time_ms in command_list
            ],
            'commanded_pwm': list(self.current_pwm),
            'known': list(self.command_known),
            'monotonic_s': round(time.monotonic(), 6),
        }
        self.publish_driver_status(status)
        magnet_commands = [
            (servo_id, pwm, time_ms)
            for servo_id, pwm, time_ms in command_list
            if servo_id == 5
        ]
        if magnet_commands:
            self.get_logger().info(
                f'MAGNET_TX sequence={self.tx_sequence} '
                f'commands={magnet_commands} wire={printable}'
            )
        if written != len(encoded):
            self.get_logger().error(
                f'Partial serial write: {written}/{len(encoded)} bytes')
            return False
        return True

    def poll_serial_rx(self):
        """发布 Arduino 回显/诊断字节，不将其冒充为关节反馈。"""
        if not self.port_open or not self.ser:
            return
        try:
            waiting = self.ser.in_waiting
            if waiting <= 0:
                return
            with self.lock:
                data = self.ser.read(waiting)
        except Exception as exc:
            self.get_logger().error(f'Serial read failed: {exc}')
            self.publish_driver_status({
                'event': 'rx_error',
                'error': str(exc),
            })
            return
        if not data:
            return
        msg = String()
        msg.data = data.decode('utf-8', errors='replace')
        self.serial_rx_pub.publish(msg)

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
        commands = []
        for match in SERVO_FRAME_RE.finditer(msg.data):
            commands.append((
                int(match.group('id')),
                int(match.group('pwm')),
                int(match.group('time_ms')),
            ))
        self.write_serial(msg.data, commands, source='raw_command')

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
        self.write_serial(
            cmd,
            [(servo_id, pwm, time_ms)],
            source='joint_command_single',
        )

    def move_multi(self, data):
        """发送多舵机控制命令
        
        格式: {#id0P...!#id1P...!...}
        
        Args:
            data: Int32MultiArray，格式为[id0, pwm0, t0, id1, pwm1, t1, ...]
        """
        frame = '{'
        commands = []
        for i in range(0, len(data), 3):
            sid = data[i]
            if sid < 0 or sid > 5:
                self.get_logger().warning(
                    f'Ignoring invalid servo id {sid} in multi command')
                continue
            pwm = max(500, min(2500, data[i+1]))
            t = max(0, min(9999, data[i+2]))
            frame += f'#{sid:03d}P{pwm:04d}T{t:04d}!'
            commands.append((sid, pwm, t))
        frame += '}'
        if commands:
            self.write_serial(
                frame,
                commands,
                source='joint_command_multi',
            )

    def publish_state(self):
        """发布当前关节状态
        
        以10Hz频率发布所有舵机的当前PWM值到/km1/joint_states话题。
        """
        msg = Int32MultiArray()
        msg.data = list(self.current_pwm)
        self.state_pub.publish(msg)
        self.commanded_pub.publish(msg)

    def release_all(self):
        """释放所有舵机的扭矩
        
        发送PULK命令释放所有舵机，使舵机进入自由状态。
        """
        if not self.port_open:
            return
        for i in range(6):
            self.write_serial(
                f'#{i:03d}PULK!',
                source='release_all',
            )
            time.sleep(0.05)
        self.get_logger().info('All servos released')

    def stop_all(self):
        """停止所有舵机运动
        
        发送$DST!命令立即停止所有舵机的运动。
        """
        if not self.port_open:
            return
        self.write_serial(b'$DST!', source='stop_all')
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
