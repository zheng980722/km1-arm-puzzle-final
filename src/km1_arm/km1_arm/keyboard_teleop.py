#!/usr/bin/env python3
"""Km1 Keyboard Teleop - 电磁吸版
  q/a: Joint 0 (base)     +/-
  w/s: Joint 1 (shoulder) +/-
  e/d: Joint 2 (elbow)    +/-
  r/f: Joint 3 (wrist1)   +/-
  t/g: Joint 4 (wrist2)   +/-
  y:   Magnet ON           h: Magnet OFF
  0:   All center          space: Release all
  x:   Stop                1-5: Release joint N
  +/-: Step size           ,/.: Time slower/faster
  ESC: Quit
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, String
import sys, termios, tty, select

JOINT_NAMES = ['Base', 'Shoulder', 'Elbow', 'Wrist1', 'Wrist2', 'Magnet']
SAFE_RANGE = [
    [500, 2500], [800, 2200], [700, 2300],
    [700, 2250], [500, 2500], [1000, 2000],
]
MAGNET_ON = 1100
MAGNET_OFF = 1500


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('km1_keyboard_teleop')
        self.pub_cmd = self.create_publisher(Int32MultiArray, '/km1/joint_command', 10)
        self.pub_raw = self.create_publisher(String, '/km1/raw_command', 10)
        self.current_pwm = [1500] * 6
        self.step = 50
        self.time_ms = 500
        self.key_map = {
            'q': (0, +1), 'a': (0, -1),
            'w': (1, +1), 's': (1, -1),
            'e': (2, -1), 'd': (2, +1),
            'r': (3, +1), 'f': (3, -1),
            't': (4, +1), 'g': (4, -1),
        }
        self.print_help()

    def print_help(self):
        print()
        print('='*56)
        print('  Km1 Teleop (Electromagnet)')
        print('='*56)
        print('  q/a Base   w/s Shoulder   e/d Elbow')
        print('  r/f Wrist1 t/g Wrist2')
        print('  y = Magnet ON    h = Magnet OFF')
        print('  0 = All center   space = Release all')
        print('  x = Stop   1-5 = Release joint N')
        print('  +/- = Step   ,/. = Time')
        print('  ESC = Quit')
        print('='*56)

    def print_state(self):
        # External capture circuit: <1.2 ms = magnet ON, >1.4 ms = OFF.
        mag = 'ON ' if self.current_pwm[5] <= 1200 else 'OFF'
        line1 = f'  B:{self.current_pwm[0]:4d} S:{self.current_pwm[1]:4d} E:{self.current_pwm[2]:4d} W1:{self.current_pwm[3]:4d} W2:{self.current_pwm[4]:4d} MAG:{mag}'
        line2 = f'  Step={self.step} Time={self.time_ms}ms'
        print(f'\r{line1} | {line2}', end='', flush=True)

    def send_joint(self, jid, pwm, t):
        msg = Int32MultiArray()
        msg.data = [jid, pwm, t]
        self.pub_cmd.publish(msg)

    def send_raw(self, s):
        msg = String()
        msg.data = s
        self.pub_raw.publish(msg)

    def process_key(self, key):
        if key in ('', ''):  # ESC or Ctrl+C
            return False
        if key in self.key_map:
            jid, d = self.key_map[key]
            pwm = max(SAFE_RANGE[jid][0], min(SAFE_RANGE[jid][1], self.current_pwm[jid] + d * self.step))
            self.current_pwm[jid] = pwm
            self.send_joint(jid, pwm, self.time_ms)
            self.print_state()
        elif key == 'y':
            self.current_pwm[5] = MAGNET_ON
            self.send_raw(f'#005P{MAGNET_ON:04d}T0100!')
            self.print_state()
        elif key == 'h':
            self.current_pwm[5] = MAGNET_OFF
            self.send_raw(f'#005P{MAGNET_OFF:04d}T0100!')
            self.print_state()
        elif key == '0':
            self.current_pwm = [1500]*6
            msg = Int32MultiArray()
            msg.data = []
            for i in range(5):
                msg.data.extend([i, 1500, self.time_ms])
            self.pub_cmd.publish(msg)
            self.print_state()
        elif key == ' ':
            self.send_raw('#000PULK!#001PULK!#002PULK!#003PULK!#004PULK!#005PULK!')
            print('\n  >>> RELEASED <<<')
        elif key == 'x':
            self.send_raw('$DST!')
            print('\n  >>> STOP <<<')
        elif key in '12345':
            j = int(key) - 1
            self.send_raw(f'#{j:03d}PULK!')
            print(f'\n  >>> Released J{j} <<<')
        elif key in '+=':
            self.step = min(500, self.step + 25)
            self.print_state()
        elif key == '-':
            self.step = max(10, self.step - 25)
            self.print_state()
        elif key == ',':
            self.time_ms = min(3000, self.time_ms + 100)
            self.print_state()
        elif key == '.':
            self.time_ms = max(0, self.time_ms - 100)
            self.print_state()
        return True

    def run(self):
        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            self.print_state()
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    if not self.process_key(sys.stdin.read(1)):
                        break
                rclpy.spin_once(self, timeout_sec=0.01)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            print('\n  Exited.')


def main():
    rclpy.init()
    node = KeyboardTeleop()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
