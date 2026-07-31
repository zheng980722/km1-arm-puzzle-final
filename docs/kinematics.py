"""Km1 机械臂逆运动学 (从 OpenMV 源码移植到 Python3)
坐标约定: x,y 水平面(mm), z 离地高度(mm), Alpha 夹爪与水平面夹角(deg)
Alpha 范围: -25 ~ -65 度较好
"""
import math


class Km1Kinematics:
    # 连杆长度 (mm, 已除10)
    L0 = 100.0   # 底座到肩关节高度
    L1 = 105.0   # 大臂
    L2 = 88.0    # 小臂
    ORIGINAL_GRIPPER_L3 = 155.0
    MODIFIED_MAGNET_L3 = 50.0
    # Real bench calibration of servo 0 on 2026-07-31.  Holding the arm at
    # r=160 mm and z=180 mm gave approximately:
    #   1500 us -> 0 deg, 1300 us -> 28 deg, 1250 us -> 35.5 deg.
    # The original port effectively used 11.11 us/deg and therefore
    # over-rotated large off-axis targets.  Keep this calibration separate
    # from the generic 270-degree joint conversion used by servos 1..3.
    BASE_CENTER_PWM = 1500
    BASE_PWM_PER_DEG = 7.0
    VERTICAL_TOOL_ALPHA_DEG = -90.0

    def __init__(self, tool_length_mm: float = MODIFIED_MAGNET_L3):
        # 改装后的旋转电磁铁：腕轴到吸附面的初始实测估计。
        # 最终值需通过分级下降试验标定，不再沿用原夹爪 155 mm。
        self.L3 = float(tool_length_mm)

    def angle_to_pwm(self, angle_deg: float, invert: bool = False) -> int:
        pwm = int(1500 + 2000.0 * angle_deg / 270.0)
        if invert:
            pwm = 3000 - pwm
        return max(500, min(2500, pwm))

    def solve(self, x: float, y: float, z: float, alpha_deg: float):
        """逆运动学求解. 返回 (pwm0,pwm1,pwm2,pwm3) 或 None(不可达)"""
        # 放大10倍计算(保持与原代码一致)
        x10 = x * 10
        y10 = y * 10
        z10 = z * 10
        l0, l1, l2, l3 = self.L0*10, self.L1*10, self.L2*10, self.L3*10
        pi = math.pi
        alpha = alpha_deg

        # 底座旋转角 (servo 0)
        theta6_deg = math.degrees(math.atan2(x10, y10))

        # 平面距离
        r = math.sqrt(x10*x10 + y10*y10)
        r = r - l3 * math.cos(alpha * pi / 180.0)
        h = z10 - l0 - l3 * math.sin(alpha * pi / 180.0)

        if h < -l0:
            return None
        if math.sqrt(r*r + h*h) > (l1 + l2):
            return None

        # 肩关节角 (servo 1)
        ccc = math.acos(r / math.sqrt(r*r + h*h))
        bbb = (r*r + h*h + l1*l1 - l2*l2) / (2 * l1 * math.sqrt(r*r + h*h))
        if bbb > 1 or bbb < -1:
            return None
        zf_flag = -1 if h < 0 else 1
        theta5 = ccc * zf_flag + math.acos(bbb)
        theta5_deg = theta5 * 180.0 / pi
        if theta5_deg > 180 or theta5_deg < 0:
            return None

        # 肘关节角 (servo 2)
        aaa = -(r*r + h*h - l1*l1 - l2*l2) / (2 * l1 * l2)
        if aaa > 1 or aaa < -1:
            return None
        theta4 = 180.0 - math.acos(aaa) * 180.0 / pi
        if theta4 > 135 or theta4 < -135:
            return None

        # 腕关节角 (servo 3)
        theta3 = alpha - theta5_deg + theta4
        # The installed TBD-K20 wrist servo is a 270-degree unit.  The
        # electromagnet must point vertically down during pickup and release,
        # which can require slightly more than 90 degrees at the wrist.
        if theta3 > 135 or theta3 < -135:
            return None

        # 转 PWM
        pwm0 = int(round(
            self.BASE_CENTER_PWM - theta6_deg * self.BASE_PWM_PER_DEG
        ))
        pwm0 = max(500, min(2500, pwm0))
        pwm1 = self.angle_to_pwm(theta5_deg - 90)  # 肩
        pwm2 = self.angle_to_pwm(theta4)   # 肘
        pwm3 = self.angle_to_pwm(theta3)   # 腕

        return (pwm0, pwm1, pwm2, pwm3)

    def solve_vertical(self, x: float, y: float, z: float):
        """Solve with the electromagnet axis constrained vertically down."""

        return self.solve(x, y, z, self.VERTICAL_TOOL_ALPHA_DEG)

    def find_best_alpha(self, x: float, y: float, z: float):
        """遍历 alpha 找最佳角度(与水平面最大夹角)"""
        best = None
        best_alpha = None
        for alpha in range(-25, -66, -1):
            result = self.solve(x, y, z, alpha)
            if result is not None:
                best = result
                best_alpha = alpha
        return best, best_alpha

    def build_frame(self, pwms, time_ms=1000):
        """构建多舵机帧字符串"""
        if pwms is None:
            return None
        parts = []
        for i, pwm in enumerate(pwms):
            parts.append(f'#{i:03d}P{pwm:04d}T{time_ms:04d}!')
        return '{' + ''.join(parts) + '}'
