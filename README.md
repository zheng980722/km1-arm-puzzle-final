# Km1 机械臂 ROS2 控制系统

## 概述

基于 ROS2 Humble 的 Km1 机械臂控制系统，末端执行器为**可旋转电磁铁**。
支持键盘遥操作和视觉自主拾放（拼图任务）。

- **硬件平台：** NVIDIA Orin NX + Arduino Uno 控制板 + CH340 USB 串口
- **视觉平台：** USB 摄像头顶视 + OpenCV 拼图检测
- **工作空间：** `~/WorkSpace/km1_arm_ws/`
- **当前台架：** 相机距纸面 600 mm；A4 横放，右半区取料、左半区拼装
- **串口所有权：** 仅常驻 `km1_serial_driver` 可以打开设备，禁止一次性脚本直开串口
- **安全状态：** 纸面接触 Z 和 XY 精度未标定，视觉输出与自动动作必须保持锁定

---

## 系统架构

```
[USB Camera] --> vision_bridge --> /km1/control_plan (JSON)
                                        |
                                        v
                                 arm_controller (IK + pick-place FSM)
                                        |
                                        v /km1/raw_command
                                 serial_driver --> CH340 --> Arduino
                                      |              |--> ID0~3 arm
                                      |              |--> ID4 rotating magnet tool
                                      |              `--> ID5 PWM capture --> magnet
                                        ^
                                 keyboard_teleop (manual control)
```

**ROS2 Topics:**

| Topic | Type | 方向 | 说明 |
|-------|------|------|------|
| `/km1/raw_command` | std_msgs/String | teleop/controller -> driver | 原始协议字符串直发 |
| `/km1/joint_command` | std_msgs/Int32MultiArray | teleop -> driver | [id, pwm, time_ms] |
| `/km1/joint_states` | std_msgs/Int32MultiArray | driver -> all | 当前6路PWM值 |
| `/km1/control_plan` | std_msgs/String | vision -> controller | JSON拾放计划 |
| `/km1/vision_trigger` | std_msgs/String | user -> vision | 触发一次识别 |
| `/km1/vision_status` | std_msgs/String | vision -> user | 识别门控、错误和诊断目录 |

---

## 目录结构

```
km1_arm_ws/
├── src/km1_arm/
│   └── km1_arm/
│       ├── serial_driver.py      # 串口驱动(保活+转发)
│       ├── keyboard_teleop.py    # 键盘遥操作
│       ├── arm_controller.py     # 自主拾放(IK+状态机)
│       └── vision_bridge.py      # 摄像头->control_plan
├── docs/
│   ├── protocol.md               # 串口协议完整文档
│   └── kinematics.py             # 逆运动学(Python3)
├── puzzle_vision/                # 拼图视觉
│   ├── puzzle_vision.py          # 核心管线(检测+求解)
│   ├── simple_detect.py          # 简洁块检测(HSV法)
│   ├── gui_app.py                # AR可视化GUI
│   └── assets/card_templates/    # 扑克牌纹理模板
├── driver/
│   ├── ch341.ko                  # 编译好的CH340驱动
│   └── ch341ser_linux/           # 驱动源码
└── README.md
```

---

## 安装与编译

### 首次配置

```bash
# 1. CH340 驱动
cd ~/WorkSpace/km1_arm_ws/driver/ch341ser_linux/driver
make
sudo insmod ch341.ko
# 已配置开机自动加载，重启后无需手动操作

# 2. 删除 brltty (防止抢占串口)
sudo apt purge -y brltty

# 3. Python 依赖
pip3 install pyserial

# 4. 编译 ROS2 包
cd ~/WorkSpace/km1_arm_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select km1_arm
```

### 每次使用

```bash
cd ~/WorkSpace/km1_arm_ws
source install/setup.bash
```

启动控制相关节点时，`km1_serial_driver`必须是唯一串口持有者，并在整个联调期间常驻。
其他节点只通过ROS话题与它通信；不得同时运行任何自行调用`pyserial`打开同一设备的脚本。

---

## 使用方法

### 模式 1: 键盘遥操作

```bash
# 终端1: 串口驱动 (启动后等5秒Arduino boot)
ros2 run km1_arm serial_driver

# 终端2: 键盘控制
ros2 run km1_arm keyboard_teleop
```

#### 键盘映射

| 按键 | 关节 | 方向 |
|------|------|------|
| q / a | 底座 (Joint 0) | 左 / 右 |
| w / s | 大臂 (Joint 1) | 上 / 下 |
| e / d | 小臂 (Joint 2) | 下 / 上 |
| r / f | 腕1 (Joint 3) | 上 / 下 |
| t / g | 腕2 (Joint 4) | 左 / 右 |
| **y** | 电磁吸 ON | 吸附 |
| **h** | 电磁吸 OFF | 释放 |
| 0 | 全部回中 | 1500 |
| 空格 | 全部释力 | PULK |
| x | 急停 | $DST! |
| 1-5 | 单关节释力 | — |
| +/- | 步进大小 | 默认50 |
| , / . | 速度慢/快 | 默认500ms |
| ESC | 退出 | — |

### 模式 2: 视觉自主拾放

```bash
# 终端1
ros2 run km1_arm serial_driver

# 终端2
ros2 run km1_arm arm_controller

# 终端3
export PUZZLE_VISION_PATH=~/WorkSpace/km1_arm_ws/puzzle_vision
ros2 run km1_arm vision_bridge --ros-args -p camera_index:=0

# 触发
ros2 topic pub --once /km1/vision_trigger std_msgs/msg/String "{data: 'go'}"
```

`vision_bridge`默认`enable_control_output=false`，`arm_controller`默认
`enable_automatic_motion=false`，并要求有效的`paper_surface_z_mm`。因此默认只会拍照、
求解和保存诊断，不会形成真实动作。当前接触Z和纸面XY精度尚未完成标定，这两把锁
必须继续保持关闭；本README不提供解锁步骤。

### 模式 3: 视觉 GUI 调试

```bash
cd ~/WorkSpace/km1_arm_ws/puzzle_vision
python3 gui_app.py
```

---

## 串口协议参考

### 物理层

| 参数 | 值 |
|------|------|
| 芯片 | CH340 (VID:1a86 PID:7523) |
| 设备 | /dev/ttyCH341USB0 |
| 波特率 | 115200 8N1 |

### 指令格式

```
单舵机:  #IDDPxxxxTxxxx!     ID=3位 PWM=4位 Time=4位ms
多舵机:  {#...!#...!#...!}   花括号包裹
停止:    $DST!
释力:    #IDDPULK!
```

### 舵机 ID 表

| ID | 关节 | 安全范围 | 备注 |
|----|------|---------|------|
| 0 | 底座 Yaw | 500-2500 | 360度旋转 |
| 1 | 大臂 | 800-2200 | — |
| 2 | 小臂 | 700-2300 | — |
| 3 | 腕1 | 700-2250 | — |
| 4 | 改装夹爪/电磁铁整体旋转 | 550-2450 | 控制碎片面内旋转 |
| 5 | 电磁铁状态信号 | ON=1100 OFF=1500 | 捕获电路按脉宽吸放 |

协议帧只应由受保护的控制节点生成，并经ROS话题交给常驻驱动。当前禁止使用一次性
发布命令或直开串口的方式继续实机试探。单步测试统一使用`control_test`的默认预览模式；
只有完成风险检查并由现场负责人明确确认后，才可按该程序自身的双重确认机制执行。

---

## 逆运动学

```
连杆: L0=100mm(底座高) L1=105mm(大臂) L2=88mm(小臂)
改装末端: 旋转电磁铁半径20mm、高20mm；腕轴到吸附面L3初值=80mm
角度->PWM: pwm = 1500 + 2000 * angle_deg / 270
输入: x,y(水平mm), z(高度mm), alpha(夹爪俯仰角 -25~-65度)
输出: 4路PWM (servo 0-3)
```

原夹爪`L3=155 mm`不适用于当前改装末端。`80 mm`仍是初值，最终值要与纸面接触Z
一起标定；在此之前不得把计算Z当作电磁铁吸附面的真实高度。

---

## 故障排除

| 问题 | 原因 | 解决 |
|------|------|------|
| 反复复位或启动动作 | 有程序绕过常驻驱动重复打开串口，触发DTR复位 | 停止该程序，只保留唯一`km1_serial_driver` |
| /dev/ttyCH341USB0 不存在 | 驱动未加载 | `sudo insmod ~/WorkSpace/km1_arm_ws/driver/ch341.ko` |
| 设备出现后消失 | brltty抢占 | `sudo apt purge -y brltty` |
| lsusb无1a86 | USB线/口问题 | 换口/换线，确认插Orin |
| 舵机发热 | 堵转(到极限还使劲) | 发PULK释力，或用安全范围 |
| 视觉检测不到纸 | 纸面占比过小/光照不均 | 核对600 mm固定高度、相机居中与均匀光；当前A4横放 |
| 视觉块轮廓锯齿 | 光照渐变 | 加均匀光源 |
| 横向台架被拒绝 | 视觉配置仍使用旧纵向上下分区 | 当前台架应为横向右取左放；控制输出继续保持锁定后再核对配置 |

---

## 手眼标定

当前横放A4的初始几何映射位于`km1_arm/control_geometry.py`：

\[
x_r=148.5-y_p,\qquad y_r=265-x_p
\]

纸面坐标定义为：`paper_x`从物理远边0 mm增加到近边210 mm，`paper_y`从物理右边
0 mm增加到左边297 mm。纸面中心`(105,148.5)`对应机械臂`(0,160)` mm。
旧`CALIB_OFFSET_X/Y/Z`只属于已废弃占位方案，不得重新用于当前台架。

该映射已完成中心和XY四方向的高位方向核验，但还没有完成独立点残差验收。后续应在
初始映射上拟合小幅仿射修正；纸面接触Z另行标定，不能用旧偏移值代替。

## 相机与深度

普通RGB相机即可完成当前平面拼图，不要求深度信息。相机安装在A4中心正上方，
光轴尽量垂直纸面；当前光心距纸面实测为600 mm，以1920×1080 MJPG输入时A4短边
约314 px。该采样密度可用于当前单帧规划，但余量偏紧，不能用数字放大冒充真实细节。
机械臂停在纸外安全位后再采图。深度相机只在碎片翘曲、叠放或需要在线估计Z高度时有价值。

## 2026-07-31阶段性实测

| 项目 | 已核对结果 |
|---|---|
| 相机与纸面 | 光心距纸面600 mm；A4横放，近侧长边中点与底座前沿中点重合 |
| 功能分区 | 视觉右半区取料、左半区拼装 |
| 中心与XY方向 | 中心高位及左右、前后四方向与初始映射一致；绝对XY精度待标定 |
| Z分级 | z=180、150、120、100、80、60、40 mm完成非接触检查；接触Z未确定 |
| ID4 | 完成+30°、−30°和回中检查 |
| ID5 | 完成1100 μs吸合、1500 μs释放信号检查 |
| 最终安全姿态 | 中心z=150 mm，ID4=1500 μs，ID5=1500 μs |

上述结果只证明方向、分级运动和控制信号已连通，不能写成接触Z、XY精度、真实吸放
或完整自动拼图已经验收。自动控制必须保持锁定。
