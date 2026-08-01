# KM1 视觉拼图机械臂（最终比赛版）

本仓库是已在 Orin NX 与 KM1 实机上完成闭环验证的比赛版本。系统使用普通 RGB 相机识别 A4 纸上的 2～4 块随机碎片，完成矩形拼图规划，再控制带可旋转电磁铁的机械臂依次吸取、旋转和放置。

当前版本支持 Task1、Task2、GPIO 实体按键和终端脚本四种入口。单次任务从启动、采图、求解、搬运到复位均在同一个流程中完成，并记录完整诊断图片和计时数据。

## 已验证基线

| 项目 | 当前值 |
|---|---:|
| 主控 | NVIDIA Orin NX 16 GB、ROS 2 Humble |
| 相机 | USB RGB，MJPG 1920×1080@30 |
| 镜头距纸面 | 430 mm |
| A4 高度 | 纸面距地面 30 mm |
| 末端 | 半径 20 mm、高 20 mm 的可旋转电磁铁 |
| 纸面控制基准 | `paper_surface_z_mm=25` |
| 吸取高度 | 纸面基准上方 20 mm |
| 放置高度 | 纸面基准上方 25 mm |
| 目标顶点间距 | 5 mm，软件上限 18 mm |
| 单轮时间限制 | 120 s |
| 完成提示 | 蜂鸣器响两次，机械臂回安全位 |

最近一次实机验收：Task1 完成 4/4 片，用时 79.959 s；Task2 完成 4/4 片，用时 83.873 s。两轮均未跳过碎片，也未使用近似可达兜底。

## 系统链路

```mermaid
flowchart LR
    A["USB RGB 相机"] --> B["A4 定位与透视校正"]
    B --> C["碎片分割与多边形拟合"]
    C --> D["矩形拼图与 5 mm 间距规划"]
    D --> E["可达性、旋转与最近点兜底"]
    E --> F["ROS 控制计划"]
    F --> G["KM1 逆运动学与动作状态机"]
    G --> H["Arduino PWM 舵机与电磁铁"]
```

视觉解算以几何为主，不依赖固定的 52 张牌面模板。相机安装、坐标系、规划优先级和电磁铁时序详见 [最终技术方案](puzzle_vision/视觉到电控整体技术文档.md)。

## 目录

```text
km1_arm_ws/
├── deploy/                         # S1/S2 脚本和 systemd 服务
├── docs/                           # KM1 协议与逆运动学
├── puzzle_vision/
│   ├── config.json                 # 最终视觉参数
│   ├── main.py                     # 纯视觉命令行入口
│   ├── puzzle_vision.py            # 检测与拼图核心
│   └── 视觉到电控整体技术文档.md
└── src/km1_arm/
    ├── launch/competition_once.launch.py
    └── km1_arm/
        ├── vision_bridge.py        # 相机到 ROS 控制计划
        ├── vertical_planner.py     # 抓取、布局与可达性规划
        ├── arm_controller.py       # 实体动作状态机
        ├── serial_driver.py        # 唯一串口持有者
        ├── task_runner.py          # 单轮任务与 120 s 计时
        └── button_launcher.py      # GPIO S1/S2 映射
```

仓库不包含历史测试截图、批量回归输出、52 张牌面模板和旧版压缩包。每次现场运行产生的图片与 JSON 保存在 Orin 的 `button_runs/`，不会进入 Git。

## Orin 安装

```bash
cd ~/WorkSpace/km1_arm_ws
source /opt/ros/humble/setup.bash
python3 -m pip install -r puzzle_vision/requirements.txt
colcon build --packages-select km1_arm --symlink-install

install -m 755 deploy/111.sh ~/111.sh
install -m 755 deploy/222.sh ~/222.sh
sudo install -m 644 deploy/km1-competition-buttons.service \
  /etc/systemd/system/km1-competition-buttons.service
sudo systemctl daemon-reload
sudo systemctl enable --now km1-competition-buttons.service
```

JetPack 自带可用的 OpenCV 时可以保留系统版 `cv2`，不必在比赛前临时源码编译 OpenCV。

## 比赛使用

### 实体按键

| 按键 | Orin NX 物理引脚 | 功能 |
|---|---:|---|
| S1 | 7 | 执行 Task1 |
| S2 | 15 | 执行 Task2 |
| GND | 6 | 公共地 |
| VCC | 1 | 外设供电 |

按键低电平有效。系统会等待按键稳定释放后重新解锁，两个按键同时按下时不执行任务。

### 终端备用入口

```bash
~/111.sh    # Task1
~/222.sh    # Task2
```

两个入口与实体按键执行完全相同的代码。`flock` 防止 Task1、Task2 同时占用相机和串口。

### 只运行视觉、不驱动机械臂

```bash
source /opt/ros/humble/setup.bash
source ~/WorkSpace/km1_arm_ws/install/setup.bash
ros2 run km1_arm competition_task_runner --task 1 --no-motion
```

## 运行输出

每轮目录位于：

```text
~/WorkSpace/km1_arm_ws/button_runs/task1/<时间戳>/
~/WorkSpace/km1_arm_ws/button_runs/task2/<时间戳>/
```

关键文件如下：

| 文件 | 内容 |
|---|---|
| `00_input.png` | 动作前原始现场画面 |
| `04_detection.jpg` | 轮廓、顶点和吸附点检测 |
| `05_solution.jpg` | 拼图解算与目标布局 |
| `07_vertical_control_plan.json` | 实际执行坐标、PWM、旋转和兜底标记 |
| `08_vertical_plan.png` | 吸取与放置规划图 |
| `12_target_vs_actual.png` | 理论目标和最终实拍对比 |
| `13_timing.json` | 启动到完成的完整计时 |
| `20～27_*` | 每片吸附、释放和前后对比照片 |

## 现场注意事项

1. `km1_serial_driver`必须是 CH341 串口的唯一持有者。不要用独立 Python 脚本反复打开串口，否则 Arduino 会复位。
2. 开始前保证机械臂在纸外、A4 与底座基准没有移动、相机高度和方向没有改变。
3. 电磁铁搬运中反复吸放时，优先检查杜邦线和电源接触，再检查软件日志。
4. 若相机、A4 高度、机械臂底板或电磁铁尺寸改变，必须重新标定外参和 Z 高度。
5. `control_test`仅用于受控标定；比赛流程使用 `111.sh`、`222.sh` 或实体按键。

## 版本锚点

最终整理前的完整实测状态保存在 Git 标签：

```text
checkpoint-before-final-release-cleanup-20260801
```

该标签保留历史测试工具和素材；默认分支仅保留最终部署所需内容。
