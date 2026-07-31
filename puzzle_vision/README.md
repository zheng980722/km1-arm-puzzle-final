# E 题拼图视觉代码 V2

本目录提供 Python/OpenCV 视觉、随机拼图求解、ROS控制计划接口和严格回归测试。
当前实物台架使用横向绿色A4：近侧长边中点与圆形底座前沿中点重合，右半区取料、
左半区拼装。相机光心距纸面实测600 mm；文档和控制坐标均以该固定条件为准。

## 已实现功能

| 模块 | V1 实现 |
|---|---|
| A4 定位 | 自动检测绿色 A4，或手动输入四角 |
| 俯视校正 | 将 A4 透视变换为毫米比例图像 |
| 碎片分割 | Lab、Lab+HSV和绿色反相三候选择优，避免人像牌绿色纹理破坏轮廓 |
| 多边形识别 | 轮廓提取、凹/凸多边形顶点拟合、质心和姿态估计 |
| 模式判断 | 自动区分自备纯色、现场白色和扑克牌模式 |
| 几何拼图 | 对2～4片进行全边/分段边匹配、矩形和外边约束 |
| 扑克牌辅助 | 使用接缝两侧颜色和梯度连续性参与候选评分 |
| 结果输出 | 输出源位姿、目标位姿、旋转量和目标轮廓 |
| 安全门控 | 横向右取左放、片数、边数、面积、空目标区、评分、1 cm间距和零重叠 |
| 控制接口 | 核心层保留安全占位；ROS `vision_bridge`默认锁定控制输出 |

## 文件说明

| 文件 | 用途 |
|---|---|
| `main.py` | 命令行入口 |
| `puzzle_vision.py` | 视觉、拼图求解和控制占位接口 |
| `config.json` | 可调参数 |
| `demo_synthetic.py` | 生成合成测试图片 |
| `download_card_templates.py` | 下载公开扑克牌演示模板 |
| `batch_test_52.py` | 52张牌面、随机目标尺寸、随机合法裁剪和随机姿态批量回归 |
| `archive_edge_cases.py` | 归档失败或达到阈值80%的临界样本 |
| `build_excel_report.mjs` | 生成含全部误差和312组截图索引的Excel |
| `测试材料准备.md` | 实物、标定和验收测试材料清单 |

## 环境

推荐 Python 3.10 或更高版本，并在本目录安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

Orin NX/JetPack 如果已经提供可用的 `cv2`，可优先保留 NVIDIA 系统版 OpenCV，
仅安装 `requirements.txt` 中的 NumPy；若 `opencv-python` 在 aarch64 上无法直接获得
预编译轮子，不建议现场临时源码编译。

生成 Excel 回归报告属于可选功能，另需 Node.js，并在本目录执行：

```bash
npm install
node build_excel_report.mjs
```

报表脚本会从当前代码目录查找批量测试结果，并将工作簿写到上一级的 `outputs`。
它默认使用
`batch_results_rule2_random_strict_v3`、`batch_results_rule2_random_strict_v2`
和异常样本目录；不生成报表时无需安装 Node.js 依赖。

## 快速测试

进入本目录后生成一张合成扑克牌碎片图片：

```bash
python3 demo_synthetic.py --mode poker --output demo_poker.png
```

如需使用真实牌面模板，下载完整52张牌面：

```bash
python3 download_card_templates.py
```

随后 `demo_synthetic.py` 会从 `assets/card_templates` 中随机选择一张牌面。可以用 `--seed` 固定随机结果，或用 `--template` 指定某张图片：

```bash
python3 demo_synthetic.py \
  --mode poker \
  --seed 7 \
  --layout-seed 1001 \
  --output demo_random_card.png

python3 demo_synthetic.py \
  --mode poker \
  --template assets/card_templates/queen_of_hearts.png \
  --output demo_queen.png
```

对52张牌面分别执行6组规则2现场模拟。每组独立生成 \(90\sim120\) mm 宽、\(50\sim90\) mm 高的目标矩形，2/3/4片各覆盖两次，同时随机裁剪、位置和角度：

```bash
python3 batch_test_52.py \
  --layouts-per-card 6 \
  --output-dir batch_results_rule2_random_strict
```

该命令共执行312轮。每组裁剪满足：2～4片、每片不超过5条边、每条边不小于20 mm、每片至少含一条目标矩形外边。`summary.json` 保存总通过率和尺寸/片数覆盖，`cases.csv`、`piece_errors.csv`、`seam_errors.csv` 分别保存组级、单片和拼接边误差；每组均保存输入、分割、检测、解算、纹理重建和汇总截图。

采用10 mm目标间距的新一轮原计划运行312组，但在完成186组后按要求停止，未形成
完整312组结论，不能写作“312/312通过”。后续若再次修改求解器，不能沿用旧版本
通过率。生成器额外要求
顶点相对相邻点连线的偏离不小于4 mm，以排除当前分辨率下不可稳定观察的近共线
退化交点；回归结论不能外推为对所有数学切割的形式化证明。

运行完整视觉流程：

```bash
python3 main.py \
  --input demo_poker.png \
  --rectified \
  --mode auto \
  --output-dir output_demo
```

输出目录包含：

| 输出文件 | 内容 |
|---|---|
| `01_rectified.png` | A4 俯视校正图 |
| `02_segmentation.png` | 碎片分割掩膜 |
| `03_detection_overlay.png` | 碎片编号、轮廓和中心 |
| `04_solution_overlay.png` | 当前横放台架左半区的目标轮廓和移动方向，不填充碎片 |
| `05_reconstructed_texture.png` | 按求解位姿刚性变换后的原纹理重建结果 |
| `result.json` | 碎片位姿、拼图结果和控制计划 |

`result.json` 的每个目标位姿还包含 `shape_preservation`。其中记录移动前后的边长、面积和最大边长误差，用来确认规划过程只有旋转和平移，没有缩放或形变。

求解器先检查未加间距的名义外接矩形是否位于规则的
\(90\sim120\) mm × \(50\sim90\) mm范围，再对每片仅做刚性平移。
所有有效拼接关系的最大对应顶点距离必须收敛到
`target_vertex_gap_mm=10.0`，允许的数值误差为±0.25 mm；另设12 mm绝对安全上限，
给规则20 mm上限继续保留控制余量。放置后的计算重叠面积必须为0，否则不输出控制计划。

## 使用相机

USB 相机索引为 0 时：

```bash
python3 main.py --input 0 --mode auto --output-dir camera_result
```

程序默认自动搜索绿色 A4 纸。如果自动定位失败，可以提供图像中的四个角点：

```bash
python3 main.py \
  --input camera.jpg \
  --corners "120,80;980,95;960,1290;105,1275" \
  --output-dir camera_result
```

四个点可以任意顺序输入，程序会自动整理为左上、右上、右下、左下。

当前固定使用MJPG 1920×1080@30，相机光心距纸面600 mm，A4短边约314 px。
只能通过5 mm低分辨率轮廓简化工作；数字裁剪或放大不会增加真实细节。若需提高采样
密度，应调整镜头视场、传感器分辨率或固定安装位置，并重新标定。普通RGB相机已经足够，
当前算法不需要深度；只有碎片翘曲、叠放或Z高度未知时才需要RGB-D。

GUI和ROS配置必须明确选择当前横向右取左放布局。若界面仍显示旧“纵向上→下”模式，
只能用于离线兼容性检查，不能向控制链发布计划。

## 坐标约定

`result.json` 使用 A4 纸毫米坐标：

| 项目 | 定义 |
|---|---|
| 原点 | A4 纸左上角 |
| \(+x\) | 向右 |
| \(+y\) | 向下 |
| 正角度 | 图像中顺时针 |

接入机械臂时，需要在控制层将纸面坐标转换为机械臂基座坐标，并根据电磁铁电机的正方向处理角度符号。

## 控制接口

`puzzle_vision.py` 中的接口为：

```python
def send_control_command(command, transport=None):
    ...
```

默认实现只打印：

```text
[CONTROL PLACEHOLDER] {...}
```

它不会打开串口、CAN、GPIO 或驱动电机，也不得新增直接串口回调。可以在命令行添加
`--print-control`检查视觉命令。真实链路位于`src/km1_arm/km1_arm/vision_bridge.py`，
并只能通过ROS话题进入常驻`km1_serial_driver`。`vision_bridge`默认
`enable_control_output=false`，`arm_controller`默认`enable_automatic_motion=false`；
当前接触Z和XY精度未标定，两把锁必须保持关闭。

## 当前版本边界

V2已经接入真实相机和ROS计划接口。当前初始映射为
`x_r=148.5-paper_y`、`y_r=265-paper_x`，已完成中心和XY四方向高位核验；
z=180～40 mm分级、ID4±30°、ID5 1100/1500 μs也已实测。接触Z和XY精度仍待标定，
控制输出必须继续锁定。改装电磁铁半径20 mm、高20 mm，`L3`暂取80 mm。

当前配置默认 `use_convex_hull_for_polygon=false`，用于保留现场随机裁剪可能产生的凹多边形。真实相机若出现阴影或局部轮廓缺口，应优先通过背景、曝光和直线全局拟合修复，不能直接取凸包，否则会把凹角抹掉。

几何求解器已经支持一条长边对应多条短边的端点锚定分段匹配。纸张翘曲、轮廓遮挡、
碎片相互接触或机械臂进入画面仍属于拒绝条件，不能靠放宽阈值强行出控制计划。
