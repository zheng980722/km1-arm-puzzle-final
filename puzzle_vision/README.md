# 拼图视觉运行模块

本目录只保留比赛运行所需的 OpenCV 代码和最终配置。历史批量回归脚本、Excel 报表、合成图片与 52 张牌面模板已从默认分支移除；需要追溯时使用仓库标签 `checkpoint-before-final-release-cleanup-20260801`。

## 文件

| 文件 | 作用 |
|---|---|
| `puzzle_vision.py` | A4 检测、分割、多边形拟合、拼图和结果绘制 |
| `config.json` | Task1/Task2 参数、HSV、间距和几何门限 |
| `main.py` | 单张图片或相机的纯视觉入口 |
| `requirements.txt` | 独立运行所需 Python 依赖 |
| `视觉到电控整体技术文档.md` | 最终视觉—电控链路说明 |

## 独立视觉运行

```bash
cd ~/WorkSpace/km1_arm_ws/puzzle_vision
python3 main.py \
  --input 0 \
  --competition-task 1 \
  --mode auto \
  --output-dir /tmp/km1_task1_preview
```

将 `--competition-task` 改为 `2` 即使用 Task2 的边长处理参数。`--input` 也可以传入现场图片的绝对路径。

## 当前视觉约束

| 项目 | Task1 | Task2 |
|---|---:|---:|
| 短伪边裁剪 | 5 mm | 10 mm |
| 目标顶点间距 | 5 mm | 5 mm |
| 搜索阶段重叠容差 | 25 mm² | 25 mm² |

两种任务的最终放置都要求计算重叠面积为 0。匹配顶点间距的软件上限为 18 mm，为题目 20 mm 上限保留控制误差。

纯视觉入口只生成结果，不直接打开机械臂串口。真实比赛链路必须从 ROS `vision_bridge` 进入 `arm_controller`，最后由唯一的 `serial_driver` 下发指令。
