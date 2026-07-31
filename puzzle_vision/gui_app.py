"""E题拼图装置 - 连续帧视觉GUI软件 v2

架构: 取帧线程(全速) + 求解线程(循环处理)
叠加结果直接画在原始帧上(分辨率/角度与输入一致)
A4纸方向自动检测(横放/竖放)
"""

from __future__ import annotations

import threading
import time
import traceback
import sys
from dataclasses import replace
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

from puzzle_vision import (
    VisionConfig,
    run_pipeline,
    detect_a4_quad,
    rectify_a4,
    segment_pieces,
    evaluate_scene_quality,
    config_for_paper_resolution,
    order_quad,
    normalize_angle_deg,
)


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

class AppState(Enum):
    IDLE = auto()
    RUNNING = auto()
    ERROR = auto()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def multi_frame_median(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) == 1:
        return frames[0]
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def detect_paper_orientation(quad: np.ndarray) -> str:
    """根据检测到的四角判断A4纸是竖放还是横放。
    返回 'portrait'(210宽×297高) 或 'landscape'(297宽×210高)。
    """
    top = np.linalg.norm(quad[1] - quad[0])
    left = np.linalg.norm(quad[3] - quad[0])
    # 如果宽>高，说明纸是横放的
    if top > left:
        return "landscape"
    return "portrait"


def draw_overlay_on_frame(
    frame_bgr: np.ndarray,
    result: dict[str, Any],
    config: VisionConfig,
) -> np.ndarray:
    """将求解结果叠加画在原始相机帧上(保持原始分辨率和角度)。

    通过逆单应矩阵把rectified坐标投影回原始图像坐标。
    """
    canvas = frame_bgr.copy()
    homography = result.get("homography")
    if homography is None:
        return canvas

    # 逆单应: rectified px -> original px
    try:
        H_inv = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return canvas

    # 画A4纸边界(白色)
    w_px = config.rectified_width_px
    h_px = config.rectified_height_px
    paper_corners_rect = np.array(
        [[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], dtype=np.float64
    ).reshape(-1, 1, 2)
    paper_corners_orig = cv2.perspectiveTransform(paper_corners_rect, H_inv)
    paper_draw = np.round(paper_corners_orig).astype(np.int32).reshape(-1, 2)
    cv2.polylines(canvas, [paper_draw], True, (255, 255, 255), 2, cv2.LINE_AA)

    scale = config.pixels_per_mm
    pieces = result.get("pieces", [])
    solution = result.get("solution")

    # 画分界线 (rectified坐标中的divider_y)
    divider_y = config.divider_y_mm * scale
    divider_pts_rect = np.array(
        [[0, divider_y], [config.rectified_width_px, divider_y]],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    divider_pts_orig = cv2.perspectiveTransform(divider_pts_rect, H_inv)
    pt1 = tuple(divider_pts_orig[0, 0].astype(int))
    pt2 = tuple(divider_pts_orig[1, 0].astype(int))
    cv2.line(canvas, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    # 画每个碎片的源轮廓
    colours = [(0, 255, 255), (255, 80, 20), (180, 0, 220), (20, 180, 255)]
    for piece in pieces:
        colour = colours[piece.piece_id % len(colours)]
        # 源轮廓 (rectified px -> original px)
        contour_rect = piece.contour_px.astype(np.float64).reshape(-1, 1, 2)
        contour_orig = cv2.perspectiveTransform(contour_rect, H_inv)
        contour_draw = np.round(contour_orig).astype(np.int32)
        cv2.polylines(canvas, [contour_draw.reshape(-1, 2)], True, colour, 2, cv2.LINE_AA)

        # 源中心标注
        center_rect = (piece.center_mm * scale).reshape(1, 1, 2)
        center_orig = cv2.perspectiveTransform(center_rect, H_inv)
        cx, cy = tuple(center_orig[0, 0].astype(int))
        cv2.circle(canvas, (cx, cy), 5, colour, -1)
        cv2.putText(
            canvas, f"P{piece.piece_id}",
            (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA,
        )

    # 画目标位置
    if solution is not None:
        for piece in pieces:
            pid = piece.piece_id
            if pid not in solution.target_polygons_mm:
                continue
            colour = colours[pid % len(colours)]
            target_poly_mm = solution.target_polygons_mm[pid]
            target_rect_px = (target_poly_mm * scale).astype(np.float64).reshape(-1, 1, 2)
            target_orig = cv2.perspectiveTransform(target_rect_px, H_inv)
            target_draw = np.round(target_orig).astype(np.int32)
            cv2.polylines(canvas, [target_draw.reshape(-1, 2)], True, colour, 3, cv2.LINE_AA)

            # 箭头: 源中心 -> 目标中心
            source_center_rect = (piece.center_mm * scale).reshape(1, 1, 2)
            source_center_orig = cv2.perspectiveTransform(
                source_center_rect,
                H_inv,
            )
            cx, cy = tuple(source_center_orig[0, 0].astype(int))
            target_center_mm = solution.target_transforms[pid].translation_mm
            tc_rect = (target_center_mm * scale).reshape(1, 1, 2)
            tc_orig = cv2.perspectiveTransform(tc_rect, H_inv)
            tx, ty = tuple(tc_orig[0, 0].astype(int))
            cv2.arrowedLine(canvas, (cx, cy), (tx, ty), colour, 2, cv2.LINE_AA, tipLength=0.05)

        # 底部信息
        info = (
            f"mode={solution.mode} pieces={len(pieces)} "
            f"score={solution.score:.2f} "
            f"rect={solution.rectangle_size_mm[0]:.1f}x{solution.rectangle_size_mm[1]:.1f}mm "
            f"gap={solution.max_matched_vertex_gap_mm:.1f}mm"
        )
        cv2.putText(canvas, info, (10, canvas.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, info, (10, canvas.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class PuzzleVisionApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("E题拼图装置 - 视觉系统 v2.0")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 700)

        self.state = AppState.IDLE
        self._frozen = False  # 按A后冻结下方画面
        self.config = VisionConfig.from_json(Path(__file__).with_name("config.json"))
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_running = False
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # 连续处理结果
        self.display_frame: Optional[np.ndarray] = None  # 带叠加的帧
        self.mask_frame: Optional[np.ndarray] = None     # 绿色掩膜帧
        self.display_lock = threading.Lock()
        self.result: Optional[dict[str, Any]] = None
        self.error_msg: str = ""
        self.fps: float = 0.0
        self.frame_count_processed: int = 0

        # 线程
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()    # 关闭相机
        self._cancel_event = threading.Event()  # 停止处理循环

        self._display_images: list[ImageTk.PhotoImage] = []
        # The current bench is rotated 90 degrees (source on the physical
        # right, assembly on the left).  GUI debug defaults to that layout,
        # while config.json and the ROS bridge remain competition-safe
        # portrait/upper-to-lower by default.
        self._layout_mode = "bench_right_to_left"

        self._build_ui()
        self._refresh_display()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        # 相机
        ttk.Label(toolbar, text="相机:").pack(side=tk.LEFT, padx=(0, 2))
        self.camera_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(
            toolbar, textvariable=self.camera_var, width=8, state="readonly",
            values=["0", "1", "2", "3", "图片文件..."]
        )
        self.camera_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.camera_combo.bind("<<ComboboxSelected>>", self._on_camera_select)

        # 模式
        ttk.Label(toolbar, text="模式:").pack(side=tk.LEFT, padx=(0, 2))
        self.mode_var = tk.StringVar(value="auto")
        ttk.Combobox(
            toolbar, textvariable=self.mode_var, width=6, state="readonly",
            values=["auto", "self", "white", "poker"]
        ).pack(side=tk.LEFT, padx=(0, 10))

        # 纸面布局
        ttk.Label(toolbar, text="纸面:").pack(side=tk.LEFT, padx=(0, 2))
        self.layout_var = tk.StringVar(value="横向调试(右→左)")
        layout_combo = ttk.Combobox(
            toolbar,
            textvariable=self.layout_var,
            width=17,
            state="readonly",
            values=["横向调试(右→左)", "比赛纵向(上→下)"],
        )
        layout_combo.pack(side=tk.LEFT, padx=(0, 10))
        layout_combo.bind("<<ComboboxSelected>>", self._on_layout_select)

        # 融合帧数
        ttk.Label(toolbar, text="融合:").pack(side=tk.LEFT, padx=(0, 2))
        self.frame_count_var = tk.StringVar(value="3")
        ttk.Spinbox(toolbar, from_=1, to=10, width=3,
                    textvariable=self.frame_count_var).pack(side=tk.LEFT, padx=(0, 10))

        # 得分阈值
        ttk.Label(toolbar, text="阈值:").pack(side=tk.LEFT, padx=(0, 2))
        self.score_threshold_var = tk.StringVar(value="12")
        ttk.Spinbox(toolbar, from_=1, to=50, width=3,
                    textvariable=self.score_threshold_var).pack(side=tk.LEFT, padx=(0, 10))

        # 搜索宽度
        ttk.Label(toolbar, text="搜索:").pack(side=tk.LEFT, padx=(0, 2))
        self.beam_width_var = tk.StringVar(value="200")
        ttk.Spinbox(toolbar, from_=100, to=20000, increment=500, width=5,
                    textvariable=self.beam_width_var).pack(side=tk.LEFT, padx=(0, 10))

        # 按钮
        self.btn_open = ttk.Button(toolbar, text="打开相机", command=self._open_camera)
        self.btn_open.pack(side=tk.LEFT, padx=2)

        self.btn_start = ttk.Button(
            toolbar, text="开始", command=self._start_continuous, state=tk.DISABLED
        )
        self.btn_start.pack(side=tk.LEFT, padx=2)

        self.btn_stop = ttk.Button(
            toolbar, text="停止", command=self._stop_continuous, state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        # 主区域
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左: 上下两个画面
        left_frame = ttk.Frame(main_frame)
        main_frame.add(left_frame, weight=3)

        ttk.Label(left_frame, text="实时预览", font=("", 9)).pack(anchor=tk.W)
        self.canvas_live = tk.Canvas(left_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas_live.pack(fill=tk.BOTH, expand=True)

        ttk.Label(left_frame, text="推理结果", font=("", 9)).pack(anchor=tk.W)
        self.canvas_result = tk.Canvas(left_frame, bg="#1a1a2e", highlightthickness=0)
        self.canvas_result.pack(fill=tk.BOTH, expand=True)

        ttk.Label(left_frame, text="绿色掩膜", font=("", 9)).pack(anchor=tk.W)
        self.canvas_mask = tk.Canvas(left_frame, bg="#0a0a0a", highlightthickness=0)
        self.canvas_mask.pack(fill=tk.BOTH, expand=True)

        # 右: 信息
        right_frame = ttk.Frame(main_frame, width=300)
        main_frame.add(right_frame, weight=1)

        state_frame = ttk.LabelFrame(right_frame, text="状态", padding=5)
        state_frame.pack(fill=tk.X, pady=(0, 5))
        self.state_label = ttk.Label(state_frame, text="空闲", font=("", 11, "bold"))
        self.state_label.pack(anchor=tk.W)
        self.fps_label = ttk.Label(state_frame, text="FPS: --")
        self.fps_label.pack(anchor=tk.W)

        result_frame = ttk.LabelFrame(right_frame, text="识别结果", padding=5)
        result_frame.pack(fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(
            result_frame, height=25, width=36, font=("Consolas", 9),
            state=tk.DISABLED, wrap=tk.WORD
        )
        scroll = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # 底部
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, padding=3).pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------
    # 相机
    # ------------------------------------------------------------------

    def _on_camera_select(self, event=None):
        if self.camera_var.get() == "图片文件...":
            path = filedialog.askopenfilename(
                filetypes=[("图片", "*.png *.jpg *.bmp"), ("所有", "*.*")]
            )
            self.camera_var.set(path if path else "0")

    def _on_layout_select(self, event=None):
        self._layout_mode = (
            "competition_portrait"
            if self.layout_var.get().startswith("比赛")
            else "bench_right_to_left"
        )
        if self._layout_mode == "competition_portrait":
            self.status_var.set("纸面模式：比赛纵向，上半区取料→下半区拼装")
        else:
            self.status_var.set("纸面模式：横向调试，右半区取料→左半区拼装")

    def _config_for_selected_layout(
        self,
        base_config: VisionConfig,
    ) -> VisionConfig:
        selected = replace(base_config)
        if self._layout_mode == "competition_portrait":
            selected.require_portrait_input = True
        else:
            selected.require_portrait_input = False
            selected.landscape_source_side = "right"
        return selected

    def _open_camera(self):
        self._close_camera()
        source = self.camera_var.get()

        if Path(source).exists():
            frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if frame is None:
                messagebox.showerror("错误", f"无法读取: {source}")
                return
            with self.frame_lock:
                self.latest_frame = frame
            self.camera_running = True
            self.btn_start.configure(state=tk.NORMAL)
            self.status_var.set(f"已加载: {Path(source).name}")
            return

        try:
            idx = int(source)
        except ValueError:
            messagebox.showerror("错误", f"无效源: {source}")
            return

        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_DSHOW
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not cap.isOpened():
            messagebox.showerror("错误", f"无法打开相机 {idx}")
            return

        # /dev/video0 supports MJPG 1920x1080@30.  Without explicitly setting
        # MJPG Linux falls back to YUYV 1280x720@9, which made the live
        # recognition both slower and less accurate.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.camera_running = True
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self.btn_open.configure(text="关闭相机")
        self.btn_start.configure(state=tk.NORMAL)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.status_var.set(f"相机 {idx} 已打开: {width}x{height}@{fps:.0f} MJPG")

    def _close_camera(self):
        self._stop_event.set()
        self._cancel_event.set()
        self.camera_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_open.configure(text="打开相机")
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.DISABLED)

    def _capture_loop(self):
        while not self._stop_event.is_set():
            if self.cap is None:
                break
            ok, frame = self.cap.read()
            if ok:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    # ------------------------------------------------------------------
    # 连续处理
    # ------------------------------------------------------------------

    def _start_continuous(self):
        if self.state == AppState.RUNNING:
            return
        with self.frame_lock:
            if self.latest_frame is None:
                messagebox.showwarning("提示", "无图像")
                return
        self._cancel_event.clear()
        self.state = AppState.RUNNING
        self.state_label.configure(text="运行中")
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def _stop_continuous(self):
        self._cancel_event.set()
        self.state = AppState.IDLE
        self.state_label.configure(text="空闲")
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.status_var.set("已停止")

    def _process_loop(self):
        """连续检测循环: 只做分割+轮廓检测，不做完整求解(按a触发)"""
        frame_count = max(1, int(self.frame_count_var.get()))
        is_image = Path(self.camera_var.get()).exists()
        t_last_fps = time.time()
        n_fps = 0

        while not self._cancel_event.is_set():
            try:
                # 取帧
                frames = []
                if is_image:
                    with self.frame_lock:
                        if self.latest_frame is not None:
                            frames.append(self.latest_frame.copy())
                else:
                    for _ in range(frame_count):
                        if self._cancel_event.is_set():
                            return
                        with self.frame_lock:
                            if self.latest_frame is not None:
                                frames.append(self.latest_frame.copy())
                        time.sleep(0.03)

                if not frames:
                    time.sleep(0.05)
                    continue

                fused = multi_frame_median(frames)

                # 轻量检测: 只做rectify+分割+画轮廓，跳过beam search
                try:
                    layout_config = self._config_for_selected_layout(self.config)
                    detected_corners = detect_a4_quad(fused, layout_config)
                    frame_config = config_for_paper_resolution(
                        layout_config,
                        detected_corners,
                    )
                    rectified, homography = rectify_a4(
                        fused,
                        frame_config,
                        corners=detected_corners,
                    )
                    pieces, seg_mask, bg_lab, green_mask = segment_pieces(
                        rectified, frame_config
                    )
                    scene_quality = evaluate_scene_quality(
                        fused,
                        rectified,
                        detected_corners,
                        pieces,
                        green_mask,
                        frame_config,
                    )
                    # 构造轻量result用于叠加显示
                    result = {
                        "homography": homography,
                        "pieces": pieces,
                        "solution": None,
                        "green_mask": green_mask,
                        "segmentation_mask": seg_mask,
                        "scene_quality": scene_quality,
                    }
                except Exception as exc:
                    error_text = str(exc)
                    self.root.after(
                        0,
                        lambda message=error_text: self.status_var.set(
                            f"[视觉待机] {message}"
                        ),
                    )
                    time.sleep(0.2)
                    continue

                # 中间: 检测叠加(原图上画轮廓)
                overlay = draw_overlay_on_frame(fused.copy(), result, frame_config)

                # 下方: 绿色掩膜+碎片轮廓
                mask_vis = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
                mask_vis[green_mask == 0] = (0, 0, 60)
                colours = [(0, 255, 255), (255, 80, 20), (180, 0, 220), (20, 180, 255)]
                for p in pieces:
                    c = colours[p.piece_id % len(colours)]
                    cv2.drawContours(mask_vis, [p.contour_px], -1, c, 2, cv2.LINE_AA)
                div_y = int(
                    frame_config.divider_y_mm * frame_config.pixels_per_mm
                )
                cv2.line(mask_vis, (0, div_y), (mask_vis.shape[1], div_y), (0, 0, 255), 1)

                # 冻结时不刷新中间画面，掩膜始终刷新
                if not self._frozen:
                    with self.display_lock:
                        self.display_frame = overlay
                with self.display_lock:
                    self.mask_frame = mask_vis
                self.result = result
                self.frame_count_processed += 1

                # FPS
                n_fps += 1
                now = time.time()
                if now - t_last_fps >= 1.0:
                    self.fps = n_fps / (now - t_last_fps)
                    n_fps = 0
                    t_last_fps = now
                    self.root.after(0, lambda: self.fps_label.configure(
                        text=f"FPS: {self.fps:.1f} | 帧: {self.frame_count_processed}"
                    ))

                if self.frame_count_processed % 5 == 0 and not self._frozen:
                    self._update_result_panel(result)

                if not self._frozen:
                    if scene_quality["passed"]:
                        status = (
                            f"[连续] pieces={len(result['pieces'])} "
                            f"lower_green={scene_quality['lower_green_ratio']:.0%} "
                            "| 按A定格求解 M查看掩膜"
                        )
                    else:
                        status = "[禁止求解] " + "; ".join(scene_quality["issues"])
                    self.root.after(
                        0,
                        lambda message=status: self.status_var.set(message),
                    )

                if is_image:
                    self.root.after(0, lambda: self._stop_continuous())
                    return

            except Exception as exc:
                error_text = str(exc)
                self.root.after(
                    0,
                    lambda message=error_text: self.status_var.set(
                        f"[连续检测错误] {message}"
                    ),
                )
                time.sleep(0.3)

    def _on_key_a(self, event=None):
        """按A键: 定格当前帧，高质量求解，下方显示重建图"""
        if self.state != AppState.RUNNING:
            return
        # 在后台线程执行求解，避免阻塞UI
        threading.Thread(target=self._solve_frozen, daemon=True).start()

    def _solve_frozen(self):
        """定格求解: 循环尝试直到得分低于阈值"""
        self.root.after(0, lambda: self.state_label.configure(text="求解中..."))

        # 高质量配置
        hq_config = VisionConfig.from_json(Path(__file__).with_name("config.json"))
        hq_config = self._config_for_selected_layout(hq_config)
        hq_config.beam_width = int(self.beam_width_var.get())
        # Random legal cuts can be concave.  A convex hull changes the piece
        # geometry and previously turned valid concave pieces into wrong
        # trapezoids, so competition solving must preserve the raw polygon.
        hq_config.use_convex_hull_for_polygon = False

        frame_count = int(self.frame_count_var.get())
        is_image = Path(self.camera_var.get()).exists()
        threshold = float(self.score_threshold_var.get())
        # Median-frame fusion already stabilises one solve.  A short retry
        # window handles a transient frame without holding the GUI for tens of
        # seconds when the physical scene is genuinely invalid.
        max_attempts = 3
        attempt = 0
        best_result = None
        best_score = float("inf")
        last_error = ""
        t_start = time.time()

        try:
            while not self._cancel_event.is_set() and attempt < max_attempts:
                attempt += 1
                elapsed = time.time() - t_start
                self.root.after(0, lambda a=attempt, e=elapsed, bs=best_score: self.status_var.set(
                    f"[求解] 第{a}/{max_attempts}次 | {e:.1f}s | 最优={bs:.1f} | 阈值{threshold:.0f}"
                ))

                # 取帧融合
                frames = []
                if is_image:
                    with self.frame_lock:
                        if self.latest_frame is not None:
                            frames.append(self.latest_frame.copy())
                else:
                    for _ in range(frame_count):
                        with self.frame_lock:
                            if self.latest_frame is not None:
                                frames.append(self.latest_frame.copy())
                        time.sleep(0.04)

                if not frames:
                    time.sleep(0.1)
                    continue

                fused = multi_frame_median(frames)

                try:
                    result = run_pipeline(fused, hq_config, mode=self.mode_var.get())
                except Exception as exc:
                    last_error = str(exc)
                    self.root.after(
                        0,
                        lambda message=last_error: self.status_var.set(
                            f"[等待合规画面] {message}"
                        ),
                    )
                    time.sleep(0.2)
                    continue

                sol = result["solution"]

                # 记录最优
                if sol.score < best_score:
                    best_score = sol.score
                    best_result = result

                # 通过阈值 → 立即接受
                if sol.score <= threshold:
                    break

                time.sleep(0.05)

            # 循环结束: 只有通过界面阈值的结果才允许定格。 视觉核心还有
            # 更严格的规则门控，二者任何一个未通过都不能形成控制输出。
            if best_result is None:
                detail = last_error or "未获得有效结果"
                self.root.after(
                    0,
                    lambda message=detail: self.status_var.set(
                        f"[失败] {message}"
                    ),
                )
                self.root.after(0, lambda: self.state_label.configure(text="失败"))
                return

            result = best_result
            sol = result["solution"]
            if sol.score > threshold:
                self.root.after(
                    0,
                    lambda s=sol: self.status_var.set(
                        f"[失败] 最优score={s.score:.3f}，超过界面阈值"
                        f"{threshold:.1f}，未输出控制计划"
                    ),
                )
                self.root.after(0, lambda: self.state_label.configure(text="失败"))
                return

            # 冻结并显示最优结果
            self._frozen = True

            out_dir = Path(__file__).parent / "vision_output"
            out_dir.mkdir(exist_ok=True)
            ts = time.strftime("%H%M%S")
            reconstructed = result.get("reconstructed_texture")
            if reconstructed is not None:
                cv2.imwrite(str(out_dir / f"{ts}_reconstructed.png"), reconstructed)
                with self.display_lock:
                    self.display_frame = reconstructed
            overlay = result.get("solution_overlay")
            if overlay is not None:
                cv2.imwrite(str(out_dir / f"{ts}_solution.png"), overlay)

            self.result = result
            self._update_result_panel(result)

            elapsed = time.time() - t_start
            self.root.after(0, lambda s=sol, a=attempt, e=elapsed: self.status_var.set(
                f"[通过] 第{a}次 score={s.score:.3f} "
                f"rect={s.rectangle_size_mm[0]:.1f}x{s.rectangle_size_mm[1]:.1f}mm "
                f"gap={s.max_matched_vertex_gap_mm:.1f}mm | {e:.1f}s | 按R恢复"
            ))
            self.root.after(0, lambda: self.state_label.configure(text="已定格"))

        except Exception as exc:
            self.root.after(0, lambda: self.status_var.set(f"[错误] {exc}"))
            self.root.after(0, lambda: self.state_label.configure(text="错误"))

    def _on_key_r(self, event=None):
        """按R键: 解冻，恢复连续检测模式"""
        self._frozen = False
        self._show_mask = False
        if self.state == AppState.RUNNING:
            self._show_mask = False
            self.root.after(0, lambda: self.state_label.configure(text="运行中"))
            self.root.after(0, lambda: self.status_var.set("[连续] 已恢复实时检测"))

    def _on_key_m(self, event=None):
        """按M键: 切换显示绿色掩膜"""
        if self.result is None:
            return
        self._show_mask = not getattr(self, '_show_mask', False)
        if self._show_mask:
            green = self.result.get("green_mask")
            if green is not None:
                with self.display_lock:
                    self.display_frame = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
            self.root.after(0, lambda: self.status_var.set(
                "[掩膜] 白=检测到绿色 | 按M切回"))
        else:
            self.root.after(0, lambda: self.status_var.set("[连续] 按M查看掩膜"))

    # ------------------------------------------------------------------
    # 面板更新
    # ------------------------------------------------------------------

    def _update_result_panel(self, result: dict):
        sol = result.get("solution")
        pieces = result.get("pieces", [])

        if sol is not None:
            lines = [
                f"模式: {sol.mode}",
                f"碎片: {len(pieces)}",
                f"得分: {sol.score:.3f}",
                f"矩形: {sol.rectangle_size_mm[0]:.1f}x{sol.rectangle_size_mm[1]:.1f}mm",
                f"顶点隙: {sol.max_matched_vertex_gap_mm:.2f}mm",
                "",
            ]
            for p in pieces:
                lines.append(
                    f"P{p.piece_id}: ({p.center_mm[0]:.1f},{p.center_mm[1]:.1f})mm "
                    f"a={p.orientation_deg:.0f}d A={p.area_mm2:.0f} V={len(p.local_polygon_mm)}"
                )
            lines.append("")
            for pid in sorted(sol.target_transforms):
                t = sol.target_transforms[pid]
                lines.append(
                    f"P{pid}->({t.translation_mm[0]:.1f},{t.translation_mm[1]:.1f}) "
                    f"rot={t.angle_deg:.0f}d"
                )
        else:
            # 连续检测模式: 只显示碎片信息
            lines = [f"[检测中] 碎片: {len(pieces)}", ""]
            for p in pieces:
                lines.append(
                    f"P{p.piece_id}: ({p.center_mm[0]:.1f},{p.center_mm[1]:.1f})mm "
                    f"a={p.orientation_deg:.0f}d A={p.area_mm2:.0f} V={len(p.local_polygon_mm)}"
                )
            lines.append("")
            lines.append("按A键定格求解")

        text = "\n".join(lines)
        self.root.after(0, lambda: self._set_text(self.result_text, text))

    @staticmethod
    def _set_text(widget: tk.Text, content: str):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)
        widget.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # 显示刷新
    # ------------------------------------------------------------------

    def _refresh_display(self):
        try:
            # 上: 实时预览
            with self.frame_lock:
                live = self.latest_frame.copy() if self.latest_frame is not None else None
            if live is not None:
                self._draw_to_canvas(self.canvas_live, live, 0)

            # 中: 推理结果
            with self.display_lock:
                result_frame = self.display_frame
                mask_frame = self.mask_frame
            if result_frame is not None:
                self._draw_to_canvas(self.canvas_result, result_frame, 1)

            # 下: 绿色掩膜
            if mask_frame is not None:
                self._draw_to_canvas(self.canvas_mask, mask_frame, 2)
        except Exception:
            pass
        self.root.after(33, self._refresh_display)

    def _draw_to_canvas(self, canvas: tk.Canvas, image_bgr: np.ndarray, slot: int):
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        img_h, img_w = image_bgr.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        if new_w < 1 or new_h < 1:
            return

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))

        # 防GC: 用slot区分两个canvas的引用
        while len(self._display_images) <= slot:
            self._display_images.append(None)
        self._display_images[slot] = tk_img

        canvas.delete("all")
        canvas.create_image(
            (canvas_w - new_w) // 2, (canvas_h - new_h) // 2,
            anchor=tk.NW, image=tk_img
        )

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("a", self._on_key_a)
        self.root.bind("A", self._on_key_a)
        self.root.bind("r", self._on_key_r)
        self.root.bind("R", self._on_key_r)
        self.root.bind("m", self._on_key_m)
        self.root.bind("M", self._on_key_m)
        self.root.mainloop()

    def _on_close(self):
        self._stop_event.set()
        self._cancel_event.set()
        self._close_camera()
        self.root.destroy()


def main():
    app = PuzzleVisionApp()
    app.run()


if __name__ == "__main__":
    main()
