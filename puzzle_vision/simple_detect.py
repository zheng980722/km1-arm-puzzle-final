"""简洁拼图块检测 v2 - 干净轮廓版"""
import cv2
import numpy as np


def detect_pieces(rectified_bgr, pixels_per_mm=6.0, min_area_mm2=200, max_area_mm2=8000):
    h, w = rectified_bgr.shape[:2]
    scale = pixels_per_mm

    # 1. HSV 绿色 = 背景
    hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([30, 40, 30]), np.array([95, 255, 255]))
    fg = cv2.bitwise_not(green)

    # 2. 去边框 (加大 margin)
    border = int(5 * scale)
    fg[:border, :] = 0; fg[-border:, :] = 0
    fg[:, :border] = 0; fg[:, -border:] = 0

    # 3. 大核形态学: 彻底去噪+填洞
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=3)

    # 4. 轮廓 + 凸包 + 多边形简化
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_px = min_area_mm2 * scale * scale
    max_px = max_area_mm2 * scale * scale

    pieces = []
    for c in contours:
        area_px = cv2.contourArea(c)
        if area_px < min_px or area_px > max_px:
            continue

        # 凸包 -> 强制凸形
        hull = cv2.convexHull(c)
        # approxPolyDP 简化到 3-6 顶点
        peri = cv2.arcLength(hull, True)
        for eps_ratio in [0.02, 0.03, 0.04, 0.05, 0.08]:
            approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
            if 3 <= len(approx) <= 6:
                break
        else:
            approx = cv2.approxPolyDP(hull, 0.05 * peri, True)

        # 中心
        M = cv2.moments(approx)
        if M['m00'] == 0:
            continue
        cx_px = M['m10'] / M['m00']
        cy_px = M['m01'] / M['m00']

        # 朝向 (minAreaRect)
        rect = cv2.minAreaRect(approx)
        angle = rect[2]
        rw, rh = rect[1]
        if rw < rh:
            angle += 90

        # 面积 (用简化后的多边形)
        clean_area = cv2.contourArea(approx)

        poly_mm = approx.reshape(-1, 2).astype(np.float64) / scale
        center_mm = np.array([cx_px / scale, cy_px / scale])

        pieces.append({
            'center_mm': center_mm,
            'orientation_deg': float(angle),
            'area_mm2': float(clean_area / (scale * scale)),
            'polygon_mm': poly_mm,
            'vertices': len(approx),
        })

    pieces.sort(key=lambda p: p['area_mm2'], reverse=True)
    for i, p in enumerate(pieces):
        p['piece_id'] = i
    return pieces


def visualize(rectified_bgr, pieces, pixels_per_mm=6.0):
    viz = rectified_bgr.copy()
    scale = pixels_per_mm
    for p in pieces:
        poly = (p['polygon_mm'] * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(viz, [poly], True, (0, 255, 255), 3)
        cx = int(p['center_mm'][0] * scale)
        cy = int(p['center_mm'][1] * scale)
        cv2.circle(viz, (cx, cy), 8, (0, 0, 255), -1)
        label = f"P{p['piece_id']} V={p['vertices']} A={p['area_mm2']:.0f}"
        cv2.putText(viz, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return viz


if __name__ == '__main__':
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else '/tmp/fix_rectified.jpg')
    pieces = detect_pieces(img)
    print(f'Detected {len(pieces)} pieces:')
    for p in pieces:
        print(f"  P{p['piece_id']}: V={p['vertices']} center=({p['center_mm'][0]:.1f},{p['center_mm'][1]:.1f}) angle={p['orientation_deg']:.1f} area={p['area_mm2']:.0f}")
    viz = visualize(img, pieces)
    cv2.imwrite('/tmp/simple_v2.jpg', viz)
    print('Saved /tmp/simple_v2.jpg')
