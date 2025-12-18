import os, glob
import cv2
import numpy as np
import pandas as pd
from segment_anything import sam_model_registry, SamPredictor

# ===== 경로 (스크립트 위치 기준) =====
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "output")
MASK_DIR = os.path.join(OUT_DIR, "masks")
OVL_DIR  = os.path.join(OUT_DIR, "overlays")
CROP_DIR = os.path.join(OUT_DIR, "cropped")  # 누끼 이미지 저장 폴더
HIST_DIR = os.path.join(OUT_DIR, "histograms")  # 히스토그램 저장 폴더
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(OVL_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

# ===== SAM 설정 (CPU, vit_b) =====
CKPT = os.path.join(ROOT, "checkpoints", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
DEVICE = "cpu"

def overlay_mask(img_bgr, mask, color=(0, 255, 0), alpha=0.5):
    """마스크를 이미지 위에 오버레이"""
    ov = img_bgr.copy()
    ov[mask] = (alpha * ov[mask] + (1-alpha) * np.array(color)).astype(np.uint8)
    return ov

def extract_masked_region(img_bgr, mask):
    """
    마스크 영역만 추출하여 투명 배경 PNG로 반환
    - 마스크 영역: 원본 이미지 유지
    - 배경: 투명 (alpha=0)
    """
    # BGR -> BGRA (알파 채널 추가)
    img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    
    # 알파 채널 설정: 마스크 영역만 불투명 (255), 나머지 투명 (0)
    img_bgra[:, :, 3] = mask.astype(np.uint8) * 255
    
    return img_bgra

def analyze_hsv(img_bgr, mask):
    """
    마스크 영역의 HSV 통계 분석
    Returns: dict with H, S, V mean/std values
    """
    # BGR -> HSV 변환
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 마스크 영역의 픽셀만 추출
    h_values = img_hsv[:, :, 0][mask]
    s_values = img_hsv[:, :, 1][mask]
    v_values = img_hsv[:, :, 2][mask]
    
    return {
        "H_mean": float(np.mean(h_values)),
        "H_std": float(np.std(h_values)),
        "S_mean": float(np.mean(s_values)),
        "S_std": float(np.std(s_values)),
        "V_mean": float(np.mean(v_values)),
        "V_std": float(np.std(v_values))
    }

def analyze_rgb(img_bgr, mask):
    """
    마스크 영역의 RGB 통계 분석
    Returns: dict with R, G, B mean/std values
    """
    # 마스크 영역의 픽셀만 추출
    b_values = img_bgr[:, :, 0][mask]
    g_values = img_bgr[:, :, 1][mask]
    r_values = img_bgr[:, :, 2][mask]
    
    return {
        "R_mean": float(np.mean(r_values)),
        "R_std": float(np.std(r_values)),
        "G_mean": float(np.mean(g_values)),
        "G_std": float(np.std(g_values)),
        "B_mean": float(np.mean(b_values)),
        "B_std": float(np.std(b_values))
    }

def create_histogram_image(img_bgr, mask, fname):
    """
    마스크 영역의 RGB 히스토그램 이미지 생성
    Returns: histogram image (BGR)
    """
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 사용
    import matplotlib.pyplot as plt
    
    # 마스크 영역의 픽셀만 추출
    b_values = img_bgr[:, :, 0][mask]
    g_values = img_bgr[:, :, 1][mask]
    r_values = img_bgr[:, :, 2][mask]
    
    # Figure 생성
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Color Analysis: {fname}', fontsize=14, fontweight='bold')
    
    # RGB 히스토그램 (개별)
    axes[0, 0].hist(r_values, bins=256, range=(0, 256), color='red', alpha=0.7, label='Red')
    axes[0, 0].hist(g_values, bins=256, range=(0, 256), color='green', alpha=0.7, label='Green')
    axes[0, 0].hist(b_values, bins=256, range=(0, 256), color='blue', alpha=0.7, label='Blue')
    axes[0, 0].set_title('RGB Histogram')
    axes[0, 0].set_xlabel('Pixel Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # HSV 변환
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_values = img_hsv[:, :, 0][mask]
    s_values = img_hsv[:, :, 1][mask]
    v_values = img_hsv[:, :, 2][mask]
    
    # Hue 히스토그램
    axes[0, 1].hist(h_values, bins=180, range=(0, 180), color='orange', alpha=0.7)
    axes[0, 1].set_title('Hue Histogram')
    axes[0, 1].set_xlabel('Hue Value (0-180)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Saturation 히스토그램
    axes[1, 0].hist(s_values, bins=256, range=(0, 256), color='purple', alpha=0.7)
    axes[1, 0].set_title('Saturation Histogram')
    axes[1, 0].set_xlabel('Saturation Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Value 히스토그램
    axes[1, 1].hist(v_values, bins=256, range=(0, 256), color='gray', alpha=0.7)
    axes[1, 1].set_title('Value (Brightness) Histogram')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 통계 정보 추가
    stats_text = (
        f"RGB Mean: R={np.mean(r_values):.1f}, G={np.mean(g_values):.1f}, B={np.mean(b_values):.1f}\n"
        f"HSV Mean: H={np.mean(h_values):.1f}, S={np.mean(s_values):.1f}, V={np.mean(v_values):.1f}\n"
        f"Pixels: {len(r_values):,}"
    )
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    # Figure를 이미지로 변환 (최신 matplotlib 호환)
    fig.canvas.draw()
    img_array = np.asarray(fig.canvas.buffer_rgba())
    img_array = img_array[:, :, :3]  # RGBA -> RGB (알파 채널 제거)
    
    plt.close(fig)
    
    # RGB -> BGR 변환 (OpenCV용)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

def postprocess_mask(mask, kernel_size=5):
    """
    마스크 후처리: 노이즈 제거 및 구멍 메우기
    - Opening: 작은 노이즈 제거
    - Closing: 작은 구멍 메우기
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Opening (침식 후 팽창) - 작은 노이즈 제거
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # Closing (팽창 후 침식) - 작은 구멍 메우기
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    return mask_uint8 > 0

def interactive_segmentation(img_bgr, predictor, win="SAM Segmentation"):
    """
    개선된 인터랙티브 세그멘테이션
    
    조작법:
    - 좌클릭: 식물(+) 포인트 추가 (초록색)
    - 우클릭: 배경(-) 포인트 추가 (빨간색)
    - S 키: 스케일 모드 ON/OFF (보라색)
    - Z 키: 마지막 포인트 되돌리기 (Undo)
    - 1/2/3 키: 마스크 선택 (3개 중 선택)
    - P 키: 후처리 ON/OFF 토글
    - Enter: 확정
    - ESC: 이미지 건너뛰기
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    
    pos_pts = []  # 식물 포인트
    neg_pts = []  # 배경 포인트
    scale_pts = []  # 스케일 포인트
    scale_mode = False
    postprocess_enabled = True
    
    current_masks = None
    current_scores = None
    selected_mask_idx = 0  # 기본: 최고 점수 마스크
    
    def predict_mask():
        """현재 포인트로 마스크 예측"""
        nonlocal current_masks, current_scores, selected_mask_idx
        
        if len(pos_pts) == 0:
            current_masks = None
            current_scores = None
            return
        
        pts = []
        labels = []
        for (x, y) in pos_pts:
            pts.append([x, y])
            labels.append(1)
        for (x, y) in neg_pts:
            pts.append([x, y])
            labels.append(0)
        
        pts = np.array(pts)
        labels = np.array(labels)
        
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=True
        )
        
        current_masks = masks
        current_scores = scores
        selected_mask_idx = int(np.argmax(scores))  # 기본: 최고 점수
    
    def get_display_image():
        """현재 상태를 반영한 디스플레이 이미지 생성"""
        disp = img_bgr.copy()
        
        # 마스크 오버레이
        if current_masks is not None:
            mask = current_masks[selected_mask_idx].astype(bool)
            if postprocess_enabled:
                mask = postprocess_mask(mask)
            disp = overlay_mask(disp, mask, color=(0, 255, 0), alpha=0.4)
        
        # 포인트 그리기
        for (x, y) in pos_pts:
            cv2.circle(disp, (x, y), 6, (0, 255, 0), -1)
            cv2.circle(disp, (x, y), 6, (255, 255, 255), 2)
        for (x, y) in neg_pts:
            cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(disp, (x, y), 6, (255, 255, 255), 2)
        for (x, y) in scale_pts:
            cv2.circle(disp, (x, y), 6, (255, 0, 255), -1)
            cv2.circle(disp, (x, y), 6, (255, 255, 255), 2)
        
        # 스케일 선 그리기
        if len(scale_pts) >= 2:
            cv2.line(disp, scale_pts[0], scale_pts[1], (255, 0, 255), 2)
        
        # 정보 표시
        info_lines = [
            "L:plant(+) R:bg(-) Z:undo S:scale",
            "1/2/3:select mask  P:postprocess  Enter:OK  ESC:skip"
        ]
        
        if current_scores is not None:
            scores_str = f"Scores: [{current_scores[0]:.2f}, {current_scores[1]:.2f}, {current_scores[2]:.2f}]"
            info_lines.append(f"{scores_str}  Selected: {selected_mask_idx + 1}")
        
        mode_str = f"Mode: {'SCALE' if scale_mode else 'POINT'}  PostProc: {'ON' if postprocess_enabled else 'OFF'}"
        info_lines.append(mode_str)
        
        for i, line in enumerate(info_lines):
            cv2.putText(disp, line, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 0), 3)
            cv2.putText(disp, line, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
        
        return disp
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal scale_mode
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if scale_mode:
                if len(scale_pts) < 2:
                    scale_pts.append((x, y))
            else:
                pos_pts.append((x, y))
                predict_mask()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not scale_mode:
                neg_pts.append((x, y))
                predict_mask()
    
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_callback)
    
    skip = False
    
    while True:
        disp = get_display_image()
        cv2.imshow(win, disp)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == 13:  # Enter - 확정
            break
        elif key == 27:  # ESC - 건너뛰기
            skip = True
            break
        elif key in (ord('z'), ord('Z')):  # Undo
            if scale_mode and len(scale_pts) > 0:
                scale_pts.pop()
            elif len(neg_pts) > 0 and (len(pos_pts) == 0 or neg_pts[-1] > pos_pts[-1]):
                neg_pts.pop()
                predict_mask()
            elif len(pos_pts) > 0:
                pos_pts.pop()
                predict_mask()
        elif key in (ord('s'), ord('S')):  # Scale mode 토글
            scale_mode = not scale_mode
        elif key in (ord('p'), ord('P')):  # Postprocess 토글
            postprocess_enabled = not postprocess_enabled
        elif key == ord('1'):  # 마스크 1 선택
            if current_masks is not None:
                selected_mask_idx = 0
        elif key == ord('2'):  # 마스크 2 선택
            if current_masks is not None:
                selected_mask_idx = 1
        elif key == ord('3'):  # 마스크 3 선택
            if current_masks is not None:
                selected_mask_idx = 2
    
    cv2.destroyWindow(win)
    
    if skip or current_masks is None:
        return None, None, np.array(scale_pts), postprocess_enabled
    
    # 최종 마스크 반환
    final_mask = current_masks[selected_mask_idx].astype(bool)
    
    return final_mask, (np.array(pos_pts), np.array(neg_pts)), np.array(scale_pts), postprocess_enabled

def main():
    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")
    
    print("Loading SAM model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CKPT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    print("Model loaded!\n")
    
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(DATA_DIR, e))
    if not paths:
        raise FileNotFoundError(f"No images in: {DATA_DIR}")
    
    print(f"Found {len(paths)} images\n")
    print("=" * 50)
    print("조작법:")
    print("  좌클릭: 식물(+) 포인트")
    print("  우클릭: 배경(-) 포인트")
    print("  Z: 되돌리기 (Undo)")
    print("  S: 스케일 모드 (면적 계산용)")
    print("  1/2/3: 마스크 선택")
    print("  P: 후처리 ON/OFF")
    print("  Enter: 확정")
    print("  ESC: 건너뛰기")
    print("=" * 50 + "\n")
    
    rows = []
    
    for i, p in enumerate(sorted(paths)):
        fname = os.path.basename(p)
        print(f"[{i+1}/{len(paths)}] Processing: {fname}")
        
        # 한글 경로 지원
        img_bgr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  -> Skip (read fail)")
            continue
        
        # 인터랙티브 세그멘테이션
        result = interactive_segmentation(img_bgr, predictor, win=fname)
        mask, points_info, scale_pts, postprocess_enabled = result
        
        if mask is None:
            print(f"  -> Skip (no mask)")
            continue
        
        # 후처리 적용
        if postprocess_enabled:
            mask = postprocess_mask(mask)
        
        pos_pts, neg_pts = points_info
        
        # ---- 지표 계산 ----
        cover_percent = 100.0 * mask.sum() / mask.size
        
        # 스케일 계산
        leaf_area_cm2 = None
        if len(scale_pts) >= 2:
            LENGTH_CM = 100.0  # 줄자 1m 기준 (필요시 수정)
            (x1, y1), (x2, y2) = scale_pts[0], scale_pts[1]
            px_len = float(np.hypot(x2 - x1, y2 - y1))
            if px_len > 0:
                cm_per_px = LENGTH_CM / px_len
                leaf_area_cm2 = mask.sum() * (cm_per_px ** 2)
        
        # ---- 저장 (한글 경로 지원) ----
        mask_png = os.path.join(MASK_DIR, fname + "_mask.png")
        ov_png = os.path.join(OVL_DIR, fname + "_overlay.png")
        crop_png = os.path.join(CROP_DIR, fname + "_cropped.png")
        
        # 흑백 마스크 저장
        _, mask_buf = cv2.imencode('.png', (mask.astype(np.uint8) * 255))
        mask_buf.tofile(mask_png)
        
        # 오버레이 이미지 저장
        _, ov_buf = cv2.imencode('.png', overlay_mask(img_bgr, mask))
        ov_buf.tofile(ov_png)
        
        # 누끼 이미지 저장 (마스크 영역만 + 투명 배경)
        cropped_img = extract_masked_region(img_bgr, mask)
        _, crop_buf = cv2.imencode('.png', cropped_img)
        crop_buf.tofile(crop_png)
        
        # HSV 및 RGB 분석 (cropped 영역에 대해)
        hsv_stats = analyze_hsv(img_bgr, mask)
        rgb_stats = analyze_rgb(img_bgr, mask)
        
        # 히스토그램 이미지 생성 및 저장
        hist_png = os.path.join(HIST_DIR, fname + "_histogram.png")
        hist_img = create_histogram_image(img_bgr, mask, fname)
        _, hist_buf = cv2.imencode('.png', hist_img)
        hist_buf.tofile(hist_png)

        row_data = {
            "file": fname,
            "canopy_cover_percent": round(cover_percent, 3),
            "leaf_area_cm2": None if leaf_area_cm2 is None else round(float(leaf_area_cm2), 3),
            "pos_points": len(pos_pts),
            "neg_points": len(neg_pts),
            "scaled": leaf_area_cm2 is not None,
            "postprocessed": postprocess_enabled,
            # RGB 통계
            "R_mean": round(rgb_stats["R_mean"], 2),
            "R_std": round(rgb_stats["R_std"], 2),
            "G_mean": round(rgb_stats["G_mean"], 2),
            "G_std": round(rgb_stats["G_std"], 2),
            "B_mean": round(rgb_stats["B_mean"], 2),
            "B_std": round(rgb_stats["B_std"], 2),
            # HSV 통계
            "H_mean": round(hsv_stats["H_mean"], 2),
            "H_std": round(hsv_stats["H_std"], 2),
            "S_mean": round(hsv_stats["S_mean"], 2),
            "S_std": round(hsv_stats["S_std"], 2),
            "V_mean": round(hsv_stats["V_mean"], 2),
            "V_std": round(hsv_stats["V_std"], 2)
        }
        rows.append(row_data)
        
        # 실시간 CSV 저장 (각 이미지 처리 후 바로 저장)
        out_csv = os.path.join(OUT_DIR, "sam_canopy_results.csv")
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        
        print(f"  -> Done! cover={cover_percent:.2f}%  area(cm2)={leaf_area_cm2}")
        print(f"     RGB: R={rgb_stats['R_mean']:.1f}, G={rgb_stats['G_mean']:.1f}, B={rgb_stats['B_mean']:.1f}")
        print(f"     HSV: H={hsv_stats['H_mean']:.1f}, S={hsv_stats['S_mean']:.1f}, V={hsv_stats['V_mean']:.1f}")
        print(f"     [CSV 저장 완료]")
    
    print(f"\n{'=' * 50}")
    print(f"Saved: {os.path.join(OUT_DIR, 'sam_canopy_results.csv')}")
    print(f"Total processed: {len(rows)} images")

if __name__ == "__main__":
    main()
