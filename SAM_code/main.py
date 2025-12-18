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
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(OVL_DIR, exist_ok=True)

# ===== SAM 설정 (CPU, vit_b) =====
CKPT = os.path.join(ROOT, "checkpoints", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
DEVICE = "cpu"

def overlay_mask(img_bgr, mask, color=(0, 255, 0), alpha=0.5):
    """마스크를 이미지 위에 오버레이"""
    ov = img_bgr.copy()
    ov[mask] = (alpha * ov[mask] + (1-alpha) * np.array(color)).astype(np.uint8)
    return ov

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
        
        _, mask_buf = cv2.imencode('.png', (mask.astype(np.uint8) * 255))
        mask_buf.tofile(mask_png)
        
        _, ov_buf = cv2.imencode('.png', overlay_mask(img_bgr, mask))
        ov_buf.tofile(ov_png)
        
        rows.append({
            "file": fname,
            "canopy_cover_percent": round(cover_percent, 3),
            "leaf_area_cm2": None if leaf_area_cm2 is None else round(float(leaf_area_cm2), 3),
            "pos_points": len(pos_pts),
            "neg_points": len(neg_pts),
            "scaled": leaf_area_cm2 is not None,
            "postprocessed": postprocess_enabled
        })
        
        print(f"  -> Done! cover={cover_percent:.2f}%  area(cm2)={leaf_area_cm2}")
    
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "sam_canopy_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 50}")
    print(f"Saved: {out_csv}")
    print(f"Total processed: {len(rows)} images")

if __name__ == "__main__":
    main()
