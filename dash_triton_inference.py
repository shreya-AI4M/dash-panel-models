"""
Dash Panel Two-Stage Triton Inference
======================================
Stage 1: Full-frame → Triton: dash_panel (objects except 'holes' class)
Stage 2: 4-crop frames → Triton: hole_detector, detections remapped to original coords

Crop logic (original 2592×1944 → 4 overlapping crops of 1555×1166):
  top_left  : (0,    0,    1555, 1166)
  top_right : (1037, 0,    2592, 1166)
  bot_left  : (0,    778,  1555, 1944)
  bot_right : (1037, 778,  2592, 1944)

ONNX output format (onnxruntime backend):
  output0 : [1, nc+4, 8400]  float32  — cx,cy,w,h + class_scores per anchor
  (dash_panel: nc=7 → dim1=11, hole_detector: nc=1 → dim1=5)

Usage:
  python dash_triton_inference.py --input <image_or_dir>
  python dash_triton_inference.py --input frame.jpg --save --output out.jpg
"""

import argparse
import os
import sys
import cv2
import numpy as np

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

TRITON_URL      = "localhost:8011"
OBJ_MODEL_NAME  = "dash_panel"
HOLE_MODEL_NAME = "hole_detector"

OBJ_INPUT_SIZE  = (640, 640)   # (width, height)
HOLE_INPUT_SIZE = (640, 640)

OBJ_CLASS_NAMES = [
    "holes", "mid_melt_sheet", "tape",
    "hole_d_lh", "hole_d_rh", "upper_melt_sheet", "bottom_melt_sheet"
]
HOLE_CLASS_NAMES = ["hole"]

HOLE_CLASS_IDX  = 0   # index of 'holes' in OBJ_CLASS_NAMES — skip in Stage 1

CONF_THRESH = 0.25
IOU_THRESH  = 0.45

# Crop size derived from the training dataset (60% of 2592×1944)
CROP_W = 1555
CROP_H = 1166

COLOURS = {
    "obj":  (0, 200, 0),    # green — object detections
    "hole": (0, 0, 255),    # red   — hole detections
}

# ──────────────────────────────────────────────────────────────────────────────
# ROI Validation Rules
# All coordinates are in the reference frame (2592×1944) and scaled at runtime.
# "roi"            : (x1, y1, x2, y2) — expected region for this class
# "required_count" : (holes only) minimum number of detections required
# ──────────────────────────────────────────────────────────────────────────────

REF_W, REF_H = 2592, 1944

ROI_RULES = {
    "upper_melt_sheet":  {"roi": (300,   0,   1200,  400)},
    "mid_melt_sheet":    {"roi": (650,   280, 1500,  700)},
    "tape":              {"roi": (0,     600, 2592, 1150)},
    "hole_d_lh":         {"roi": (0,     380,  950,  700)},
    "hole_d_rh":         {"roi": (1650,  380, 2592,  700)},
    "bottom_melt_sheet": {"roi": (0,     700, 2592, 1944)},
    "hole":              {"roi": (0,     0,   2592, 1944), "required_count": 6},
}

COL_OK     = (0,   200,   0)   # green  — detection within ROI
COL_DEFECT = (0,    0,  255)   # red    — detection outside ROI / missing
COL_ROI    = (255, 165,   0)   # orange — expected ROI boundary

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def letterbox(img: np.ndarray, new_wh: tuple) -> tuple:
    """Resize with letterbox padding. Returns (padded_img, scale, (pad_left, pad_top))."""
    h, w = img.shape[:2]
    target_w, target_h = new_wh
    scale   = min(target_w / w, target_h / h)
    new_w   = int(round(w * scale))
    new_h   = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w   = target_w - new_w
    pad_h   = target_h - new_h
    pad_l   = pad_w // 2
    pad_t   = pad_h // 2
    padded  = cv2.copyMakeBorder(
        resized, pad_t, pad_h - pad_t, pad_l, pad_w - pad_l,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, (pad_l, pad_t)


def preprocess(img_bgr: np.ndarray, input_wh: tuple) -> tuple:
    """BGR image → normalised NCHW float32 blob + letterbox meta."""
    padded, scale, padding = letterbox(img_bgr, input_wh)
    rgb  = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[np.newaxis]   # [1, 3, H, W]
    return blob, scale, padding


# ──────────────────────────────────────────────────────────────────────────────
# Box helpers
# ──────────────────────────────────────────────────────────────────────────────

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


# ──────────────────────────────────────────────────────────────────────────────
# NMS (used to merge overlapping hole detections across crops)
# ──────────────────────────────────────────────────────────────────────────────

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


# ──────────────────────────────────────────────────────────────────────────────
# Triton inference + ONNX raw output decoding
# ──────────────────────────────────────────────────────────────────────────────

def triton_infer(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    blob: np.ndarray,
) -> np.ndarray:
    """Send blob to Triton ONNX model; return raw output0 numpy array [1, nc+4, 8400]."""
    inp = grpcclient.InferInput("images", list(blob.shape), "FP32")
    inp.set_data_from_numpy(blob)

    outs = [grpcclient.InferRequestedOutput("output0")]
    result = client.infer(model_name=model_name, inputs=[inp], outputs=outs)
    return result.as_numpy("output0")   # [1, nc+4, 8400]


def decode_output(
    raw:        np.ndarray,
    orig_w: int,
    orig_h: int,
    scale:      float,
    padding:    tuple,
    conf_thresh:float,
    iou_thresh: float,
    exclude_classes: set | None = None,
) -> list:
    """Decode ONNX output [1, nc+4, 8400] → list of dicts in original-image pixel coords.

    Format: cx, cy, w, h, cls0_score, cls1_score, ... (no objectness — ultralytics style)
    """
    pred = raw[0].T       # [8400, nc+4]
    boxes_xywh   = pred[:, :4]
    class_scores = pred[:, 4:]

    confs   = class_scores.max(axis=1)
    cls_ids = class_scores.argmax(axis=1)

    mask = confs >= conf_thresh
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    confs      = confs[mask]
    cls_ids    = cls_ids[mask]

    if exclude_classes:
        valid = np.array([int(c) not in exclude_classes for c in cls_ids])
        if not valid.any():
            return []
        boxes_xywh, confs, cls_ids = boxes_xywh[valid], confs[valid], cls_ids[valid]

    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # Undo letterbox → original image coordinates
    pad_l, pad_t = padding
    boxes_xyxy[:, [0, 2]] -= pad_l
    boxes_xyxy[:, [1, 3]] -= pad_t
    boxes_xyxy /= scale
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, orig_w)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, orig_h)

    keep = nms(boxes_xyxy, confs, iou_thresh)
    boxes_xyxy = boxes_xyxy[keep]
    confs      = confs[keep]
    cls_ids    = cls_ids[keep]

    return [
        {
            "cls_id": int(cls_ids[i]),
            "conf":   float(confs[i]),
            "x1": float(boxes_xyxy[i, 0]),
            "y1": float(boxes_xyxy[i, 1]),
            "x2": float(boxes_xyxy[i, 2]),
            "y2": float(boxes_xyxy[i, 3]),
        }
        for i in range(len(boxes_xyxy))
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Crop regions
# ──────────────────────────────────────────────────────────────────────────────

def get_crop_regions(img_w: int, img_h: int) -> list:
    """4 overlapping crops — matches the cropped_holes_dataset generation logic."""
    cw = min(CROP_W, img_w)
    ch = min(CROP_H, img_h)
    xo = img_w - cw
    yo = img_h - ch
    return [
        ("top_left",  0,  0,  cw,     ch),
        ("top_right", xo, 0,  img_w,  ch),
        ("bot_left",  0,  yo, cw,     img_h),
        ("bot_right", xo, yo, img_w,  img_h),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def draw_detections(img, dets, class_names, colour, prefix=""):
    for d in dets:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        name  = class_names[d["cls_id"]] if d["cls_id"] < len(class_names) else str(d["cls_id"])
        # Use validation colour if available, else fall back to passed colour
        col   = COL_OK if d.get("valid") is True else (COL_DEFECT if d.get("valid") is False else colour)
        tag   = "" if d.get("valid") is not False else "[DEFECT] "
        label = f"{prefix}{tag}{name} {d['conf']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        by1 = max(0, y1 - th - 4)
        cv2.rectangle(img, (x1, by1), (x1 + tw + 2, y1), col, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# ROI Validation
# ──────────────────────────────────────────────────────────────────────────────

def scale_roi(roi, img_w, img_h):
    """Scale ROI from reference (2592×1944) to actual image dimensions."""
    x1, y1, x2, y2 = roi
    sx, sy = img_w / REF_W, img_h / REF_H
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


def in_roi(d, roi):
    """Check if detection centre falls inside roi (x1,y1,x2,y2)."""
    cx = (d["x1"] + d["x2"]) / 2
    cy = (d["y1"] + d["y2"]) / 2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def validate_detections(obj_dets, hole_dets, img_w, img_h):
    """
    Tag every detection with 'valid': True/False.
    Returns list of defect description strings.
    """
    defects = []

    # --- object classes ---
    for d in obj_dets:
        cls_name = OBJ_CLASS_NAMES[d["cls_id"]]
        rule = ROI_RULES.get(cls_name)
        if rule is None:
            d["valid"] = True
            continue
        roi = scale_roi(rule["roi"], img_w, img_h)
        d["valid"] = in_roi(d, roi)
        if not d["valid"]:
            defects.append(f"DEFECT: {cls_name} outside expected region")

    # --- holes: position + count ---
    rule = ROI_RULES.get("hole", {})
    roi  = scale_roi(rule["roi"], img_w, img_h) if "roi" in rule else (0, 0, img_w, img_h)
    for d in hole_dets:
        d["valid"] = in_roi(d, roi)
        if not d["valid"]:
            defects.append("DEFECT: hole outside expected region")

    valid_hole_count = sum(1 for d in hole_dets if d.get("valid"))
    required = rule.get("required_count", 6)
    if valid_hole_count < required:
        defects.append(
            f"DEFECT: hole count — found {valid_hole_count}, expected >= {required}"
        )

    return defects


def draw_roi_boxes(img, img_w, img_h):
    """Draw expected ROI boundaries in orange for all classes."""
    for cls_name, rule in ROI_RULES.items():
        roi = scale_roi(rule["roi"], img_w, img_h)
        cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), COL_ROI, 1)
        cv2.putText(img, cls_name, (roi[0] + 4, roi[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_ROI, 1, cv2.LINE_AA)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(img_path, client, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    orig_h, orig_w = img.shape[:2]
    vis = img.copy()

    # ── Stage 1: Full-frame object detection (skip 'holes' class) ─────────────
    print(f"  [Stage 1] Object detection on full frame ({orig_w}×{orig_h})")

    blob, scale, padding = preprocess(img, OBJ_INPUT_SIZE)
    raw_obj = triton_infer(client, OBJ_MODEL_NAME, blob)
    obj_dets = decode_output(
        raw_obj, orig_w, orig_h, scale, padding, conf_thresh, iou_thresh,
        exclude_classes={HOLE_CLASS_IDX},
    )
    print(f"    → {len(obj_dets)} object(s)")

    # ── Stage 2: Cropped-frame hole detection ──────────────────────────────────
    print(f"  [Stage 2] Hole detection on 4 crops")

    all_hole_dets = []
    for name, cx1, cy1, cx2, cy2 in get_crop_regions(orig_w, orig_h):
        crop = img[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]

        blob_c, sc_c, pad_c = preprocess(crop, HOLE_INPUT_SIZE)
        raw_hole = triton_infer(client, HOLE_MODEL_NAME, blob_c)

        crop_dets = decode_output(
            raw_hole, cw, ch, sc_c, pad_c, conf_thresh, iou_thresh,
        )

        # Shift crop-local coords → original frame coords
        for det in crop_dets:
            det["x1"] += cx1;  det["x2"] += cx1
            det["y1"] += cy1;  det["y2"] += cy1
            det["crop"] = name

        print(f"    [{name}] {len(crop_dets)} hole(s)")
        all_hole_dets.extend(crop_dets)

    # Merge duplicate detections from overlapping crop regions
    if len(all_hole_dets) > 1:
        h_boxes  = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in all_hole_dets])
        h_scores = np.array([d["conf"] for d in all_hole_dets])
        keep     = nms(h_boxes, h_scores, iou_thresh)
        all_hole_dets = [all_hole_dets[k] for k in keep]

    print(f"    → {len(all_hole_dets)} hole(s) after NMS merge")

    # ── ROI Validation ─────────────────────────────────────────────────────────
    defects = validate_detections(obj_dets, all_hole_dets, orig_w, orig_h)

    # Draw ROI boundaries, then detections (coloured by validity)
    vis = draw_roi_boxes(vis, orig_w, orig_h)
    vis = draw_detections(vis, obj_dets, OBJ_CLASS_NAMES, COLOURS["obj"])
    vis = draw_detections(vis, all_hole_dets, HOLE_CLASS_NAMES, COLOURS["hole"])

    # Print defect summary
    if defects:
        print(f"\n  *** {len(defects)} DEFECT(S) FOUND ***")
        for d in defects:
            print(f"    {d}")
    else:
        print("\n  All detections within expected regions. No defects.")

    return {
        "obj_detections":  obj_dets,
        "hole_detections": all_hole_dets,
        "defects":         defects,
        "image":           vis,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dash panel two-stage Triton inference")
    p.add_argument("--input",      required=True,
                   help="Image file or directory of images")
    p.add_argument("--output",     default="",
                   help="Output image path or directory (used with --save)")
    p.add_argument("--save",       action="store_true",
                   help="Save annotated images instead of displaying them")
    p.add_argument("--url",        default=TRITON_URL,
                   help=f"Triton gRPC URL (default: {TRITON_URL})")
    p.add_argument("--obj-model",  default=OBJ_MODEL_NAME)
    p.add_argument("--hole-model", default=HOLE_MODEL_NAME)
    p.add_argument("--conf",       type=float, default=CONF_THRESH)
    p.add_argument("--iou",        type=float, default=IOU_THRESH)
    return p.parse_args()


def main():
    args = parse_args()

    global OBJ_MODEL_NAME, HOLE_MODEL_NAME
    OBJ_MODEL_NAME  = args.obj_model
    HOLE_MODEL_NAME = args.hole_model

    print(f"Connecting to Triton at {args.url} ...")
    try:
        client = grpcclient.InferenceServerClient(url=args.url, verbose=False)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not client.is_server_live():
        print("ERROR: Triton server not live")
        sys.exit(1)
    if not client.is_server_ready():
        print("ERROR: Triton server not ready")
        sys.exit(1)

    for model in (OBJ_MODEL_NAME, HOLE_MODEL_NAME):
        if not client.is_model_ready(model):
            print(f"ERROR: Model '{model}' not ready on Triton")
            sys.exit(1)

    print("Triton connection OK.\n")

    # Collect images
    if os.path.isdir(args.input):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in exts
        )
    else:
        image_files = [args.input]

    if not image_files:
        print("No images found.")
        sys.exit(1)

    for img_path in image_files:
        print(f"\nProcessing: {img_path}")
        try:
            result = run_inference(img_path, client, args.conf, args.iou)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        vis = result["image"]
        print(f"  Total: {len(result['obj_detections'])} objects, "
              f"{len(result['hole_detections'])} holes, "
              f"{len(result['defects'])} defect(s)")

        if args.save or args.output:
            if args.output:
                if os.path.isdir(args.output) or len(image_files) > 1:
                    os.makedirs(args.output, exist_ok=True)
                    out_path = os.path.join(args.output, os.path.basename(img_path))
                else:
                    out_path = args.output
            else:
                base, ext = os.path.splitext(img_path)
                out_path = f"{base}_result{ext}"
            cv2.imwrite(out_path, vis)
            print(f"  Saved → {out_path}")
        else:
            cv2.imshow("Dash Panel Inference", vis)
            if cv2.waitKey(0) == ord("q"):
                break

    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
