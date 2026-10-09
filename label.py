"""SAM 3 AI labelling block for Edge Impulse.

Each image in the selection goes to a Hugging Face Space that runs SAM 3
(eoinedge/sam3 by default) with the prompt's phrases; the boxes that come back
become bounding box labels. In preview, Studio passes --propose-actions and the
boxes are proposed for review instead of written.

    python label.py --prompt "hard hat (hardhat, 0.5)" --data-ids-file ids.json
"""

import argparse
import io
import json
import os
import sys
import tempfile
import time

import requests
from PIL import Image

MAX_CONCEPTS = 8  # the Space takes at most 8 phrases per call
DEFAULT_MIN_CONFIDENCE = 0.5


class QuotaExceeded(Exception):
    pass


def parse_prompt(text, default_min=DEFAULT_MIN_CONFIDENCE):
    """One object per line: 'phrase', 'phrase (label)' or 'phrase (label, min confidence)'."""
    objects, seen = [], set()
    # Studio may pass newlines escaped
    for line in text.replace("\\n", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        start, end = line.rfind("("), line.rfind(")")
        if start != -1 and end > start:
            search_for = line[:start].strip()
            inside = [p.strip() for p in line[start + 1:end].split(",")]
            label = inside[0] or search_for
            min_conf = float(inside[1]) if len(inside) > 1 and inside[1] else default_min
        else:
            search_for, label, min_conf = line, line, default_min
        # the Space splits phrases on commas
        search_for = " ".join(search_for.replace(",", " ").split())
        if not search_for or search_for.lower() in seen:
            continue
        seen.add(search_for.lower())
        objects.append({"search_for": search_for, "label": label, "min_confidence": min_conf})
    return objects


def load_ids(path):
    """The IDs file: a JSON array, or an object with an 'ids' array (both appear in Edge Impulse's material)."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("ids", data.get("dataIds", []))
    return [int(x) for x in data]


def iou(a, b):
    """IoU of two boxes as dicts with x, y, width, height."""
    ix = max(0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, threshold):
    """Greedy per-label non-max suppression; boxes carry a score."""
    kept = []
    for box in sorted(boxes, key=lambda b: -b["score"]):
        if all(k["label"] != box["label"] or iou(k, box) <= threshold for k in kept):
            kept.append(box)
    return kept


def boxes_from_detections(detections, objects, width, height, smaller_than=0.0, larger_than=100.0, nms_iou=None):
    """SAM 3 detections (concept, score, box_xyxy) -> Edge Impulse boxes, filtered and deduplicated."""
    by_phrase = {o["search_for"].lower(): o for o in objects}
    area = float(width * height)
    out = []
    for d in detections:
        obj = by_phrase.get(d["concept"].lower())
        if obj is None or d["score"] < obj["min_confidence"]:
            continue
        x0, y0, x1, y1 = d["box_xyxy"]
        x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
        x1, y1 = min(width, int(round(x1))), min(height, int(round(y1)))
        w, h = x1 - x0, y1 - y0
        if w < 1 or h < 1:
            continue
        pct = 100.0 * w * h / area
        if smaller_than and pct < smaller_than:
            continue
        if larger_than is not None and larger_than < 100 and pct > larger_than:
            continue
        out.append({"label": obj["label"], "x": x0, "y": y0, "width": w, "height": h, "score": d["score"]})
    if nms_iou is not None:
        out = nms(out, nms_iou)
    return out


def merge(existing, found, mode, labels_in_prompt, duplicate_iou=0.5):
    """Combine the sample's boxes with SAM 3's.

    no: keep every existing box, and add SAM 3 boxes that don't repeat an existing box of the same label.
    matching-prompt: drop existing boxes whose label is in the prompt, add all SAM 3 boxes.
    yes: drop all existing boxes, add all SAM 3 boxes.
    """
    if mode == "yes":
        kept = []
    elif mode == "matching-prompt":
        kept = [b for b in existing if b["label"] not in labels_in_prompt]
    else:
        kept = list(existing)
    added = []
    for box in found:
        if mode == "no" and any(e["label"] == box["label"] and iou(e, box) >= duplicate_iou for e in existing):
            continue
        added.append({k: box[k] for k in ("label", "x", "y", "width", "height")})
    return kept + added, len(added)


class SpaceDetector:
    """Calls the Space's /detect endpoint: boxes only, in the original image's pixels."""

    def __init__(self, space, token=None, retries=3):
        from gradio_client import Client

        self.space = space
        self.retries = retries
        self.client = Client(space, token=token or None, verbose=False)

    def detect(self, image_bytes, phrases, threshold):
        from gradio_client import handle_file

        threshold = min(0.95, max(0.05, threshold))
        detections = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "image.png")
            Image.open(io.BytesIO(image_bytes)).convert("RGB").save(path)
            for i in range(0, len(phrases), MAX_CONCEPTS):
                batch = ", ".join(phrases[i:i + MAX_CONCEPTS])
                result = self._call(handle_file(path), batch, threshold)
                detections.extend(result["detections"])
        return detections

    def _call(self, image, concepts, threshold):
        for attempt in range(1, self.retries + 1):
            try:
                return self.client.predict(image, concepts, threshold, api_name="/detect")
            except Exception as e:
                message = str(e)
                if "quota" in message.lower():
                    raise QuotaExceeded(message) from e
                if attempt == self.retries:
                    raise
                wait = 5 * 3 ** (attempt - 1)
                print(f"    Space call failed ({message[:200]}); retrying in {wait} s")
                time.sleep(wait)


def as_dict(bb):
    return {"label": bb.label, "x": bb.x, "y": bb.y, "width": bb.width, "height": bb.height}


def main():
    p = argparse.ArgumentParser(description="SAM 3 AI labelling block for Edge Impulse")
    p.add_argument("--prompt", type=str, required=True,
                   help="One object per line: 'phrase (label, min confidence)', e.g. 'hard hat (hardhat, 0.5)'")
    p.add_argument("--data-ids-file", type=str, required=True, help="JSON file with the sample IDs")
    p.add_argument("--propose-actions", type=int, required=False,
                   help="Set by Studio for a preview: propose changes instead of writing them")
    p.add_argument("--delete-existing-bounding-boxes", "--delete_existing_bounding_boxes",
                   dest="delete_existing", type=str, default="no", choices=["no", "matching-prompt", "yes"])
    p.add_argument("--ignore-objects-smaller-than", type=float, default=0.0)
    p.add_argument("--ignore-objects-larger-than", type=float, default=100.0)
    p.add_argument("--nms", action="store_true", help="Remove overlapping boxes of the same label")
    p.add_argument("--nms-iou-threshold", type=float, default=0.5)
    p.add_argument("--space", type=str, default="eoinedge/sam3", help="Hugging Face Space that runs SAM 3")
    args, unknown = p.parse_known_args()

    api_key = os.environ.get("EI_PROJECT_API_KEY")
    if not api_key:
        sys.exit("Missing EI_PROJECT_API_KEY")
    endpoint = os.environ.get("EI_API_ENDPOINT", "https://studio.edgeimpulse.com/v1")

    objects = parse_prompt(args.prompt)
    if not objects:
        sys.exit("The prompt has no objects; write one per line, e.g. 'hard hat (hardhat, 0.5)'.")
    phrases = [o["search_for"] for o in objects]
    labels_in_prompt = {o["label"] for o in objects}
    threshold = min(o["min_confidence"] for o in objects)
    data_ids = load_ids(args.data_ids_file)

    import edgeimpulse_api as ei

    configuration = ei.Configuration(host=endpoint)
    configuration.api_key["ApiKeyAuthentication"] = api_key
    api = ei.ApiClient(configuration)
    raw_data_api = ei.RawDataApi(api)
    project_id = int(os.environ.get("EI_PROJECT_ID") or ei.ProjectsApi(api).list_projects().projects[0].id)

    print(f"Labelling with SAM 3 through the Hugging Face Space {args.space}")
    for o in objects:
        print(f"    '{o['search_for']}' -> label {o['label']}, min confidence {o['min_confidence']}")
    print(f"    {len(data_ids)} samples; existing boxes: {args.delete_existing}"
          + ("; preview (proposed changes only)" if args.propose_actions else ""))
    print("Connecting to the Space (a sleeping Space can take a minute or two to wake)...")
    detector = SpaceDetector(args.space, os.environ.get("HF_TOKEN"))

    def propose_nothing(data_id):
        if args.propose_actions:
            raw_data_api.set_sample_proposed_changes(project_id, data_id, ei.SetSampleProposedChangesRequest.from_dict(
                {"jobId": args.propose_actions, "proposedChanges": {}}))

    failed, skipped, total_added = [], 0, 0
    width = len(str(len(data_ids)))
    for ix, data_id in enumerate(data_ids, 1):
        prefix = f"[{str(ix).rjust(width)}/{len(data_ids)}]"
        start = time.time()
        try:
            sample = raw_data_api.get_sample(project_id, data_id, proposed_actions_job_id=args.propose_actions).sample
            if sample.chart_type != "image":
                print(prefix, f"Skipping {sample.filename} (ID {data_id}): not an image ({sample.chart_type})")
                propose_nothing(data_id)
                skipped += 1
                continue

            res = requests.get(f"{endpoint}/api/{project_id}/raw-data/{data_id}/image",
                               headers={"x-api-key": api_key}, timeout=60)
            if res.status_code != 200:
                raise RuntimeError(f"fetching the image failed ({res.status_code}): {res.text[:200]}")
            image = Image.open(io.BytesIO(res.content))
            img_w, img_h = image.size

            detections = detector.detect(res.content, phrases, threshold)
            found = boxes_from_detections(detections, objects, img_w, img_h, args.ignore_objects_smaller_than,
                                          args.ignore_objects_larger_than, args.nms_iou_threshold if args.nms else None)
            existing = [as_dict(bb) for bb in (sample.bounding_boxes or [])]
            boxes, added = merge(existing, found, args.delete_existing, labels_in_prompt)
            total_added += added

            metadata = dict(sample.metadata or {})
            metadata["labeled_by"] = "sam3"
            metadata["sam3_prompt"] = "; ".join(f"{o['search_for']} ({o['label']}, {o['min_confidence']})" for o in objects)

            if args.propose_actions:
                raw_data_api.set_sample_proposed_changes(project_id, data_id, ei.SetSampleProposedChangesRequest.from_dict({
                    "jobId": args.propose_actions,
                    "proposedChanges": {"boundingBoxes": boxes, "metadata": metadata},
                }))
            else:
                raw_data_api.set_sample_bounding_boxes(project_id, data_id, ei.SampleBoundingBoxesRequest.from_dict(
                    {"boundingBoxes": boxes}))
                raw_data_api.set_sample_metadata(project_id, data_id, ei.SetSampleMetadataRequest.from_dict(
                    {"metadata": metadata}))

            counts = {}
            for b in found:
                counts[b["label"]] = counts.get(b["label"], 0) + 1
            summary = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "nothing above the thresholds"
            print(prefix, f"{sample.filename} (ID {data_id}): {summary}; {added} added, {len(boxes)} boxes now "
                          f"({time.time() - start:.1f} s)")
        except QuotaExceeded as e:
            print(prefix, f"Stopped: the Space's GPU quota is used up ({str(e)[:300]}).")
            print("Set the block's Hugging Face token to an account with more ZeroGPU time (PRO), wait for the "
                  "quota to reset, or point the block at your own copy of the Space on paid hardware.")
            failed.extend(data_ids[ix - 1:])
            break
        except Exception as e:
            print(prefix, f"ID {data_id} failed: {e}")
            failed.append(data_id)
            try:
                propose_nothing(data_id)
            except Exception:
                pass

    print("")
    print(f"Done: {len(data_ids) - len(failed) - skipped} of {len(data_ids)} samples labelled, {total_added} boxes added"
          + (f", {skipped} skipped (not images)." if skipped else "."))
    if failed:
        shown = ", ".join(str(x) for x in failed[:20]) + (" ..." if len(failed) > 20 else "")
        print(f"{len(failed)} samples were not labelled: {shown}")
        sys.exit(1)


if __name__ == "__main__":
    main()
