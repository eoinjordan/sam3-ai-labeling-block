"""Tests for the SAM 3 labelling block, with no Edge Impulse key, no Space and no GPU.

The Edge Impulse API is faked at the method level, but every request goes
through the real edgeimpulse_api models, so a wrong field name fails here.

    python tests/test_label.py
    SPACE=http://127.0.0.1:7861 python tests/test_label.py   # also call a running Space's /detect
"""

import io
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import edgeimpulse_api as ei
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import label  # noqa: E402


def test_parse_prompt():
    objs = label.parse_prompt("hard hat (hardhat, 0.4)\\nperson\nforklift (truck)\n\nPerson (dup, 0.9)\nred, shiny car (car, 0.3)")
    assert [o["search_for"] for o in objs] == ["hard hat", "person", "forklift", "red shiny car"], objs
    assert objs[0] == {"search_for": "hard hat", "label": "hardhat", "min_confidence": 0.4}
    assert objs[1]["label"] == "person" and objs[1]["min_confidence"] == 0.5
    assert objs[2]["label"] == "truck" and objs[2]["min_confidence"] == 0.5


def test_load_ids():
    with tempfile.TemporaryDirectory() as tmp:
        for content in ([1, 2, 3], {"ids": [1, 2, 3]}, ["1", "2", "3"]):
            path = os.path.join(tmp, "ids.json")
            json.dump(content, open(path, "w"))
            assert label.load_ids(path) == [1, 2, 3], content


def test_boxes_from_detections():
    objects = label.parse_prompt("hard hat (hardhat, 0.5)\nperson (person, 0.3)")
    dets = [
        {"concept": "hard hat", "score": 0.9, "box_xyxy": [10.4, 20.6, 50.2, 60.0]},
        {"concept": "hard hat", "score": 0.85, "box_xyxy": [12, 21, 50, 61]},  # duplicate, NMS removes it
        {"concept": "hard hat", "score": 0.4, "box_xyxy": [100, 100, 120, 120]},  # under 0.5
        {"concept": "Person", "score": 0.35, "box_xyxy": [-5, -5, 90, 210]},  # clipped to the image
        {"concept": "person", "score": 0.9, "box_xyxy": [0, 0, 1, 1]},  # tiny, area filter
        {"concept": "dog", "score": 0.99, "box_xyxy": [0, 0, 10, 10]},  # not asked for
    ]
    out = label.boxes_from_detections(dets, objects, 100, 200, smaller_than=0.1, nms_iou=0.5)
    assert [(b["label"], b["x"], b["y"], b["width"], b["height"]) for b in out] == [
        ("hardhat", 10, 21, 40, 39), ("person", 0, 0, 90, 200)], out
    no_nms = label.boxes_from_detections(dets, objects, 100, 200, smaller_than=0.1)
    assert len(no_nms) == 3
    big_cut = label.boxes_from_detections(dets, objects, 100, 200, larger_than=50)
    assert all(b["label"] != "person" or b["width"] * b["height"] <= 10000 for b in big_cut)


def test_merge():
    existing = [{"label": "hardhat", "x": 10, "y": 20, "width": 40, "height": 40},
                {"label": "vest", "x": 0, "y": 0, "width": 5, "height": 5}]
    found = [{"label": "hardhat", "x": 11, "y": 21, "width": 40, "height": 39, "score": 0.9},
             {"label": "hardhat", "x": 60, "y": 60, "width": 20, "height": 20, "score": 0.8}]
    boxes, added = label.merge(existing, found, "no", {"hardhat"})
    assert added == 1 and len(boxes) == 3 and "score" not in boxes[-1]
    boxes, added = label.merge(existing, found, "matching-prompt", {"hardhat"})
    assert added == 2 and [b["label"] for b in boxes] == ["vest", "hardhat", "hardhat"]
    boxes, added = label.merge(existing, found, "yes", {"hardhat"})
    assert added == 2 and len(boxes) == 2


class FakeDetector:
    calls = []

    def __init__(self, space, token=None):
        self.space, self.token = space, token

    def detect(self, image_bytes, phrases, threshold):
        FakeDetector.calls.append((Image.open(io.BytesIO(image_bytes)).size, tuple(phrases), threshold))
        return [{"concept": "hard hat", "score": 0.9, "box_xyxy": [100, 50, 200, 150]},
                {"concept": "hard hat", "score": 0.7, "box_xyxy": [5, 5, 45, 45]}]


def run_main(argv, samples):
    """Run label.main() against a fake project; returns the requests the block sent."""
    sent = []
    png = io.BytesIO()
    Image.new("RGB", (320, 240), "gray").save(png, format="PNG")

    def get_sample(self, project_id, sample_id, proposed_actions_job_id=None):
        assert isinstance(project_id, int) and isinstance(sample_id, int)
        return SimpleNamespace(sample=samples[sample_id])

    def record(name):
        def fn(self, project_id, sample_id, request):
            assert type(request).__name__.endswith("Request"), request
            sent.append((name, sample_id, json.loads(request.to_json())))
        return fn

    patches = {
        (ei.RawDataApi, "get_sample"): get_sample,
        (ei.RawDataApi, "set_sample_proposed_changes"): record("proposed"),
        (ei.RawDataApi, "set_sample_bounding_boxes"): record("boxes"),
        (ei.RawDataApi, "set_sample_metadata"): record("metadata"),
    }
    saved = {k: getattr(*k) for k in patches}
    saved_get, saved_detector = label.requests.get, label.SpaceDetector
    try:
        for (cls, name), fn in patches.items():
            setattr(cls, name, fn)
        label.requests.get = lambda url, headers, timeout: SimpleNamespace(status_code=200, content=png.getvalue(), text="")
        label.SpaceDetector = FakeDetector
        os.environ.update(EI_PROJECT_API_KEY="ei_test", EI_PROJECT_ID="42")
        sys.argv = ["label.py"] + argv
        code = 0
        try:
            label.main()
        except SystemExit as e:
            code = e.code or 0
        return sent, code
    finally:
        for (cls, name), fn in saved.items():
            setattr(cls, name, fn)
        label.requests.get, label.SpaceDetector = saved_get, saved_detector


def sample(boxes=(), chart="image"):
    return SimpleNamespace(chart_type=chart, filename="img.jpg", metadata={"site": "a"},
                           bounding_boxes=[ei.BoundingBox(label=l, x=x, y=y, width=w, height=h) for l, x, y, w, h in boxes])


def test_main_writes_and_proposes():
    with tempfile.TemporaryDirectory() as tmp:
        ids = os.path.join(tmp, "ids.json")
        json.dump({"ids": [1, 2, 3]}, open(ids, "w"))
        samples = {1: sample([("hardhat", 100, 50, 100, 100)]), 2: sample(), 3: sample(chart="table")}
        base = ["--prompt", "hard hat (hardhat, 0.6)", "--data-ids-file", ids, "--nms"]

        sent, code = run_main(base, samples)
        assert code == 0, code
        writes = [s for s in sent if s[0] == "boxes"]
        # sample 1 already has the 100,50 box, so only the 5,5 one is added; sample 2 gets both
        known = {"label": "hardhat", "x": 100, "y": 50, "width": 100, "height": 100}
        new = {"label": "hardhat", "x": 5, "y": 5, "width": 40, "height": 40}
        assert writes[0][2]["boundingBoxes"] == [known, new], writes[0]
        assert writes[1][2]["boundingBoxes"] == [known, new], writes[1]
        meta = [s for s in sent if s[0] == "metadata"][0][2]["metadata"]
        assert meta["site"] == "a" and meta["labeled_by"] == "sam3", meta
        assert not any(s[1] == 3 for s in sent), "a non-image sample was written"
        assert FakeDetector.calls[0] == ((320, 240), ("hard hat",), 0.6)

        sent, code = run_main(base + ["--propose-actions", "77", "--delete-existing-bounding-boxes", "yes"], samples)
        assert code == 0 and all(s[0] == "proposed" for s in sent), sent
        by_id = {s[1]: s[2] for s in sent}
        # "replace all": the existing box goes, both found boxes are proposed
        assert by_id[1]["jobId"] == 77 and by_id[1]["proposedChanges"]["boundingBoxes"] == [known, new], by_id[1]
        assert by_id[3]["proposedChanges"] == {}, "a skipped sample needs an empty proposal in preview"


def test_live_space():
    space = os.environ.get("SPACE")
    if not space:
        print("  (skipped: set SPACE to a running Space's URL or id)")
        return
    det = label.SpaceDetector(space, os.environ.get("HF_TOKEN"))
    buf = io.BytesIO()
    Image.new("RGB", (2400, 1200), "white").save(buf, format="JPEG")  # wider than the Space's 2048 limit
    phrases = [f"thing {i}" for i in range(10)]  # more than 8, so two calls
    out = det.detect(buf.getvalue(), phrases, 0.05)
    assert isinstance(out, list)
    for d in out:
        assert set(d) == {"concept", "score", "box_xyxy"} and d["concept"] in phrases
        assert max(d["box_xyxy"]) <= 2400 * 1.01
    print(f"  live Space: {len(out)} detections over 2 calls")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
