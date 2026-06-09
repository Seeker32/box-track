import csv
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "manual_annotate_video.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("manual_annotate_video", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ManualAnnotateVideoTests(unittest.TestCase):
    def test_normalize_box_orders_points_and_clips_to_frame_bounds(self):
        module = load_script_module()

        box = module.normalize_box((110, -5), (10, 60), width=100, height=50)

        self.assertEqual(box, module.Box(x1=10, y1=0, x2=99, y2=49))

    def test_normalize_box_ignores_tiny_boxes(self):
        module = load_script_module()

        box = module.normalize_box((10, 10), (11, 20), width=100, height=50)

        self.assertIsNone(box)

    def test_write_annotations_saves_json_and_csv_with_fixed_id_zero_label(self):
        module = load_script_module()
        temp_dir = Path(self.create_temp_dir())
        json_path = temp_dir / "annotations.json"
        csv_path = temp_dir / "annotations.csv"
        annotations = {3: [module.Box(x1=10, y1=20, x2=40, y2=50)]}

        module.write_annotations(
            annotations=annotations,
            source_video=Path("input/demo.mp4"),
            json_path=json_path,
            csv_path=csv_path,
            fps=25.0,
            width=100,
            height=80,
        )

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["source_video"], "input/demo.mp4")
        self.assertEqual(payload["annotations"][0]["frame_id"], 3)
        self.assertEqual(payload["annotations"][0]["boxes"][0]["id"], 0)
        self.assertEqual(payload["annotations"][0]["boxes"][0]["label"], "ID 0")

        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["frame_id"], "3")
        self.assertEqual(rows[0]["id"], "0")
        self.assertEqual(rows[0]["label"], "ID 0")

    def test_draw_boxes_uses_red_box_and_id_zero_label(self):
        module = load_script_module()
        frame = np.zeros((80, 100, 3), dtype=np.uint8)

        annotated = module.draw_boxes(frame, [module.Box(x1=10, y1=20, x2=40, y2=50)])

        self.assertEqual(annotated[20, 10].tolist(), [0, 0, 255])
        self.assertGreater(int(annotated[:, :, 2].sum()), 0)

    def test_default_output_paths_follow_output_conventions(self):
        module = load_script_module()

        paths = module.default_output_paths(Path("input/demo.mp4"))

        self.assertEqual(paths.video, Path("output/videos/demo_manual_id0_red.mp4"))
        self.assertEqual(paths.json, Path("output/annotations/demo_manual_id0.json"))
        self.assertEqual(paths.csv, Path("output/annotations/demo_manual_id0.csv"))

    def test_wait_delay_refreshes_window_even_when_paused(self):
        module = load_script_module()

        self.assertGreater(module.wait_delay_ms(playing=False, fps=30.0), 0)
        self.assertEqual(module.wait_delay_ms(playing=True, fps=25.0), 40)

    def create_temp_dir(self):
        import tempfile

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return temp_dir.name


if __name__ == "__main__":
    unittest.main()
