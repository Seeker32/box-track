import csv
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "annotate_videos_from_csv.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("annotate_videos_from_csv", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnnotateVideosFromCsvTests(unittest.TestCase):
    def test_default_video_mapping_uses_sorted_vid_files_as_camera_ids(self):
        module = load_script_module()
        input_dir = Path(self.create_temp_dir())
        (input_dir / "VID_20260609_112517.mp4").write_bytes(b"")
        (input_dir / "other.mp4").write_bytes(b"")
        (input_dir / "VID_20260609_112343.mp4").write_bytes(b"")
        (input_dir / "VID_20260609_112417.mp4").write_bytes(b"")

        mapping = module.default_video_mapping(input_dir)

        self.assertEqual(
            mapping,
            {
                0: input_dir / "VID_20260609_112343.mp4",
                1: input_dir / "VID_20260609_112417.mp4",
                2: input_dir / "VID_20260609_112517.mp4",
            },
        )

    def test_load_detections_groups_csv_rows_by_camera_and_frame(self):
        module = load_script_module()
        csv_path = Path(self.create_temp_dir()) / "detections.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "global_id",
                    "camera_id",
                    "local_id",
                    "frame_id",
                    "timestamp",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "width",
                    "height",
                    "conf",
                    "cls_id",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "global_id": "3",
                "camera_id": "2",
                "local_id": "2",
                "frame_id": "114",
                "timestamp": "4.56",
                "x1": "666.0",
                "y1": "779.0",
                "x2": "1243.0",
                "y2": "1080.0",
                "width": "577.0",
                "height": "301.0",
                "conf": "0.4311",
                "cls_id": "0",
            })

        detections = module.load_detections(csv_path)

        self.assertEqual(len(detections[2][114]), 1)
        detection = detections[2][114][0]
        self.assertEqual((detection.x1, detection.y1, detection.x2, detection.y2), (666, 779, 1243, 1080))

    def test_draw_detections_uses_fixed_red_box_and_id_zero_label(self):
        module = load_script_module()
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        detection = module.Detection(x1=10, y1=20, x2=40, y2=50)

        annotated = module.draw_detections(frame, [detection], label="ID 0")

        self.assertEqual(annotated[20, 10].tolist(), [0, 0, 255])
        self.assertGreater(int(annotated[:, :, 2].sum()), 0)

    def create_temp_dir(self):
        import tempfile

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return temp_dir.name


if __name__ == "__main__":
    unittest.main()
