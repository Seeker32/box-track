import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "render_cross_camera_demo.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("render_cross_camera_demo", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RenderCrossCameraDemoTests(unittest.TestCase):
    def test_discover_demo_videos_uses_sorted_id0_red_files(self):
        module = load_script_module()
        input_dir = Path(self.create_temp_dir())
        self.write_tiny_video(input_dir / "VID_20260609_112517_csv_id0_red.mp4")
        self.write_tiny_video(input_dir / "VID_20260609_112343_csv_id0_red.mp4")
        self.write_tiny_video(input_dir / "VID_20260609_112417_csv_id0_red.mp4")
        (input_dir / "VID_20260609_112417_annotated.mp4").write_bytes(b"")

        videos = module.discover_demo_videos(input_dir, "*_csv_id0_red.mp4")

        self.assertEqual(
            [segment.label for segment in videos],
            ["摄像头1", "摄像头2", "摄像头3"],
        )
        self.assertEqual(
            [segment.path.name for segment in videos],
            [
                "VID_20260609_112343_csv_id0_red.mp4",
                "VID_20260609_112417_csv_id0_red.mp4",
                "VID_20260609_112517_csv_id0_red.mp4",
            ],
        )

    def test_build_video_segments_accepts_explicit_video_order(self):
        module = load_script_module()
        input_dir = Path(self.create_temp_dir())
        first = input_dir / "cam_b.mp4"
        second = input_dir / "cam_a.mp4"
        self.write_tiny_video(first)
        self.write_tiny_video(second)

        videos = module.build_video_segments([first, second])

        self.assertEqual([segment.label for segment in videos], ["摄像头1", "摄像头2"])
        self.assertEqual([segment.path for segment in videos], [first, second])

    def test_discover_demo_videos_allows_more_than_three_matches(self):
        module = load_script_module()
        input_dir = Path(self.create_temp_dir())
        for index in range(4):
            self.write_tiny_video(input_dir / f"cam{index}_csv_id0_red.mp4")

        videos = module.discover_demo_videos(input_dir, "*_csv_id0_red.mp4")

        self.assertEqual(len(videos), 4)
        self.assertEqual(videos[-1].label, "摄像头4")

    def test_build_timeline_accumulates_segment_durations(self):
        module = load_script_module()
        segments = [
            module.VideoSegment(Path("cam1.mp4"), "摄像头1", 2.5, 24.0, 60, 640, 480),
            module.VideoSegment(Path("cam2.mp4"), "摄像头2", 1.25, 25.0, 31, 640, 480),
        ]

        timeline = module.build_timeline(segments)

        self.assertEqual(timeline[0].label, "摄像头1")
        self.assertAlmostEqual(timeline[0].start_seconds, 0.0)
        self.assertAlmostEqual(timeline[0].end_seconds, 2.5)
        self.assertEqual(timeline[1].label, "摄像头2")
        self.assertAlmostEqual(timeline[1].start_seconds, 2.5)
        self.assertAlmostEqual(timeline[1].end_seconds, 3.75)

    def test_output_frame_count_preserves_duration_at_target_fps(self):
        module = load_script_module()
        segment = module.VideoSegment(Path("cam3.mp4"), "摄像头3", 7.68, 25.0, 192, 1440, 1080)

        frame_count = module.output_frame_count(segment, target_fps=24.0)

        self.assertEqual(frame_count, 184)

    def test_compose_frame_draws_video_area_and_sidebar(self):
        module = load_script_module()
        timeline = [
            module.TimelineItem("摄像头1", 0.0, 1.0),
            module.TimelineItem("摄像头2", 1.0, 2.0),
            module.TimelineItem("摄像头3", 2.0, 3.0),
        ]
        source = np.full((120, 160, 3), 80, dtype=np.uint8)

        canvas = module.compose_frame(
            source,
            timeline=timeline,
            active_index=1,
            elapsed_seconds=1.4,
            width=960,
            height=540,
        )

        self.assertEqual(canvas.shape, (540, 960, 3))
        self.assertGreater(int(canvas[:, :, 0].sum()), 0)
        self.assertGreater(int(canvas[:, :, 1].sum()), 0)
        self.assertGreater(int(canvas[:, :, 2].sum()), 0)
        self.assertFalse(np.array_equal(canvas[:, :600], canvas[:, 600:960].mean(axis=1, keepdims=True)))

    def create_temp_dir(self):
        import tempfile

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return temp_dir.name

    def write_tiny_video(self, path):
        import cv2

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (16, 16))
        self.assertTrue(writer.isOpened())
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.release()


if __name__ == "__main__":
    unittest.main()
