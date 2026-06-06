"""Global ID manager for cross-camera identity propagation.

Phase 2: Supports both batch-mode (Phase 1) and online real-time matching.
Maintains full trajectory history per global target and supports
timeout-based cleanup of disappeared targets.
"""

import logging

from src.matching.hungarian_matcher import cross_camera_match
from src.utils.tracklet import Tracklet

logger = logging.getLogger(__name__)


class GlobalIDManager:
    """Maintains cross-camera global target IDs.

    Phase 1 (batch): register_new() + match_and_assign() for two-pass matching.
    Phase 2 (online): register_or_match() for real-time matching as tracklets
    complete, with cleanup_expired() to prune disappeared targets.

    Usage (batch — Phase 1):
        mgr = GlobalIDManager()
        for t in cam_a_tracklets:
            mgr.register_new(t)
        mgr.match_and_assign(cam_a_tracklets, cam_b_tracklets, config)

    Usage (online — Phase 2):
        mgr = GlobalIDManager(disappearance_timeout=60.0)
        # As each tracklet completes:
        gid = mgr.register_or_match(tracklet, config)
        # Periodically:
        mgr.cleanup_expired(now)
    """

    def __init__(self, disappearance_timeout: float = 60.0):
        """Initialize the manager.

        Args:
            disappearance_timeout: Seconds before a target is considered
                                   gone and removed by cleanup_expired().
        """
        self._next_id: int = 0
        self.disappearance_timeout = disappearance_timeout
        # global_id → list of Tracklets (full trajectory history)
        self._active_targets: dict[int, list[Tracklet]] = {}

    # ------------------------------------------------------------------
    # Phase 1: batch-mode API (kept for backward compatibility)
    # ------------------------------------------------------------------

    def register_new(self, tracklet: Tracklet) -> int:
        """Assign a new global ID to a tracklet.

        Args:
            tracklet: The tracklet to register.

        Returns:
            The newly assigned global ID.
        """
        gid = self._next_id
        self._next_id += 1
        tracklet.global_id = gid
        self._active_targets[gid] = [tracklet]
        logger.debug("Registered new global ID %d for camera %d local %d",
                     gid, tracklet.camera_id, tracklet.local_id)
        return gid

    def match_and_assign(
        self,
        tracklets_a: list[Tracklet],
        tracklets_b: list[Tracklet],
        config: dict,
    ) -> dict[int, int]:
        """Match camera B tracklets against camera A and assign global IDs.

        Args:
            tracklets_a: Tracklets from the earlier camera.
            tracklets_b: Tracklets from the later camera.
            config: Matching configuration dict.

        Returns:
            dict mapping {global_id: local_id} for newly assigned tracklets
            from camera B.
        """
        # Ensure camera A tracklets have global IDs
        for t in tracklets_a:
            if t.global_id is None:
                self.register_new(t)

        # Run cross-camera matching
        matches, unmatched_a, unmatched_b = cross_camera_match(
            tracklets_a, tracklets_b, config
        )

        # Apply matches: B tracklet inherits A's global ID
        for b_idx, a_idx in matches.items():
            gid = tracklets_a[a_idx].global_id
            if gid is None:
                gid = self.register_new(tracklets_a[a_idx])
            tracklets_b[b_idx].global_id = gid
            if gid in self._active_targets:
                self._active_targets[gid].append(tracklets_b[b_idx])
            else:
                self._active_targets[gid] = [tracklets_b[b_idx]]
            logger.debug("Matched: cam %d local %d → global %d (from cam %d local %d)",
                         tracklets_b[b_idx].camera_id, tracklets_b[b_idx].local_id,
                         gid, tracklets_a[a_idx].camera_id, tracklets_a[a_idx].local_id)

        # Unmatched B tracklets get new global IDs
        result: dict[int, int] = {}
        for b_idx in unmatched_b:
            gid = self.register_new(tracklets_b[b_idx])
            result[gid] = tracklets_b[b_idx].local_id

        return result

    # ------------------------------------------------------------------
    # Phase 2: online-mode API
    # ------------------------------------------------------------------

    def register_or_match(
        self,
        new_tracklet: Tracklet,
        config: dict,
    ) -> int:
        """Register a newly completed tracklet or match it to an existing target.

        For online mode: called each time a tracklet finishes in its camera.
        Matches the new tracklet against the latest tracklet of each active
        global target. If a match is found, the new tracklet inherits that
        global ID. Otherwise, a new global ID is created.

        Args:
            new_tracklet: A completed tracklet from any camera.
            config: Matching configuration dict.

        Returns:
            The assigned global ID.
        """
        active_list = self.get_active()
        if not active_list:
            # No active targets yet — create a new one
            return self._create_new_target(new_tracklet)

        # Build list of latest tracklets for matching
        latest_tracklets = [t for _, t in active_list]

        # Match new_tracklet against each active target's latest tracklet
        matches, unmatched_a, unmatched_b = cross_camera_match(
            latest_tracklets, [new_tracklet], config
        )

        if 0 in matches:
            # new_tracklet matched to an existing active target
            a_idx = matches[0]
            gid = latest_tracklets[a_idx].global_id
            if gid is None:
                gid = self._create_new_target(new_tracklet)
            else:
                new_tracklet.global_id = gid
                self._active_targets[gid].append(new_tracklet)
                logger.info(
                    "Online match: cam %d local %d → global %d",
                    new_tracklet.camera_id, new_tracklet.local_id, gid,
                )
            return gid
        else:
            # No match — create a new global target
            return self._create_new_target(new_tracklet)

    def _create_new_target(self, tracklet: Tracklet) -> int:
        """Create a new global target from a tracklet."""
        gid = self._next_id
        self._next_id += 1
        tracklet.global_id = gid
        self._active_targets[gid] = [tracklet]
        logger.info(
            "New global target %d: cam %d local %d",
            gid, tracklet.camera_id, tracklet.local_id,
        )
        return gid

    def cleanup_expired(self, current_time: float) -> list[int]:
        """Remove targets that have been gone longer than the timeout.

        A target is "expired" when its most recent tracklet's end_time is
        more than `disappearance_timeout` seconds before `current_time`.

        Args:
            current_time: Current timestamp (seconds).

        Returns:
            List of global IDs that were removed.
        """
        expired: list[int] = []
        for gid, tracklets in self._active_targets.items():
            last_seen = max(t.end_time for t in tracklets)
            if current_time - last_seen > self.disappearance_timeout:
                expired.append(gid)

        for gid in expired:
            tracklets = self._active_targets.pop(gid)
            last_t = tracklets[-1] if tracklets else None
            logger.info(
                "Cleanup global %d: last seen %.1fs ago (cam %d)",
                gid,
                current_time - (last_t.end_time if last_t else 0),
                last_t.camera_id if last_t else -1,
            )

        return expired

    def get_active(self) -> list[tuple[int, Tracklet]]:
        """Return (global_id, latest_tracklet) for all active targets.

        Returns:
            List of (global_id, latest_tracklet) tuples, sorted by global_id.
        """
        result: list[tuple[int, Tracklet]] = []
        for gid, tracklets in sorted(self._active_targets.items()):
            if tracklets:
                result.append((gid, tracklets[-1]))
        return result

    def get_trajectory(self, global_id: int) -> list[Tracklet] | None:
        """Return the full trajectory (all tracklets) for a global target.

        Args:
            global_id: The global target ID.

        Returns:
            List of Tracklets across all cameras, or None if not found.
        """
        return self._active_targets.get(global_id)

    @property
    def num_active(self) -> int:
        """Number of currently active global targets."""
        return len(self._active_targets)

    @property
    def all_tracklets(self) -> list[Tracklet]:
        """Return all tracklets from all global targets (flat list)."""
        result: list[Tracklet] = []
        for tracklets in self._active_targets.values():
            result.extend(tracklets)
        return result
