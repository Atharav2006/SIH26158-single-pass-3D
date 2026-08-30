import bisect
from typing import List, Dict, Any, Optional

class TemporalSynchronizer:
    """
    Nearest-neighbor temporal association utility for multi-sensor streams.
    Synchronizes discrete sensor records (GPS, IMU, Pose) to query timestamps (e.g. Image frames).
    """

    @staticmethod
    def find_nearest(
        target_ts: float,
        stream: List[Dict[str, Any]],
        timestamp_key: str = "timestamp_seconds",
        max_tolerance: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find the nearest item in a time-sorted stream to target_ts.

        Args:
            target_ts: Query timestamp in seconds.
            stream: List of records containing a timestamp key, sorted ascending.
            timestamp_key: Field name containing the timestamp.
            max_tolerance: Maximum allowable absolute time difference in seconds.

        Returns:
            Dict containing source_timestamp, matched_timestamp, absolute time_difference, and matched record.
        """
        if not stream:
            return None

        # Extract timestamps for binary search
        timestamps = [item[timestamp_key] for item in stream]
        idx = bisect.bisect_left(timestamps, target_ts)

        candidates = []
        if idx < len(stream):
            candidates.append(stream[idx])
        if idx > 0:
            candidates.append(stream[idx - 1])

        if not candidates:
            return None

        best_match = min(candidates, key=lambda item: abs(item[timestamp_key] - target_ts))
        dt = abs(best_match[timestamp_key] - target_ts)

        if max_tolerance is not None and dt > max_tolerance:
            return None

        return {
            "source_timestamp": target_ts,
            "matched_timestamp": best_match[timestamp_key],
            "time_difference": round(dt, 6),
            "record": best_match
        }

    @staticmethod
    def synchronize(
        reference_stream: List[Dict[str, Any]],
        target_stream: List[Dict[str, Any]],
        ref_key: str = "timestamp_seconds",
        target_key: str = "timestamp_seconds",
        max_tolerance: Optional[float] = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Synchronize every reference record with its nearest neighbor in target_stream.
        """
        results = []
        for ref in reference_stream:
            target_ts = ref[ref_key]
            match_info = TemporalSynchronizer.find_nearest(
                target_ts=target_ts,
                stream=target_stream,
                timestamp_key=target_key,
                max_tolerance=max_tolerance
            )
            if match_info is not None:
                results.append({
                    "reference": ref,
                    "matched": match_info["record"],
                    "source_timestamp": target_ts,
                    "matched_timestamp": match_info["matched_timestamp"],
                    "time_difference": match_info["time_difference"]
                })
        return results
