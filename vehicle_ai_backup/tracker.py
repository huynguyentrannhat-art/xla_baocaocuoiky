"""Ghép đối tượng đơn giản bằng IoU cho video (KHÔNG định danh tuyệt đối)."""
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass
class Track:
    bbox: tuple
    last_frame: int
    labels: deque = field(default_factory=lambda: deque(maxlen=7))
    colors: deque = field(default_factory=lambda: deque(maxlen=7))


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = max(0, ax2-ax1)*max(0, ay2-ay1) + max(0, bx2-bx1)*max(0, by2-by1) - intersection
    return intersection/union if union > 0 else 0.


class IoUTracker:
    def __init__(self, min_iou=.23, max_age=18):
        self.min_iou, self.max_age = min_iou, max_age
        self.tracks = {}
        self.next_id = 1
        self.frame = 0

    def update(self, detections):
        self.frame += 1
        candidates = []
        for di, det in enumerate(detections):
            for tid, track in self.tracks.items():
                score = iou(det.bbox, track.bbox)
                if score >= self.min_iou:
                    candidates.append((score, di, tid))
        used_d, used_t = set(), set()
        for score, di, tid in sorted(candidates, reverse=True):
            if di in used_d or tid in used_t:
                continue
            used_d.add(di); used_t.add(tid)
            self._assign(detections[di], tid)
        for di, det in enumerate(detections):
            if di not in used_d:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(det.bbox, self.frame)
                self._assign(det, tid)
        self.tracks = {tid: track for tid, track in self.tracks.items()
                       if self.frame - track.last_frame <= self.max_age}
        return detections

    def _assign(self, det, tid):
        track = self.tracks[tid]
        track.bbox = det.bbox
        track.last_frame = self.frame
        track.labels.append(det.label)
        if det.color != 'Không rõ' and not det.color.startswith('Không rõ'):
            track.colors.append(det.color)
        det.track_id = tid
        if len(track.labels) >= 3:
            det.label = Counter(track.labels).most_common(1)[0][0]
        if len(track.colors) >= 3:
            det.color = Counter(track.colors).most_common(1)[0][0]
