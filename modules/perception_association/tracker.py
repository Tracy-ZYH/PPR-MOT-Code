import numpy as np
from yolox.tracker.byte_tracker import BYTETracker

class ByteTrackArgs:
    """
    Simulation of argparse arguments required by BYTETracker.
    Parameters are aligned with the paper's implementation details.
    """
    def __init__(self, track_thresh=0.35, track_buffer=30, match_thresh=0.8, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20

class MOTTracker:
    """
    Association Module (PAM).
    Responsible for maintaining identity consistency across frames using ByteTrack logic.
    """
    def __init__(self, track_thresh=0.35, track_buffer=30, match_thresh=0.8):
        args = ByteTrackArgs(track_thresh, track_buffer, match_thresh)
        self.tracker = BYTETracker(args)

    def _format_detections(self, detections):
        """
        Converts dictionary-based detections into the [x1, y1, x2, y2, score] 
        numpy format required by ByteTrack.
        """
        if not detections:
            return np.empty((0, 5))
            
        formatted_dets = []
        for det in detections:
            # Extract [xmin, ymin, xmax, ymax] and confidence
            bbox = det.get('bbox_xyxy')
            conf = det.get('confidence', 0.0)
            formatted_dets.append(bbox + [conf])
            
        return np.array(formatted_dets)

    def update(self, detections, frame_size):
        """
        Updates the tracker with detections from the current frame.
        
        Args:
            detections (list): Results from UnifiedDetector.
            frame_size (tuple): (height, width) of the current frame.
            
        Returns:
            list: A list of active track objects containing track_id and bbox.
        """
        dets_np = self._format_detections(detections)
        
        online_targets = self.tracker.update(dets_np, frame_size, frame_size)
        
        active_tracks = []
        for t in online_targets:
            active_tracks.append({
                "track_id": t.track_id,
                "bbox_xyxy": t.tlbr.tolist(), # tlbr is [x1, y1, x2, y2]
                "confidence": t.score
            })
            
        return active_tracks