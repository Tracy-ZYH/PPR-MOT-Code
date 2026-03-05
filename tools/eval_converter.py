import os
import shutil

class TrackEvalConverter:
    """
    Utility to convert PPR-MOT inference results into the standard TrackEval format.
    Ensures compatibility with MOTChallenge evaluation protocols.
    """
    def __init__(self, output_root):
        self.output_root = output_root
        self.gt_root = os.path.join(output_root, "gt/train")
        self.tracker_root = os.path.join(output_root, "trackers/predict/data")
        self.seqmap_path = os.path.join(output_root, "seqmap.txt")

    def _ensure_dir(self, path):
        """Creates directory if it does not exist."""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def _format_seq_name(self, video_id, query_phrase):
        """Generates a standard sequence name by cleaning the query string."""
        clean_query = query_phrase.replace(' ', '_').replace('/', '_').replace('"', '')
        return f"{video_id}_{clean_query}"

    def convert(self, source_results_dir):
        """
        Walks through prediction results and organizes them for TrackEval.
        
        Args:
            source_results_dir (str): Path to your raw model outputs 
                                      (organized by video_id/query.txt).
        """
        print(f"Starting conversion to TrackEval format at: {self.output_root}")
        self._ensure_dir(self.gt_root)
        self._ensure_dir(self.tracker_root)

        sequences = []

        for video_id in sorted(os.listdir(source_results_dir)):
            video_path = os.path.join(source_results_dir, video_id)
            if not os.path.isdir(video_path):
                continue

            for file_name in sorted(os.listdir(video_path)):
                if not file_name.endswith(".txt") or file_name == "gt.txt":
                    continue

                query_name = file_name.replace(".txt", "")
                seq_name = self._format_seq_name(video_id, query_name)
                sequences.append(seq_name)

                pred_src = os.path.join(video_path, file_name)
                pred_dst = os.path.join(self.tracker_root, f"{seq_name}.txt")
                shutil.copy(pred_src, pred_dst)

                gt_src = os.path.join(video_path, "gt.txt") # Assuming per-query GT exists
                if os.path.exists(gt_src):
                    gt_seq_dir = os.path.join(self.gt_root, seq_name, "gt")
                    self._ensure_dir(gt_seq_dir)
                    shutil.copy(gt_src, os.path.join(gt_seq_dir, "gt.txt"))

        with open(self.seqmap_path, "w") as f:
            f.write("name\n") # Header
            for seq in sequences:
                f.write(f"{seq}\n")

        print(f"Successfully converted {len(sequences)} sequences.")
        print(f"Sequence map generated at: {self.seqmap_path}")

if __name__ == "__main__":
    # Example usage for manual execution
    # converter = TrackEvalConverter(output_root="TrackEval_Data")
    # converter.convert(source_results_dir="runs/exp30/results")
    pass