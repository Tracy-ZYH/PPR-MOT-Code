from .experts import GroundingDINOExpert, QwenVLExpert

class UnifiedDetector:
    """
    Perception Module (PAM) Orchestrator.
    Dynamically routes queries to experts based on the Planning stage strategy.
    """
    def __init__(self, gd_config, gd_weights, qwen_client, device="cuda"):
        # Initialize strategies
        self.easy_expert = GroundingDINOExpert(gd_config, gd_weights, device)
        self.hard_expert = QwenVLExpert(qwen_client)

    def run_grounding(self, image_path, phrases, strategy="easy"):
        """
        Executes grounding on an image frame.
        Args:
            image_path (str): Path to frame.
            phrases (list): List of decomposed static appearance phrases.
            strategy (str): 'easy' or 'hard' assigned by the QPM.
        """
        if not phrases:
            return []
            
        if strategy == "easy":
            return self.easy_expert.predict(image_path, phrases)
        else:
            return self.hard_expert.predict(image_path, phrases)