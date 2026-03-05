import json
import re
import yaml
from openai import OpenAI
from .kinematic import extract_motion_summary

class CognitiveReasoner:
    """Validates trajectories against behavioral constraints (CRM)."""

    def __init__(self, api_key, base_url, template_path="config/reasoning_templates.yaml", model_name="qwen-max"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        with open(template_path, 'r') as f:
            self.templates = yaml.safe_load(f)
        self.prompt_template = self.templates['crm_template']['prompt']

    def validate(self, track_id, track_data, constraints):
        """Judgment logic for trajectory-query alignment."""
        motion_data = extract_motion_summary(track_data)
        sampled_coords = track_data[-10:] 
        
        prompt = self.prompt_template.format(
            contextual_details=constraints,
            kinematic_data=json.dumps(motion_data),
            track_coordinates=sampled_coords
        )

        try:
            messages = [
                {"role": "system", "content": self.templates['crm_template']['system_role']},
                {"role": "user", "content": prompt}
            ]
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            
            response_text = completion.choices[0].message.content
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                decision = json.loads(match.group(0))
                return decision.get("is_match", False)
        except Exception as e:
            print(f"CRM Error for track {track_id}: {e}")
            
        return False