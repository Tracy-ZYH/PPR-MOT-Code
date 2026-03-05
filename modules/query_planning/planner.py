import json
import yaml
from openai import OpenAI

class QueryPlanner:
    """Decomposes natural language queries into structured plans (QPM)."""
    
    def __init__(self, api_key, base_url, template_path="config/reasoning_templates.yaml"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        with open(template_path, 'r') as f:
            self.templates = yaml.safe_load(f)
        self.prompt_template = self.templates['qpm_template']['prompt']

    def generate_plan(self, query, model="deepseek-chat"):
        """Generates a structured plan using LLM."""
        prompt = self.prompt_template.format(user_query=query)
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"QPM Error: {e}")

            return {
                "appearance_semantics": query, 
                "spatial_semantics": "", 
                "temporal_semantics": "", 
                "expert_strategy": "hard"
            }