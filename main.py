import argparse
import os
import yaml
import json
from tqdm import tqdm

from modules.query_planning.planner import QueryPlanner
from modules.perception_association.detector import PerceptionExpert
from modules.perception_association.tracker import AssociationEngine
from modules.cognitive_reasoning.reasoner import CognitiveReasoner

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline(video_id, query, cfg):
    # Planning
    planner = QueryPlanner(
        api_key=cfg['api']['deepseek_key'], 
        base_url=cfg['api']['deepseek_base']
    )
    plan = planner.generate_plan(query)
    
    # Perception 
    detector = PerceptionExpert(
        gd_config=cfg['models']['gd_config'],
        gd_weights=cfg['models']['gd_weights'],
        device="cuda"
    )
    tracker = AssociationEngine(
        track_thresh=cfg['pam']['track_threshold'],
        match_thresh=cfg['pam']['match_threshold']
    )

    # Replace this placeholder logic with your specific frame-loading loop
    candidate_trajectories = [] 

    # Save intermediate trajectories for transparency
    save_dir = "outputs/intermediate_trajectories"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{video_id}_candidates.json"), 'w') as f:
        json.dump(candidate_trajectories, f)

    # Reasoning 
    reasoner = CognitiveReasoner(
        model_name=cfg['crm']['reasoning_model']
    )
    
    final_verified_tracks = []
    for track in tqdm(candidate_trajectories, desc="CRM Reasoning"):
        if reasoner.validate(track['id'], track['data'], plan.get('temporal_semantics')):
            final_verified_tracks.append(track)

    return final_verified_tracks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPR-MOT Pipeline")
    parser.add_argument("--video_id", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_pipeline(args.video_id, args.query, config)
    
    print(f"Tracking finished. {len(results)} objects validated.")