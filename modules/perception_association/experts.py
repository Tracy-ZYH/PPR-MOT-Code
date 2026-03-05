import torch
import base64
import re
import json
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.vl_utils import create_positive_map_from_span

class GroundingDINOExpert:
    """Lightweight expert for simple appearance attributes."""
    def __init__(self, config_path, checkpoint_path, device="cuda"):
        args = SLConfig.fromfile(config_path)
        args.device = device
        self.model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path, phrases, threshold=0.2):
        image_pil = Image.open(image_path).convert("RGB")
        image_tensor, _ = self.transform(image_pil, None)
        caption = " . ".join(phrases)
        
        with torch.no_grad():
            outputs = self.model(image_tensor[None].to(self.device), captions=[caption])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        
        # Mapping logits to phrases
        tokenized = self.model.tokenizer(caption)
        text_spans = []
        start = 0
        for part in caption.split(' . '):
            end = start + len(part)
            text_spans.append([(start, end)])
            start = end + 3
        positive_maps = create_positive_map_from_span(tokenized, text_spans)
        logits_for_phrases = positive_maps @ logits.T

        results = []
        for i, phrase in enumerate(phrases):
            scores = logits_for_phrases[i]
            mask = scores > threshold
            for box, conf in zip(boxes[mask], scores[mask]):
                # Rescale to pixel coordinates
                box_px = box * torch.Tensor([image_pil.width, image_pil.height, image_pil.width, image_pil.height])
                xyxy = box_ops.box_cxcywh_to_xyxy(box_px).tolist()
                results.append({
                    "phrase": phrase, "confidence": float(conf),
                    "bbox_xyxy": [int(c) for c in xyxy]
                })
        return results

class QwenVLExpert:
    """Heavyweight expert for fine-grained and complex appearance attributes."""
    def __init__(self, client, model_name="qwen-vl-max"):
        self.client = client
        self.model_name = model_name

    def _encode_image(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def predict(self, image_path, phrases):
        img_base64 = self._encode_image(image_path)
        prompt = "Identify objects matching these descriptions: " + ", ".join(phrases) + \
                 ". Return a JSON list: [{'phrase': str, 'confidence': float, 'bbox_xyxy': [xmin, ymin, xmax, ymax]}]."
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    {"type": "text", "text": prompt}
                ]}],
                temperature=0.0
            )
            content = completion.choices[0].message.content
            # Extraction logic for JSON inside markdown blocks
            match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            return json.loads(match.group(1) if match else content)
        except Exception as e:
            print(f"Qwen-VL Error: {e}")
            return []