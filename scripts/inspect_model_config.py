import json
from transformers import AutoConfig

def inspect_config():
    model_path = "manipulation_tactic_detector_model"
    print(f"Inspecting config at {model_path}")
    
    try:
        config = AutoConfig.from_pretrained(model_path)
        print(f"Num Labels: {config.num_labels}")
        print("ID2Label Keys:", list(config.id2label.keys()))
        print("ID2Label Values:", list(config.id2label.values()))
    except Exception as e:
        print(f"Error loading config: {e}")

if __name__ == "__main__":
    inspect_config()
