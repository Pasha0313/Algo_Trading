import json

file_path = "strategies_config.json"

with open(file_path, "r") as f:
    data = json.load(f)

risk_block = {
    "stop_loss_pct": [0.01, 0.06, 0.01],
    "take_profit_pct": [0.02, 0.12, 0.02]
}

for strategy in data.values():
    strategy["risk_param_ranges"] = risk_block

with open(file_path, "w") as f:
    json.dump(data, f, indent=2)

print("✅ All strategies updated with risk_param_ranges")