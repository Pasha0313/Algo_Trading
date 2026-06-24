import json

class LoadingForecastModel:
    def __init__(self, config_file):
        """Load forecasting model configuration from a JSON file"""
        with open(config_file, "r") as file:
            self.models = json.load(file)

    def get_model_config(self, model_name):
        """Fetch the model config by its name"""
        return self.models.get(model_name, None)

    def process_model(self, model_name):
        """
        Process and return the model configuration without converting ranges.
        """
        # Fetch the model configuration from your JSON or dictionary
        model_config = self.get_model_config(model_name)

        if model_config is None:
            raise ValueError(f"Model '{model_name}' not found.")

        # Extract description, parameters, and parameter ranges
        description = model_config.get("description", "")
        parameters = model_config.get("parameters", {})
        param_ranges = {}

        # Keep parameter ranges as original lists
        for key, value in model_config.get("param_ranges", {}).items():
            if isinstance(value, list):
                param_ranges[key] = value  # Retain original list format
            else:
                raise ValueError(f"Invalid format for parameter range '{key}': {value}")

        return description, parameters, param_ranges

    def print_model_details(self, model_name):
        """Print the model details, including parameters and parameter ranges"""
        model_config = self.get_model_config(model_name)

        if model_config is None:
            raise ValueError(f"Model '{model_name}' not found.")

        # Print the model description
        description = model_config["description"]
        print(f"Description: {description}\n")

        # Print parameters
        parameters = model_config.get("parameters", {})
        print(f"Parameters: {parameters}")

        # Print parameter ranges
        param_ranges = model_config.get("param_ranges", {})
        print("Parameter Ranges:")
        for key, value in param_ranges.items():
            print(f"  {key}: {value}")
        print("\n")            
