import json

class StrategyLoader:
    def __init__(self, config_file):
        # Load strategy configuration from a JSON file
        with open(config_file, "r") as file:
            self.strategies = json.load(file)

    def get_strategy_config(self, strategy_name):
        """Fetch the strategy config by its name"""
        return self.strategies.get(strategy_name, None)

    def process_strategy(self, strategy_name, Print_Data=False):
        """Process and return the strategy config in a suitable format."""
        strategy_config = self.get_strategy_config(strategy_name)

        if strategy_config is None:
            raise ValueError(f"Strategy '{strategy_name}' not found.")

        description = strategy_config["description"]
        parameters = tuple(strategy_config["parameters"])

        if Print_Data:
            print("CheckUP : ", parameters)

        param_ranges = {}

        def expand_range(value):
            start, stop, step = value

            if step == 0:
                raise ValueError(f"Step cannot be zero for range: {value}")

            if all(isinstance(v, int) for v in value):
                return range(start, stop, step)

            elif all(isinstance(v, (int, float)) for v in value):
                n = int((stop - start) / step)
                return [round(start + i * step, 10) for i in range(n)]

            else:
                raise TypeError(f"Invalid range values: {value}")

        for key, value in strategy_config.get("param_ranges", {}).items():
            param_ranges[key] = expand_range(value)

        for key, value in strategy_config.get("risk_param_ranges", {}).items():
            param_ranges[key] = expand_range(value)

        return description, parameters, param_ranges
    
    def print_strategy_details(self, strategy_name):
        """Print the strategy details, including parameters"""
        strategy_config = self.get_strategy_config(strategy_name)
       
        # Get the description, parameters, and ranges
        parameters = tuple(strategy_config["parameters"])

        # Print parameters in the desired format: Key1: Value1, Key2: Value2, ...
        param_names = list(self.strategies[strategy_name]["param_ranges"].keys())
        parameter_string = ", ".join([f"{param_names[i]}: {parameters[i]}" for i in range(len(parameters))])
        
        print(f"Parameters: {parameter_string}")
