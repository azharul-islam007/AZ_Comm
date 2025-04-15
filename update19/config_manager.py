# config_manager.py
import yaml
import os


class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = {}
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config: {e}")

    def get(self, key, default=None):
        """Get config value with fallback to default"""
        # Support nested keys with dot notation: "physical.snr_db"
        if "." in key:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current

        return self.config.get(key, default)

    def set(self, key, value):
        """Set config value"""
        self.config[key] = value

    def save(self):
        """Save config to file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False