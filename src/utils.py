import os


def _base_dir():
    return os.path.join(os.path.dirname(__file__), "..")


def get_data_path(site: str, filename: str) -> str:
    base = _base_dir()
    upload_path = os.path.join(base, "uploads", site, filename)
    if os.path.exists(upload_path):
        return upload_path
    return os.path.join(base, f"{site}_data", filename)


def get_config_path() -> str:
    base = _base_dir()
    upload_config = os.path.join(base, "uploads", "config.json")
    if os.path.exists(upload_config):
        return upload_config
    config_path = os.path.join(base, "config.json")
    if os.path.exists(config_path):
        return config_path
    return os.path.join(base, "sample.config.json")
