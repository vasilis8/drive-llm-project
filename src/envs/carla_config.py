"""
CARLA environment configuration defaults.

Provides helper functions to build sensor blueprints and spawn points
from the YAML config. Separated from carla_env.py to keep the env
wrapper clean.
"""

from typing import Dict, Any


# Default CARLA configuration (used if YAML not loaded)
DEFAULT_CARLA_CONFIG = {
    "host": "localhost",
    "port": 2000,
    "timeout": 30.0,
    "town": "Town04",
    "weather": "ClearNoon",
    "fixed_delta_seconds": 0.05,
}

DEFAULT_LIDAR_CONFIG = {
    "channels": 1,
    "range": 50.0,
    "points_per_second": 56000,
    "rotation_frequency": 20,
    "upper_fov": 0.0,
    "lower_fov": 0.0,
    "n_beams": 1080,
    "position": {"x": 0.0, "y": 0.0, "z": 2.4},
}

DEFAULT_VEHICLE_CONFIG = {
    "blueprint": "vehicle.tesla.model3",
    "max_speed": 40.0,
}


def get_lidar_attributes(lidar_config: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert lidar config dict into CARLA blueprint attribute strings.

    Args:
        lidar_config: LiDAR sensor configuration dict.

    Returns:
        Dict of attribute_name -> string_value for CARLA blueprint.
    """
    return {
        "channels": str(lidar_config.get("channels", 1)),
        "range": str(lidar_config.get("range", 50.0)),
        "points_per_second": str(lidar_config.get("points_per_second", 56000)),
        "rotation_frequency": str(lidar_config.get("rotation_frequency", 20)),
        "upper_fov": str(lidar_config.get("upper_fov", 0.0)),
        "lower_fov": str(lidar_config.get("lower_fov", 0.0)),
    }


# Weather presets (subset of CARLA weathers)
WEATHER_PRESETS = {
    "ClearNoon": "carla.WeatherParameters.ClearNoon",
    "CloudyNoon": "carla.WeatherParameters.CloudyNoon",
    "WetNoon": "carla.WeatherParameters.WetNoon",
    "ClearSunset": "carla.WeatherParameters.ClearSunset",
    "HardRainNoon": "carla.WeatherParameters.HardRainNoon",
}
