def get_camera_info() -> dict:
    """Data in mm"""
    return {
        "sensor_size_x": 5.333,
        "sensor_size_y": 3,
        "focal_length": 3.6
    }

def get_classes() -> list:
    """
    Use: information, mandatory, priority, prohibitory, stop, warning, yield
    Size is the length of the shorter side of the bounding rectangle, in mm
    """
    return [
        {"name": "information", "size": 600, "color": (255, 153, 51)},  # <- good
        {"name": "information_add", "size": None, "color": (0, 255, 0)}, 
        {"name": "information_big", "size": None, "color": (0, 255, 0)}, 
        {"name": "mandatory", "size": 600, "color": (0, 255, 0)}, # <- good
        {"name": "priority", "size": 848, "color": (51, 255, 255)}, # <- good
        {"name": "prohibitory", "size": 600,
            "color": (102, 102, 255)},  # <- good
        {"name": "prohibitory_big", "size": None, "color": (0, 255, 0)}, 
        {"name": "stop", "size": 600, "color": (10 ,10, 255)}, # <- good
        {"name": "warning", "size": 650, "color": (10, 153, 153)}, # <- good
        {"name": "warning_add", "size": None, "color": (0, 204, 204)}, 
        {"name": "yield", "size": 650, "color": (153, 255, 255)} # <- good
    ]