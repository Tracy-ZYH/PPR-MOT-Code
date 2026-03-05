import numpy as np

def extract_kinematic_features(track_data, fps=10):
    """Calculates motion dynamics including speed, acceleration, and angle changes (Eq 4-8)."""
    if len(track_data) < 3:
        return None
    
    # Coordinates extraction
    pts = [np.array([t[1]+t[3]/2, t[2]+t[4]/2]) for t in track_data]
    
    # Velocity
    v_t = (pts[-1] - pts[-2]) * fps
    v_prev = (pts[-2] - pts[-3]) * fps
    
    # Speed magnitude
    speed = np.linalg.norm(v_t)
    
    # Acceleration Projection
    accel_vec = v_t - v_prev
    accel_proj = np.dot(accel_vec, v_t) / (speed + 1e-6)
    
    # Turning Angle
    cos_theta = np.dot(v_t, v_prev) / (np.linalg.norm(v_t) * np.linalg.norm(v_prev) + 1e-6)
    angle_change = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    return {
        "speed_pixels_per_sec": round(speed, 2),
        "acceleration_projection": round(accel_proj, 2),
        "angle_change_degrees": round(angle_change, 2)
    }