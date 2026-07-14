import numpy as np

def make_transform(translation, rotation=None):
    import coal

    if rotation is None:
        rotation = np.eye(3)
    T = coal.Transform3s()
    T.setRotation(np.asarray(rotation, dtype=float))
    T.setTranslation(np.asarray(translation, dtype=float))
    return T


def coal_to_se3(T):
    import pinocchio as pin

    return pin.SE3(T.getRotation(), T.getTranslation())

def rotation_from_x_axis(direction):
    x_axis = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(x_axis)
    if norm < 1e-12:
        return np.eye(3)
    x_axis = x_axis / norm
    reference = np.array([0.0, 0.0, 1.0]) if abs(x_axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(reference, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])
