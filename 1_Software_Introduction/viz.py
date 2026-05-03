import numpy as np


def create_server():
    import viser

    server = viser.ViserServer()
    share_url = server.request_share_url()
    return server, share_url


def se3_to_vis_pose(T):
    import pinocchio as pin

    xyzquat = pin.SE3ToXYZQUAT(T).copy()
    position = xyzquat[:3]
    wxyz = np.array([xyzquat[6], xyzquat[3], xyzquat[4], xyzquat[5]])
    return position, wxyz


def update_object_pose(handle, T):
    position, wxyz = se3_to_vis_pose(T)
    handle.position = tuple(position)
    handle.wxyz = tuple(wxyz)
    return handle


def add_mesh_handle(server, name, mesh, T):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    color = (90, 200, 255)
    opacity = None
    visual = getattr(mesh, "visual", None)
    face_colors = getattr(visual, "face_colors", None) if visual is not None else None
    if face_colors is not None and len(face_colors) > 0:
        rgba = np.asarray(face_colors[0]).astype(float)
        color = tuple(rgba[:3])
        if rgba.shape[0] >= 4:
            opacity = float(rgba[3]) / 255.0

    position, wxyz = se3_to_vis_pose(T)
    return server.scene.add_mesh_simple(
        name=name,
        vertices=vertices,
        faces=faces,
        color=color,
        opacity=opacity,
        position=position,
        wxyz=wxyz,
    )


class PinNotebookViz:
    def __init__(self, server):
        self.server = server
        self.robot_viz = None
        self.frame_handles = {}

    def attach_robot(self, robot, root_name="pinocchio"):
        from pinocchio.visualize import ViserVisualizer

        self.server.scene.reset()
        viz = ViserVisualizer(robot.model, robot.collision_model, robot.visual_model)
        viz.initViewer(viewer=self.server)
        viz.loadViewerModel(rootNodeName=root_name)
        self.robot_viz = viz
        self.frame_handles = {}
        return viz

    def display(self, q):
        if self.robot_viz is None:
            raise RuntimeError("Attach a robot first with notebook_viz.attach_robot(robot).")
        self.robot_viz.display(q)

    def show_frame(self, name, T, axes_length=0.16, axes_radius=0.008):
        position, wxyz = se3_to_vis_pose(T)
        handle = self.frame_handles.get(name)
        if handle is None:
            handle = self.server.scene.add_frame(
                name=f"/frames/{name}",
                axes_length=axes_length,
                axes_radius=axes_radius,
            )
            self.frame_handles[name] = handle
        handle.position = tuple(position)
        handle.wxyz = tuple(wxyz)
        return handle

    def clear_frames(self):
        for handle in self.frame_handles.values():
            handle.remove()
        self.frame_handles = {}
