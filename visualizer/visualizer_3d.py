import pyximport; pyximport.install()
import numpy as np
import vtk
from .util import *

class Visualizer(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, h=800, w=600):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.1)

        # Add axes
        axes = vtk.vtkAxesActor()
        axes.GetXAxisCaptionActor2D().SetHeight(0.05)
        axes.GetYAxisCaptionActor2D().SetHeight(0.05)
        axes.GetZAxisCaptionActor2D().SetHeight(0.05)
        axes.SetCylinderRadius(0.03)
        axes.SetShaftTypeToCylinder()
        self.renderer.AddActor(axes)

        # Add render window
        self.renwin = vtk.vtkRenderWindow()
        self.renwin.SetWindowName("Point Cloud Viewer")
        self.renwin.SetSize(h, w)
        self.renwin.AddRenderer(self.renderer)

        # An interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        interstyle = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(interstyle)
        self.interactor.SetRenderWindow(self.renwin)

        self.camera_actors = []

    def bind_keyboard_event(self, event_func):
        """
        Bind the keyboard event
        :param event_func: event define function
        Function Example:

        def keyPressEvent(self, obj, event):
            key = obj.GetKeySym()
            print(key)
            return
        """
        self.interactor.AddObserver("KeyPressEvent", event_func)

    def add_frame_pose(self, R, t, color=(1.0, 1.0, 1.0), camera_obj_scale=0.25):
        """
        mat = [R|t]
        add a keyframe camera model
        """
        camera_actor = create_camera_actor(R, t, color, camera_obj_scale)
        self.renderer.AddActor(camera_actor)
        self.camera_actors.append(camera_actor)

        # Update rendering
        self.renwin.Render()

    def clear_frame_poses(self):
        for camera_actor in self.camera_actors:
            self.renderer.RemoveActor(camera_actor)

    def set_point_cloud(self, points, colors=None, pt_size=3):
        if hasattr(self, 'pointcloud_actor'):
            self.renderer.RemoveActor(self.pointcloud_actor)
        if colors is not None:
            colors = np.clip(np.uint8(colors * 255 + 0.5), 0, 255)
        self.pointcloud_actor = create_pointcloud_actor(points, colors, pt_size)
        self.renderer.AddActor(self.pointcloud_actor)

        # Update rendering
        self.renwin.Render()

    def show(self):
        self.interactor.Initialize()
        self.interactor.Start()

    def close(self):
        self.renwin.Finalize()
        self.interactor.TerminateApp()
        del self.renwin, self.interactor