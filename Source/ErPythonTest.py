import vtk
import vtkErCorePython

# print vtkErCorePython

# The colors module defines various useful colors.
from vtk.util.colors import tomato

Renderer = vtk.vtkRenderer()
RendererWin = vtk.vtkRenderWindow()
RendererWin.AddRenderer(Renderer)
Interactor = vtk.vtkRenderWindowInteractor()
Interactor.SetRenderWindow(RendererWin)

Volume = vtk.vtkVolume()

# Read volume
Reader = vtk.vtkMetaImageReader()
Reader.SetFileName("C:/Volumes/bonsai_small.mhd")
Reader.Update()

# Exposure Rendererder volume mapper
ErVolumeMapper = vtkErCorePython.vtkErVolumeMapper()

ErVolumeMapper.SetInput(Reader.GetOutput())

Volume.SetMapper(ErVolumeMapper)

# Exposure Rendererder volume property
ErVolumeProperty = vtkErCorePython.vtkErVolumeProperty()

Opacity = vtk.vtkPiecewiseFunction()
Opacity.AddPoint(0, 0)
Opacity.AddPoint(8, 0)
Opacity.AddPoint(15, 1)
Opacity.AddPoint(255, 1)

ErVolumeProperty.SetOpacity(Opacity)
ErVolumeProperty.SetStepSizeFactorPrimary(1.0)
ErVolumeProperty.SetStepSizeFactorSecondary(2.0)

# Assign the ER volume 
Volume.SetProperty(ErVolumeProperty)

Renderer.AddVolume(Volume)

# Create ER camera
ErCamera = vtkErCorePython.vtkErCamera()
ErCamera.SetRenderer(Renderer)
ErCamera.SetFocalDisk(0)
Renderer.SetActiveCamera(ErCamera)
Renderer.ResetCamera()
	
# First remove all lights
Renderer.RemoveAllLights()

# Configure the light
ErAreaLight = vtkErCorePython.vtkErAreaLight()
ErAreaLight.SetPosition(0, -1.5, 0);
ErAreaLight.SetFocalPoint(0, 0, 0);
ErAreaLight.SetColor(1000, 1000, 1000);
ErAreaLight.SetPositional(1);
ErAreaLight.SetSize(0.001, 0.001, 0.001)

# Add the area light to the Renderer
Renderer.AddLight(ErAreaLight);

ErBackgroundLight = vtkErCorePython.vtkErBackgroundLight();
ErBackgroundLight.SetDiffuseColor(100000, 100000, 100000);

# Add the background light to the Renderer
Renderer.AddLight(ErBackgroundLight);

# SlicePlane = vtkErCorePython.vtkErSlicePlane()
# Renderer.AddViewProp(SlicePlane)

PlaneWidget = vtkErCorePython.vtkErSlicePlaneWidget()
PlaneWidget.SetInteractor(Interactor)
PlaneWidget.On()
PlaneWidget.PlaceWidget(-150, 150, -150, 150, 0, 0)

InteractorStyle = vtk.vtkInteractorStyleTrackballCamera()
Interactor.SetInteractorStyle(InteractorStyle)

Renderer.SetBackground(0, 0, 0)
RendererWin.SetSize(600, 600)
RendererWin.Render()

def TimeOut(obj, event):
    Interactor.Render()

# ER requires progressive updates, so create a repeating timer and Rendererder when it times out
RendererWin.GetInteractor().AddObserver("TimerEvent", TimeOut)
RendererWin.GetInteractor().CreateRepeatingTimer(1)
PlaneWidget.GetPlaneActor().AddObserver("ModifiedEvent", TimeOut)

def CheckAbort(obj, event):
    if obj.GetEventPending() != 0:
        obj.SetAbortRender(1)



RendererWin.AddObserver("AbortCheckEvent", CheckAbort)

Interactor.Initialize()

RendererWin.Render()

# Start the event loop.
Interactor.Start()
