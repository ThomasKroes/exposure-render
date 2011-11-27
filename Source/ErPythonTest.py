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
Reader.SetFileName("C:/Volumes/macoessix_small.mhd")
Reader.Update()

# Exposure Rendererder volume mapper
ErVolumeMapper = vtkErCorePython.vtkErVolumeMapper()

ErVolumeMapper.SetInput(Reader.GetOutput())

Volume.SetMapper(ErVolumeMapper)

# Exposure Rendererder volume property
ErVolumeProperty = vtkErCorePython.vtkErVolumeProperty()

Opacity = vtk.vtkPiecewiseFunction()
Opacity.AddPoint(0, 0.000)
Opacity.AddPoint(100, 0)
Opacity.AddPoint(2000, 1)
Opacity.AddPoint(2055, 1)

ErVolumeProperty.SetOpacity(Opacity)
ErVolumeProperty.SetStepSizeFactorPrimary(1.0)
ErVolumeProperty.SetStepSizeFactorSecondary(1.0)
ErVolumeProperty.SetDensityScale(100)

# Assign the ER volume 
Volume.SetProperty(ErVolumeProperty)

Renderer.AddVolume(Volume)

# Create ER camera
ErCamera = vtkErCorePython.vtkErCamera()
ErCamera.SetRenderer(Renderer)
ErCamera.SetFocalDisk(0)
ErCamera.SetPosition(150, 150, 150)
ErCamera.SetFocalPoint(75, 75, 75)
Renderer.SetActiveCamera(ErCamera)
Renderer.ResetCamera()
	
# First remove all lights
Renderer.RemoveAllLights()

# Configure the light
ErAreaLight = vtkErCorePython.vtkErAreaLight()
ErAreaLight.SetPosition(0, -5000, 0);
ErAreaLight.SetFocalPoint(300, 300, 300);
ErAreaLight.SetColor(1000000, 1000000, 1000000);
ErAreaLight.SetPositional(1);
ErAreaLight.SetSize(0.001, 0.001, 0.001)

# Add the area light to the Renderer
Renderer.AddLight(ErAreaLight);

ErBackgroundLight = vtkErCorePython.vtkErBackgroundLight();
ErBackgroundLight.SetDiffuseColor(500000, 500000, 1500000);

# Add the background light to the Renderer
Renderer.AddLight(ErBackgroundLight);

# SlicePlane = vtkErCorePython.vtkErSlicePlane()
# Renderer.AddViewProp(SlicePlane)

ErBoxWidget = vtkErCorePython.vtkErSliceBoxWidget()
#PlaneWidget.SetPlaceFactor(10000)
ErBoxWidget.SetInteractor(Interactor)
# PlaneWidget.SetOrigin(0, 0, 0)
# PlaneWidget.SetPoint1(100, 0, 0)
# PlaneWidget.SetPoint2(0, 100, 0)
ErBoxWidget.SetVolume(Volume)

#laneWidget.PlaceWidget(0, 0, 0, 150, 150, 150)
#PlaneWidget.UpdatePlacement()

ErVolumeMapper.SetSliceWidget(ErBoxWidget)

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
#PlaneWidget.GetPlaneActor().AddObserver("ModifiedEvent", TimeOut)

def CheckAbort(obj, event):
    obj.SetAbortRender(5)
    
def CheckKeyPress(obj, event) :
    if (obj.GetKeySym() == "Escape"):
        print "Terminating app..."
        obj.GetRenderWindow().Finalize();
        obj.TerminateApp()
        
RendererWin.AddObserver("AbortCheckEvent", CheckAbort)
RendererWin.GetInteractor().AddObserver("KeyPressEvent", CheckKeyPress)

Interactor.Initialize()

RendererWin.Render()

PlaneWidget.On()

# Start the event loop.
Interactor.Start()


