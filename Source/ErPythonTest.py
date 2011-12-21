import vtk
import vtkErCorePython

print vtkErCorePython

# The colors module defines various useful colors.

VolumeFile = "C:/volumes/engine_small.mhd"

Renderer = vtk.vtkRenderer()
RendererWin = vtk.vtkRenderWindow()
RendererWin.AddRenderer(Renderer)
Interactor = vtk.vtkRenderWindowInteractor()
Interactor.SetRenderWindow(RendererWin)

Volume = vtk.vtkVolume()

# Read volume
Reader = vtk.vtkMetaImageReader()
Reader.SetFileName(VolumeFile)

# if Reader.CanReadFile(VolumeFile) == 0:
#    Interactor.TerminateApp()

Reader.Update()

# Exposure Rendererder volume mapper
ErVolumeMapper = vtkErCorePython.vtkErVolumeMapper()

Volume.SetMapper(ErVolumeMapper)

# Exposure Rendererder volume property
ErVolumeProperty = vtkErCorePython.vtkErVolumeProperty()

Opacity = vtk.vtkPiecewiseFunction()
Opacity.AddPoint(0, 0)
Opacity.AddPoint(10, 0)
Opacity.AddPoint(11, 1)
Opacity.AddPoint(255, 1)

ErVolumeProperty.SetOpacity(Opacity)

# Diffuse
DiffuseR = vtk.vtkPiecewiseFunction()
DiffuseG = vtk.vtkPiecewiseFunction()
DiffuseB = vtk.vtkPiecewiseFunction()

DiffuseR.AddPoint(0, 1)
DiffuseR.AddPoint(255, 1)
DiffuseG.AddPoint(0, 1)
DiffuseG.AddPoint(255, 1)
DiffuseB.AddPoint(0, 1)
DiffuseB.AddPoint(255, 1)

ErVolumeProperty.SetDiffuse(0, DiffuseR)
ErVolumeProperty.SetDiffuse(1, DiffuseG)
ErVolumeProperty.SetDiffuse(2, DiffuseB)

# Specular
Specular = vtk.vtkPiecewiseFunction()

Specular.AddPoint(0, 1)
Specular.AddPoint(255, 1)

ErVolumeProperty.SetSpecular(0, Specular)
ErVolumeProperty.SetSpecular(1, Specular)
ErVolumeProperty.SetSpecular(2, Specular)

# Glossiness
Glossiness = vtk.vtkPiecewiseFunction()
Glossiness.AddPoint(0, 15)
Glossiness.AddPoint(255, 15)
ErVolumeProperty.SetGlossiness(Glossiness)

# IOR
IOR = vtk.vtkPiecewiseFunction()
IOR.AddPoint(0, 5)
IOR.AddPoint(255, 5)
ErVolumeProperty.SetIOR(IOR)

ErVolumeProperty.SetStepSizeFactorPrimary(1)
ErVolumeProperty.SetStepSizeFactorSecondary(1)
ErVolumeProperty.SetDensityScale(10000000)
ErVolumeProperty.SetShadingType(1)

# Assign the ER volume 
Volume.SetProperty(ErVolumeProperty)

ErVolumeMapper.SetInput(Reader.GetOutput())

Renderer.AddVolume(Volume)

# Create ER camera
ErCamera = vtkErCorePython.vtkErCamera()
ErCamera.SetRenderer(Renderer)
ErCamera.SetFocalDisk(0)
ErCamera.SetPosition(2, 2, 2)
ErCamera.SetFocalPoint(0.5, 0.5, 0.5)
ErCamera.SetExposure(0.1)
ErCamera.SetViewFront()
Renderer.SetActiveCamera(ErCamera)

	
# First remove all lights
Renderer.RemoveAllLights()

Fill = vtkErCorePython.vtkErAreaLight()
Fill.SetType(1)
Fill.SetPosition(75, 1, 1)
Fill.SetFocalPoint(0, 0, 0)
Fill.SetColor(6000, 6000, 6000)
Fill.SetPositional(1);
Fill.SetShapeType(1)
Fill.SetOneSided(1)
Fill.SetDistance(2)
#Fill.SetScale(100, 100, 10)
Fill.SetElevation(45)
Fill.SetAzimuth(45)
Fill.SetInnerRadius(90)
Fill.SetOuterRadius(100)
Fill.SetSize(500, 500, 500)
#Fill.SetCamera(ErCamera)

ErVolumeMapper.AddLight(Fill)

ErBackgroundLight = vtkErCorePython.vtkErBackgroundLight();
ErBackgroundLight.SetDiffuseColor(0.1, 0.1, 0.1);

ErBoxWidget = vtkErCorePython.vtkErBoxWidget()
ErBoxWidget.SetPlaceFactor(1)
ErBoxWidget.SetInteractor(Interactor)
# PlaneWidget.SetOrigin(0, 0, 0)
# PlaneWidget.SetPoint1(100, 0, 0)
# PlaneWidget.SetPoint2(0, 100, 0)
ErBoxWidget.SetVolume(Volume)

#laneWidget.PlaceWidget(0, 0, 0, 150, 150, 150)
#PlaneWidget.UpdatePlacement()

ErVolumeMapper.SetSliceWidget(ErBoxWidget)

axes = vtk.vtkAxesActor()

widget = vtk.vtkOrientationMarkerWidget()

widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
widget.SetOrientationMarker(axes)
widget.SetInteractor(Interactor)
widget.SetViewport(0.0, 0.0, 0.2, 0.2)
widget.SetEnabled(1)
#widget.InteractiveOn()

# Camera widget
CameraWidget = vtkErCorePython.vtkErCameraWidget()
#CameraWidget.SetInteractor(Interactor)
#CameraWidget.SetEnabled(1)

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

#ErBoxWidget.On()

# Start the event loop.
Interactor.Start()


