import vtk
import vtkErCorePython

VolumeFile = "examples/bonsai_small.mhd"

Renderer = vtk.vtkRenderer()
RendererWin = vtk.vtkRenderWindow()
RendererWin.AddRenderer(Renderer)
Interactor = vtk.vtkRenderWindowInteractor()
Interactor.SetRenderWindow(RendererWin)

Volume = vtk.vtkVolume()

# Read volume
Reader = vtk.vtkMetaImageReader()
Reader.SetFileName(VolumeFile)

Reader.Update()

# Exposure Rendererder volume mapper
ErVolumeMapper = vtkErCorePython.vtkErVolumeMapper()

Volume.SetMapper(ErVolumeMapper)

# Exposure Rendererder volume property
ErVolumeProperty = vtkErCorePython.vtkErVolumeProperty()

Opacity = vtk.vtkPiecewiseFunction()
Opacity.AddPoint(0, 0.001)
Opacity.AddPoint(20, 0)
Opacity.AddPoint(25, 1)
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

Specular.AddPoint(0, 0.5)
Specular.AddPoint(255, 0.5)

ErVolumeProperty.SetSpecular(0, Specular)
ErVolumeProperty.SetSpecular(1, Specular)
ErVolumeProperty.SetSpecular(2, Specular)

# Glossiness
Glossiness = vtk.vtkPiecewiseFunction()
Glossiness.AddPoint(0, 150)
Glossiness.AddPoint(255, 150)
ErVolumeProperty.SetGlossiness(Glossiness)

# IOR
IOR = vtk.vtkPiecewiseFunction()
IOR.AddPoint(0, 5)
IOR.AddPoint(255, 5)
ErVolumeProperty.SetIOR(IOR)

ErVolumeProperty.SetStepSizeFactorPrimary(0.1)
ErVolumeProperty.SetStepSizeFactorSecondary(0.1)
ErVolumeProperty.SetDensityScale(100)
ErVolumeProperty.SetShadingType(1)

# Assign the ER volume 
Volume.SetProperty(ErVolumeProperty)

ChangeInformation = vtk.vtkImageChangeInformation()

ChangeInformation.SetInput(Reader.GetOutput())
ChangeInformation.Update()
ChangeInformation.CenterImageOn()
ChangeInformation.SetOutputSpacing(0.001 * Reader.GetOutput().GetSpacing()[0], 0.001 * Reader.GetOutput().GetSpacing()[1], 0.001 * Reader.GetOutput().GetSpacing()[2])

ErVolumeMapper.SetInput(ChangeInformation.GetOutput())

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
Fill.SetType(0)
Fill.SetPosition(0.03, 0.03, 0.03)
Fill.SetFocalPoint(0, 0, 0)
Fill.SetDiffuseColor(.9, 0.6, 0.23)
Fill.SetIntensity(0.01)
Fill.SetPositional(1)
Fill.SetShapeType(0)
Fill.SetOneSided(1)
Fill.SetDistance(0.1)
Fill.SetElevation(45)
Fill.SetAzimuth(0)
Fill.SetInnerRadius(0.08)
Fill.SetOuterRadius(0.01)
Fill.SetSize(0.005, 0.005, 0.03)
#Fill.SetCamera(ErCamera)

ErVolumeMapper.AddLight(Fill)

ErBackgroundLight = vtkErCorePython.vtkErBackgroundLight();
ErBackgroundLight.SetDiffuseColor(1, 1, 1);
ErBackgroundLight.SetIntensity(0.8)

#ErVolumeMapper.AddLight(ErBackgroundLight)

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
        obj.GetRenderWindow().Finalize();
        obj.TerminateApp()
        
RendererWin.AddObserver("AbortCheckEvent", CheckAbort)
RendererWin.GetInteractor().AddObserver("KeyPressEvent", CheckKeyPress)

Interactor.Initialize()

RendererWin.Render()

#ErBoxWidget.On()

# Start the event loop.
Interactor.Start()


