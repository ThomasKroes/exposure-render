import vtk
import vtkErCorePython

# print vtkErCorePython

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

ErVolumeProperty.SetStepSizeFactorPrimary(3)
ErVolumeProperty.SetStepSizeFactorSecondary(3)
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

# Configure the light
Key = vtkErCorePython.vtkErAreaLight()
Key.SetPosition(500, 500, 500);
Key.SetFocalPoint(300, 300, 300);
Key.SetColor(100000, 100000, 100000);
Key.SetPositional(1);
Key.SetShapeType(0)

Fill = vtkErCorePython.vtkErAreaLight()
#Fill.SetPosition(0, 0, 1000000);
Fill.SetFocalPoint(300, 300, 300);
Fill.SetColor(600, 1600, 600)
Fill.SetPositional(1);
Fill.SetShapeType(0)
Fill.SetScale(1, 1, 1);
#Fill.GetTransform().RotateWXYZ(45, 0.5, 0.1, 0.9)
Fill.GetTransform().Translate(250, 250, 250)
Fill.GetTransform().Scale(1, 1, 1)
    
# Add the area light to the Renderer
#ErVolumeMapper.AddLight(Key);
ErVolumeMapper.AddLight(Fill);

Rim = vtkErCorePython.vtkErAreaLight()
#Fill.SetPosition(0, 0, 1000000);
Rim.SetFocalPoint(300, 300, 300);
Rim.SetColor(600, 10, 100)
Rim.SetPositional(1);
Rim.SetShapeType(0)
Rim.SetScale(1, 1, 1);
#Rim.GetTransform().RotateWXYZ(45, 0.5, 0.1, 0.9)
Rim.GetTransform().Translate(50, 450, 250)
Rim.GetTransform().Scale(100, 100, 250)

ErVolumeMapper.AddLight(Rim);

ErBackgroundLight = vtkErCorePython.vtkErBackgroundLight();
ErBackgroundLight.SetDiffuseColor(0.1, 0.1, 0.1);

# Add the background light to the Renderer
#ErVolumeMapper.AddLight(ErBackgroundLight);

# SlicePlane = vtkErCorePython.vtkErSlicePlane()
# Renderer.AddViewProp(SlicePlane)

Key.SetPosition(500, 500, 500);

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


