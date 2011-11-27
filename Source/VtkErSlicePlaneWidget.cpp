/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "ErCoreStable.h"

#include "VtkErSlicePlaneWidget.h"

vtkStandardNewMacro(vtkErSlicePlaneWidget);

vtkCxxSetObjectMacro(vtkErSlicePlaneWidget,PlaneProperty,vtkProperty);

vtkErSlicePlaneWidget::vtkErSlicePlaneWidget() : vtkPolyDataSourceWidget()
{
  this->State = vtkErSlicePlaneWidget::Start;
  this->EventCallbackCommand->SetCallback(vtkErSlicePlaneWidget::ProcessEvents);
  
  this->NormalToXAxis = 0;
  this->NormalToYAxis = 0;
  this->NormalToZAxis = 0;
  this->Representation = VTK_PLANE_WIREFRAME;

  //Build the representation of the widget
  // Represent the plane
  this->PlaneSource = vtkPlaneSource::New();
  this->PlaneSource->SetXResolution(4);
  this->PlaneSource->SetYResolution(4);
  this->PlaneOutline = vtkPolyData::New();
  vtkPoints *pts = vtkPoints::New();
  pts->SetNumberOfPoints(4);
  vtkCellArray *outline = vtkCellArray::New();
  outline->InsertNextCell(4);
  outline->InsertCellPoint(0);
  outline->InsertCellPoint(1);
  outline->InsertCellPoint(2);
  outline->InsertCellPoint(3);
  this->PlaneOutline->SetPoints(pts);
  pts->Delete();
  this->PlaneOutline->SetPolys(outline);
  outline->Delete();
  this->PlaneMapper = vtkPolyDataMapper::New();
  this->PlaneMapper->SetInput(this->PlaneSource->GetOutput());
  this->PlaneActor = vtkActor::New();
  this->PlaneActor->SetMapper(this->PlaneMapper);

  this->ConeSource = vtkConeSource::New();
  this->ConeSource->SetResolution(12);
  this->ConeSource->SetAngle(25.0);
  this->ConeMapper = vtkPolyDataMapper::New();
  this->ConeMapper->SetInput(this->ConeSource->GetOutput());
  this->ConeActor = vtkActor::New();
  this->ConeActor->SetMapper(this->ConeMapper);

  this->Transform = vtkTransform::New();

  // Define the point coordinates
  double bounds[6];
  bounds[0] = -0.5;
  bounds[1] = 0.5;
  bounds[2] = -0.5;
  bounds[3] = 0.5;
  bounds[4] = -0.5;
  bounds[5] = 0.5;


  this->PlanePicker = vtkCellPicker::New();
  this->PlanePicker->SetTolerance(0.005); //need some fluff
  this->PlanePicker->AddPickList(this->PlaneActor);
  this->PlanePicker->AddPickList(this->ConeActor);
  this->PlanePicker->PickFromListOn();
  
  this->CurrentHandle = NULL;

  this->LastPickValid = 0;
  this->HandleSizeFactor = 0.1;
  this->SetHandleSize( 0.5 );
  
  // Set up the initial properties
  this->CreateDefaultProperties();
  
  this->SelectRepresentation();
  
  this->CubeSource				= vtkCubeSource::New();
  this->CubeMapper				= vtkPolyDataMapper::New();
  this->CubePolyData			= vtkPolyData::New();
  this->CubeCutter				= vtkCutter::New();
  this->CubeCutterMapper		= vtkPolyDataMapper::New();
  this->CubeCutterPlaneActor	= vtkActor::New();

  this->CubeMapper->SetInput(this->CubePolyData);
  this->CubeCutterMapper->SetInput(this->CubeCutter->GetOutput());

  this->CubeCutterPlaneActor->GetProperty()->SetColor(0.9, 0.6, 0);
  this->CubeCutterPlaneActor->GetProperty()->SetLineWidth(1);
  this->CubeCutterPlaneActor->SetMapper(this->CubeCutterMapper);

  this->ArrowSource		= vtkArrowSource::New();
  this->ArrowMapper		= vtkPolyDataMapper::New();
  this->ArrowActor		= vtkActor::New();

  this->ArrowMapper->SetInput(this->ArrowSource->GetOutput());
  this->ArrowActor->SetMapper(this->ArrowMapper);

	this->ArrowSource->SetShaftRadius(10);
	this->ArrowSource->SetTipLength(10);
	this->ArrowSource->SetTipRadius(10);



  // Initial creation of the widget, serves to initialize it
  // Call PlaceWidget() LAST in the constructor as it depends on ivar
  // values.
  this->PlaceWidget(bounds);
}

vtkErSlicePlaneWidget::~vtkErSlicePlaneWidget()
{
  this->PlaneActor->Delete();
  this->PlaneMapper->Delete();
  this->PlaneSource->Delete();
  this->PlaneOutline->Delete();

  this->ConeActor->Delete();
  this->ConeMapper->Delete();
  this->ConeSource->Delete();

  this->PlanePicker->Delete();

  if (this->HandleProperty)
    {
    this->HandleProperty->Delete();
    this->HandleProperty = 0;
    }

  if (this->SelectedHandleProperty)
    {
    this->SelectedHandleProperty->Delete();
    this->SelectedHandleProperty = 0;
    }

  if (this->PlaneProperty)
    {
    this->PlaneProperty->Delete();
    this->PlaneProperty = 0;
    }

  if (this->SelectedPlaneProperty)
    {
    this->SelectedPlaneProperty->Delete();
    this->SelectedPlaneProperty = 0;
    }

  this->Transform->Delete();
}

void vtkErSlicePlaneWidget::SetEnabled(int enabling)
{
  if ( ! this->Interactor )
    {
    vtkErrorMacro(<<"The interactor must be set prior to enabling/disabling widget");
    return;
    }

  if ( enabling ) //-----------------------------------------------------------
    {
    vtkDebugMacro(<<"Enabling plane widget");

    if ( this->Enabled ) //already enabled, just return
      {
      return;
      }
    
    if ( ! this->CurrentRenderer )
      {
      this->SetCurrentRenderer(this->Interactor->FindPokedRenderer(
        this->Interactor->GetLastEventPosition()[0],
        this->Interactor->GetLastEventPosition()[1]));
      if (this->CurrentRenderer == NULL)
        {
        return;
        }
      }

    this->Enabled = 1;

    // listen for the following events
    vtkRenderWindowInteractor *i = this->Interactor;
    i->AddObserver(vtkCommand::MouseMoveEvent, this->EventCallbackCommand, 
                   this->Priority);
    i->AddObserver(vtkCommand::LeftButtonPressEvent, 
                   this->EventCallbackCommand, this->Priority);
    i->AddObserver(vtkCommand::LeftButtonReleaseEvent, 
                   this->EventCallbackCommand, this->Priority);
    i->AddObserver(vtkCommand::MiddleButtonPressEvent, 
                   this->EventCallbackCommand, this->Priority);
    i->AddObserver(vtkCommand::MiddleButtonReleaseEvent, 
                   this->EventCallbackCommand, this->Priority);
    i->AddObserver(vtkCommand::RightButtonPressEvent, 
                   this->EventCallbackCommand, this->Priority);
    i->AddObserver(vtkCommand::RightButtonReleaseEvent, 
                   this->EventCallbackCommand, this->Priority);

    // Add the plane
    this->CurrentRenderer->AddActor(this->PlaneActor);
    this->PlaneActor->SetProperty(this->PlaneProperty);

//	this->CurrentRenderer->AddActor(this->CubeCutterPlaneActor);
//	this->CurrentRenderer->AddActor(this->ArrowActor);

//	this->CurrentRenderer->AddActor(this->ThetaHandleActor);
//	this->CurrentRenderer->AddActor(this->PhiHandleActor);

    this->SelectRepresentation();
    this->InvokeEvent(vtkCommand::EnableEvent,NULL);
    }
  
  else //disabling----------------------------------------------------------
    {
    vtkDebugMacro(<<"Disabling plane widget");

    if ( ! this->Enabled ) //already disabled, just return
      {
      return;
      }
    
    this->Enabled = 0;

    // don't listen for events any more
    this->Interactor->RemoveObserver(this->EventCallbackCommand);

    // turn off the plane
    this->CurrentRenderer->RemoveActor(this->PlaneActor);

    // turn off the normal vector
    this->CurrentRenderer->RemoveActor(this->ConeActor);

    this->CurrentHandle = NULL;
    this->InvokeEvent(vtkCommand::DisableEvent,NULL);
    this->SetCurrentRenderer(NULL);
    }

  this->Interactor->Render();
}

void vtkErSlicePlaneWidget::ProcessEvents(vtkObject* vtkNotUsed(object), 
                                   unsigned long event,
                                   void* clientdata, 
                                   void* vtkNotUsed(calldata))
{
  vtkErSlicePlaneWidget* self = reinterpret_cast<vtkErSlicePlaneWidget *>( clientdata );

  //okay, let's do the right thing
  switch(event)
    {
    case vtkCommand::LeftButtonPressEvent:
      self->OnLeftButtonDown();
      break;
    case vtkCommand::LeftButtonReleaseEvent:
      self->OnLeftButtonUp();
      break;
    case vtkCommand::MiddleButtonPressEvent:
      self->OnMiddleButtonDown();
      break;
    case vtkCommand::MiddleButtonReleaseEvent:
      self->OnMiddleButtonUp();
      break;
    case vtkCommand::RightButtonPressEvent:
      self->OnRightButtonDown();
      break;
    case vtkCommand::RightButtonReleaseEvent:
      self->OnRightButtonUp();
      break;
    case vtkCommand::MouseMoveEvent:
      self->OnMouseMove();
      break;
    }
}

void vtkErSlicePlaneWidget::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  if ( this->HandleProperty )
    {
    os << indent << "Handle Property: " << this->HandleProperty << "\n";
    }
  else
    {
    os << indent << "Handle Property: (none)\n";
    }
  if ( this->SelectedHandleProperty )
    {
    os << indent << "Selected Handle Property: " 
       << this->SelectedHandleProperty << "\n";
    }
  else
    {
    os << indent << "SelectedHandle Property: (none)\n";
    }

  if ( this->PlaneProperty )
    {
    os << indent << "Plane Property: " << this->PlaneProperty << "\n";
    }
  else
    {
    os << indent << "Plane Property: (none)\n";
    }
  if ( this->SelectedPlaneProperty )
    {
    os << indent << "Selected Plane Property: " 
       << this->SelectedPlaneProperty << "\n";
    }
  else
    {
    os << indent << "Selected Plane Property: (none)\n";
    }

  os << indent << "Plane Representation: ";
  if ( this->Representation == VTK_PLANE_WIREFRAME )
    {
    os << "Wireframe\n";
    }
  else if ( this->Representation == VTK_PLANE_SURFACE )
    {
    os << "Surface\n";
    }
  else //( this->Representation == VTK_PLANE_OUTLINE )
    {
    os << "Outline\n";
    }

  os << indent << "Normal To X Axis: " 
     << (this->NormalToXAxis ? "On" : "Off") << "\n";
  os << indent << "Normal To Y Axis: "
     << (this->NormalToYAxis ? "On" : "Off") << "\n";
  os << indent << "Normal To Z Axis: " 
     << (this->NormalToZAxis ? "On" : "Off") << "\n";

  int res = this->PlaneSource->GetXResolution();
  double *o = this->PlaneSource->GetOrigin();
  double *pt1 = this->PlaneSource->GetPoint1();
  double *pt2 = this->PlaneSource->GetPoint2();

  os << indent << "Resolution: " << res << "\n";
  os << indent << "Origin: (" << o[0] << ", "
     << o[1] << ", "
     << o[2] << ")\n";
  os << indent << "Point 1: (" << pt1[0] << ", "
     << pt1[1] << ", "
     << pt1[2] << ")\n";
  os << indent << "Point 2: (" << pt2[0] << ", "
     << pt2[1] << ", "
     << pt2[2] << ")\n";
}

void vtkErSlicePlaneWidget::PositionHandles()
{
  double *o = this->PlaneSource->GetOrigin();
  double *pt1 = this->PlaneSource->GetPoint1();
  double *pt2 = this->PlaneSource->GetPoint2();

  double x[3];
  x[0] = pt1[0] + pt2[0] - o[0];
  x[1] = pt1[1] + pt2[1] - o[1];
  x[2] = pt1[2] + pt2[2] - o[2];

  // set up the outline
  if ( this->Representation == VTK_PLANE_OUTLINE )
    {
    this->PlaneOutline->GetPoints()->SetPoint(0,o);
    this->PlaneOutline->GetPoints()->SetPoint(1,pt1);
    this->PlaneOutline->GetPoints()->SetPoint(2,x);
    this->PlaneOutline->GetPoints()->SetPoint(3,pt2);
    this->PlaneOutline->Modified();
    }
  this->SelectRepresentation();

  // Create the normal vector
  double center[3];
  this->PlaneSource->GetCenter(center);
  double p2[3];
  this->PlaneSource->GetNormal(this->Normal);
  vtkMath::Normalize(this->Normal);
  double d = sqrt( 
    vtkMath::Distance2BetweenPoints(
      this->PlaneSource->GetPoint1(),this->PlaneSource->GetPoint2()) );

  p2[0] = center[0] + 0.35 * d * this->Normal[0];
  p2[1] = center[1] + 0.35 * d * this->Normal[1];
  p2[2] = center[2] + 0.35 * d * this->Normal[2];
  this->ConeSource->SetCenter(p2);
  this->ConeSource->SetDirection(this->Normal);

  p2[0] = center[0] - 0.35 * d * this->Normal[0];
  p2[1] = center[1] - 0.35 * d * this->Normal[1];
  p2[2] = center[2] - 0.35 * d * this->Normal[2];

  vtkSmartPointer<vtkPlane> Plane = vtkPlane::New();

  this->GetPlane(Plane);

  this->CubeCutter->SetCutFunction(Plane);
  this->CubeCutter->SetInput(CubeMapper->GetInput());
  this->CubeCutter->GenerateValues(1, 0, 0);

  this->ArrowActor->SetPosition(this->PlaneSource->GetCenter());
}

int vtkErSlicePlaneWidget::HighlightHandle(vtkProp *prop)
{
  // first unhighlight anything picked
  if ( this->CurrentHandle )
    {
    this->CurrentHandle->SetProperty(this->HandleProperty);
    }

  this->CurrentHandle = static_cast<vtkActor *>(prop);

  if ( this->CurrentHandle )
    {
    this->ValidPick = 1;
//    this->HandlePicker->GetPickPosition(this->LastPickPosition);
    this->CurrentHandle->SetProperty(this->SelectedHandleProperty);
    }
  
  return -1;
}

void vtkErSlicePlaneWidget::HighlightNormal(int highlight)
{
  if ( highlight )
    {
    this->ValidPick = 1;
    this->PlanePicker->GetPickPosition(this->LastPickPosition);
    this->ConeActor->SetProperty(this->SelectedHandleProperty);
    }
  else
    {
    }
}

void vtkErSlicePlaneWidget::HighlightPlane(int highlight)
{
  if ( highlight )
    {
    this->ValidPick = 1;
    this->PlanePicker->GetPickPosition(this->LastPickPosition);
    this->PlaneActor->SetProperty(this->SelectedPlaneProperty);
    }
  else
    {
    this->PlaneActor->SetProperty(this->PlaneProperty);
    }
}

void vtkErSlicePlaneWidget::OnLeftButtonDown()
{
  int X = this->Interactor->GetEventPosition()[0];
  int Y = this->Interactor->GetEventPosition()[1];

  // Okay, make sure that the pick is in the current renderer
  if (!this->CurrentRenderer || !this->CurrentRenderer->IsInViewport(X, Y))
    {
    this->State = vtkErSlicePlaneWidget::Outside;
    return;
    }
  
  // Okay, we can process this. Try to pick handles first;
  // if no handles picked, then try to pick the plane.
  vtkAssemblyPath *path = NULL;
//  this->HandlePicker->Pick(X,Y,0.0,this->CurrentRenderer);
//  path = this->HandlePicker->GetPath();
  if ( path != NULL )
    {
    this->State = vtkErSlicePlaneWidget::Moving;
    this->HighlightHandle(path->GetFirstNode()->GetViewProp());
    }
  else
    {
    this->PlanePicker->Pick(X,Y,0.0,this->CurrentRenderer);
    path = this->PlanePicker->GetPath();
    if ( path != NULL )
      {
      vtkProp *prop = path->GetFirstNode()->GetViewProp();
      if ( prop == this->ConeActor)
        {
        this->State = vtkErSlicePlaneWidget::Rotating;
        this->HighlightNormal(1);
        }
      else if (this->Interactor->GetControlKey())
        {
        this->State = vtkErSlicePlaneWidget::Spinning;
        this->HighlightNormal(1);
        }
      else
        {
        this->State = vtkErSlicePlaneWidget::Moving;
        this->HighlightPlane(1);
        }
      }
    else
      {
      this->State = vtkErSlicePlaneWidget::Outside;
      this->HighlightHandle(NULL);
      return;
      }
    }
  
  this->EventCallbackCommand->SetAbortFlag(1);
  this->StartInteraction();
  this->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
  this->Interactor->Render();
}

void vtkErSlicePlaneWidget::OnLeftButtonUp()
{
  if ( this->State == vtkErSlicePlaneWidget::Outside ||
       this->State == vtkErSlicePlaneWidget::Start )
    {
    return;
    }

  this->State = vtkErSlicePlaneWidget::Start;
  this->HighlightHandle(NULL);
  this->HighlightPlane(0);
  this->HighlightNormal(0);
  this->SizeHandles();

  this->EventCallbackCommand->SetAbortFlag(1);
  this->EndInteraction();
  this->InvokeEvent(vtkCommand::EndInteractionEvent,NULL);
  this->Interactor->Render();
}

void vtkErSlicePlaneWidget::OnMiddleButtonDown()
{
  int X = this->Interactor->GetEventPosition()[0];
  int Y = this->Interactor->GetEventPosition()[1];

  // Okay, make sure that the pick is in the current renderer
  if (!this->CurrentRenderer || !this->CurrentRenderer->IsInViewport(X, Y))
    {
    this->State = vtkErSlicePlaneWidget::Outside;
    return;
    }
  
  // Okay, we can process this. If anything is picked, then we
  // can start pushing the plane.
  vtkAssemblyPath *path = NULL;
//  this->HandlePicker->Pick(X,Y,0.0,this->CurrentRenderer);
 // path = this->HandlePicker->GetPath();
  if ( path != NULL )
    {
    this->State = vtkErSlicePlaneWidget::Pushing;
    this->HighlightPlane(1);
    this->HighlightNormal(1);
    this->HighlightHandle(path->GetFirstNode()->GetViewProp());
    }
  else
    {
    this->PlanePicker->Pick(X,Y,0.0,this->CurrentRenderer);
    path = this->PlanePicker->GetPath();
    if ( path == NULL ) //nothing picked
      {
      this->State = vtkErSlicePlaneWidget::Outside;
      return;
      }
    else
      {
      this->State = vtkErSlicePlaneWidget::Pushing;
      this->HighlightNormal(1);
      this->HighlightPlane(1);
      }
    }
  
  this->EventCallbackCommand->SetAbortFlag(1);
  this->StartInteraction();
  this->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
  this->Interactor->Render();
}

void vtkErSlicePlaneWidget::OnMiddleButtonUp()
{
  if ( this->State == vtkErSlicePlaneWidget::Outside ||
       this->State == vtkErSlicePlaneWidget::Start )
    {
    return;
    }

  this->State = vtkErSlicePlaneWidget::Start;
  this->HighlightPlane(0);
  this->HighlightNormal(0);
  this->HighlightHandle(NULL);
  this->SizeHandles();
  
  this->EventCallbackCommand->SetAbortFlag(1);
  this->EndInteraction();
  this->InvokeEvent(vtkCommand::EndInteractionEvent,NULL);
  this->Interactor->Render();
}

void vtkErSlicePlaneWidget::OnRightButtonDown()
{
  int X = this->Interactor->GetEventPosition()[0];
  int Y = this->Interactor->GetEventPosition()[1];

  // Okay, make sure that the pick is in the current renderer
  if (!this->CurrentRenderer || !this->CurrentRenderer->IsInViewport(X, Y))
    {
    this->State = vtkErSlicePlaneWidget::Outside;
    return;
    }
  
  // Okay, we can process this. Try to pick handles first;
  // if no handles picked, then pick the bounding box.
  vtkAssemblyPath *path = NULL;
//  this->HandlePicker->Pick(X,Y,0.0,this->CurrentRenderer);
//  path = this->HandlePicker->GetPath();
  if ( path != NULL )
    {
    this->State = vtkErSlicePlaneWidget::Scaling;
    this->HighlightPlane(1);
    this->HighlightHandle(path->GetFirstNode()->GetViewProp());
    }
  else //see if we picked the plane or a normal
    {
    this->PlanePicker->Pick(X,Y,0.0,this->CurrentRenderer);
    path = this->PlanePicker->GetPath();
    if ( path == NULL )
      {
      this->State = vtkErSlicePlaneWidget::Outside;
      return;
      }
    else
      {
      this->State = vtkErSlicePlaneWidget::Scaling;
      this->HighlightPlane(1);
      }
    }
  
  this->EventCallbackCommand->SetAbortFlag(1);
  this->StartInteraction();
  this->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
  this->Interactor->Render();
}

void vtkErSlicePlaneWidget::OnRightButtonUp()
{
  if ( this->State == vtkErSlicePlaneWidget::Outside ||
       this->State == vtkErSlicePlaneWidget::Start )
    {
    return;
    }

  this->State = vtkErSlicePlaneWidget::Start;
  this->HighlightPlane(0);
  this->SizeHandles();
  
  this->EventCallbackCommand->SetAbortFlag(1);
  this->EndInteraction();
  this->InvokeEvent(vtkCommand::EndInteractionEvent,NULL);
  this->Interactor->Render();
}

void vtkErSlicePlaneWidget::OnMouseMove()
{
  // See whether we're active
  if ( this->State == vtkErSlicePlaneWidget::Outside || 
       this->State == vtkErSlicePlaneWidget::Start )
    {
    return;
    }
  
  int X = this->Interactor->GetEventPosition()[0];
  int Y = this->Interactor->GetEventPosition()[1];

  // Do different things depending on state
  // Calculations everybody does
  double focalPoint[4], pickPoint[4], prevPickPoint[4];
  double z, vpn[3];

  vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
  if ( !camera )
    {
    return;
    }

  // Compute the two points defining the motion vector
  this->ComputeWorldToDisplay(this->LastPickPosition[0], 
                              this->LastPickPosition[1],
                              this->LastPickPosition[2], focalPoint);
  z = focalPoint[2];
  this->ComputeDisplayToWorld(
    double(this->Interactor->GetLastEventPosition()[0]),
    double(this->Interactor->GetLastEventPosition()[1]),
    z, prevPickPoint);
  this->ComputeDisplayToWorld(double(X), double(Y), z, pickPoint);
  
  // Process the motion
  if ( this->State == vtkErSlicePlaneWidget::Moving )
    {
    // Okay to process
    if ( this->CurrentHandle )
      {

      }
    else //must be moving the plane
      {
      this->Translate(prevPickPoint, pickPoint);
      }
    }
  else if ( this->State == vtkErSlicePlaneWidget::Scaling )
    {
    }
  else if ( this->State == vtkErSlicePlaneWidget::Pushing )
    {
    this->Push(prevPickPoint, pickPoint);
    }
  else if ( this->State == vtkErSlicePlaneWidget::Rotating )
    {
    camera->GetViewPlaneNormal(vpn);
    this->Rotate(X, Y, prevPickPoint, pickPoint, vpn);
    }
  else if ( this->State == vtkErSlicePlaneWidget::Spinning )
    {
    }


  // Interact, if desired
  this->EventCallbackCommand->SetAbortFlag(1);
  this->InvokeEvent(vtkCommand::InteractionEvent,NULL);
  
  this->Interactor->Render();
}

void vtkErSlicePlaneWidget::MoveOrigin(double *p1, double *p2)
{
  //Get the plane definition
  double *o = this->PlaneSource->GetOrigin();
  double *pt1 = this->PlaneSource->GetPoint1();
  double *pt2 = this->PlaneSource->GetPoint2();

  //Get the vector of motion
  double v[3];
  v[0] = p2[0] - p1[0];
  v[1] = p2[1] - p1[1];
  v[2] = p2[2] - p1[2];

  // The point opposite the origin (pt3) stays fixed
  double pt3[3];
  pt3[0] = o[0] + (pt1[0] - o[0]) + (pt2[0] - o[0]);
  pt3[1] = o[1] + (pt1[1] - o[1]) + (pt2[1] - o[1]);
  pt3[2] = o[2] + (pt1[2] - o[2]) + (pt2[2] - o[2]);

  // Define vectors from point pt3
  double p13[3], p23[3];
  p13[0] = pt1[0] - pt3[0];
  p13[1] = pt1[1] - pt3[1];
  p13[2] = pt1[2] - pt3[2];
  p23[0] = pt2[0] - pt3[0];
  p23[1] = pt2[1] - pt3[1];
  p23[2] = pt2[2] - pt3[2];

  double vN = vtkMath::Norm(v);
  double n13 = vtkMath::Norm(p13);
  double n23 = vtkMath::Norm(p23);

  // Project v onto these vector to determine the amount of motion
  // Scale it by the relative size of the motion to the vector length
  double d1 = (vN/n13) * vtkMath::Dot(v,p13) / (vN*n13);
  double d2 = (vN/n23) * vtkMath::Dot(v,p23) / (vN*n23);

  double point1[3], point2[3], origin[3];
  for (int i=0; i<3; i++)
    {
    point1[i] = pt3[i] + (1.0+d1)*p13[i];
    point2[i] = pt3[i] + (1.0+d2)*p23[i];
    origin[i] = pt3[i] + (1.0+d1)*p13[i] + (1.0+d2)*p23[i];
    }
  
  this->PlaneSource->SetOrigin(origin);
  this->PlaneSource->SetPoint1(point1);
  this->PlaneSource->SetPoint2(point2);
  this->PlaneSource->Update();

  this->PositionHandles();
}

void vtkErSlicePlaneWidget::Rotate(int X, int Y, double *p1, double *p2, double *vpn)
{
  double *o = this->PlaneSource->GetOrigin();
  double *pt1 = this->PlaneSource->GetPoint1();
  double *pt2 = this->PlaneSource->GetPoint2();
  double *center = this->PlaneSource->GetCenter();

  double v[3]; //vector of motion
  double axis[3]; //axis of rotation
  double theta; //rotation angle

  // mouse motion vector in world space
  v[0] = p2[0] - p1[0];
  v[1] = p2[1] - p1[1];
  v[2] = p2[2] - p1[2];

  // Create axis of rotation and angle of rotation
  vtkMath::Cross(vpn,v,axis);
  if ( vtkMath::Normalize(axis) == 0.0 )
    {
    return;
    }
  int *size = this->CurrentRenderer->GetSize();
  double l2 =
    (X-this->Interactor->GetLastEventPosition()[0])*
    (X-this->Interactor->GetLastEventPosition()[0]) + 
    (Y-this->Interactor->GetLastEventPosition()[1])*
    (Y-this->Interactor->GetLastEventPosition()[1]);
  theta = 360.0 * sqrt(l2/(size[0]*size[0]+size[1]*size[1]));

  //Manipulate the transform to reflect the rotation
  this->Transform->Identity();
  this->Transform->Translate(center[0],center[1],center[2]);
  this->Transform->RotateWXYZ(theta,axis);
  this->Transform->Translate(-center[0],-center[1],-center[2]);

  //Set the corners
  double oNew[3], pt1New[3], pt2New[3];
  this->Transform->TransformPoint(o,oNew);
  this->Transform->TransformPoint(pt1,pt1New);
  this->Transform->TransformPoint(pt2,pt2New);

  this->PlaneSource->SetOrigin(oNew);
  this->PlaneSource->SetPoint1(pt1New);
  this->PlaneSource->SetPoint2(pt2New);
  this->PlaneSource->Update();

  this->PositionHandles();
}

// Loop through all points and translate them
void vtkErSlicePlaneWidget::Translate(double *p1, double *p2)
{
  //Get the motion vector
  double v[3];
  v[0] = p2[0] - p1[0];
  v[1] = p2[1] - p1[1];
  v[2] = p2[2] - p1[2];
  
  //int res = this->PlaneSource->GetXResolution();
  double *o = this->PlaneSource->GetOrigin();
  double *pt1 = this->PlaneSource->GetPoint1();
  double *pt2 = this->PlaneSource->GetPoint2();

  double origin[3], point1[3], point2[3];
  for (int i=0; i<3; i++)
    {
    origin[i] = o[i] + v[i];
    point1[i] = pt1[i] + v[i];
    point2[i] = pt2[i] + v[i];
    }
  
  this->PlaneSource->SetOrigin(origin);
  this->PlaneSource->SetPoint1(point1);
  this->PlaneSource->SetPoint2(point2);
  this->PlaneSource->Update();

  this->PositionHandles();
}

void vtkErSlicePlaneWidget::Push(double *p1, double *p2)
{
  //Get the motion vector
  double v[3];
  v[0] = p2[0] - p1[0];
  v[1] = p2[1] - p1[1];
  v[2] = p2[2] - p1[2];
  
  this->PlaneSource->Push( vtkMath::Dot(v,this->Normal) );
  this->PlaneSource->Update();
  this->PositionHandles();
}

void vtkErSlicePlaneWidget::CreateDefaultProperties()
{
  // Handle properties
  this->HandleProperty = vtkProperty::New();
  this->HandleProperty->SetColor(1,1,1);

  this->SelectedHandleProperty = vtkProperty::New();
  this->SelectedHandleProperty->SetColor(1,0,0);

  // Plane properties
  this->PlaneProperty = vtkProperty::New();
  this->PlaneProperty->SetAmbient(1.0);
  this->PlaneProperty->SetAmbientColor(1.0,1.0,1.0);

  this->SelectedPlaneProperty = vtkProperty::New();
  this->SelectRepresentation();
  this->SelectedPlaneProperty->SetAmbient(1.0);
  this->SelectedPlaneProperty->SetAmbientColor(0.0,1.0,0.0);
}

void vtkErSlicePlaneWidget::PlaceWidget(double bds[6])
{
  int i;
  double bounds[6], center[3];

  this->AdjustBounds(bds, bounds, center);

  if (this->Input || this->Prop3D)
    {
    if ( this->NormalToYAxis )
      {
      this->PlaneSource->SetOrigin(bounds[0],center[1],bounds[4]);
      this->PlaneSource->SetPoint1(bounds[1],center[1],bounds[4]);
      this->PlaneSource->SetPoint2(bounds[0],center[1],bounds[5]);
      }
    else if ( this->NormalToZAxis )
      {
      this->PlaneSource->SetOrigin(bounds[0],bounds[2],center[2]);
      this->PlaneSource->SetPoint1(bounds[1],bounds[2],center[2]);
      this->PlaneSource->SetPoint2(bounds[0],bounds[3],center[2]);
      }
    else //default or x-normal
      {
      this->PlaneSource->SetOrigin(center[0],bounds[2],bounds[4]);
      this->PlaneSource->SetPoint1(center[0],bounds[3],bounds[4]);
      this->PlaneSource->SetPoint2(center[0],bounds[2],bounds[5]);
      }
    }

  this->PlaneSource->Update();

  // Position the handles at the end of the planes
  this->PositionHandles();

  for (i=0; i<6; i++)
    {
    this->InitialBounds[i] = bounds[i];
    }
  

  if (this->Input || this->Prop3D)
    {
    this->InitialLength = sqrt((bounds[1]-bounds[0])*(bounds[1]-bounds[0]) +
                               (bounds[3]-bounds[2])*(bounds[3]-bounds[2]) +
                               (bounds[5]-bounds[4])*(bounds[5]-bounds[4]));
    }
  else
    {
    // this means we have to make use of the PolyDataSource, so
    // we just calculate the magnitude of the longest diagonal on
    // the plane and use that as InitialLength
    double origin[3], point1[3], point2[3];
    this->PlaneSource->GetOrigin(origin);
    this->PlaneSource->GetPoint1(point1);
    this->PlaneSource->GetPoint2(point2);
    double sqr1 = 0, sqr2 = 0;
    for (i = 0; i < 3; i++)
      {
      sqr1 += (point1[i] - origin[i]) * (point1[i] - origin[i]);
      sqr2 += (point2[i] - origin[i]) * (point2[i] - origin[i]);
      }

    this->InitialLength = sqrt(sqr1 + sqr2);
    }

  this->CubeSource->SetBounds(bds);
  this->CubeSource->Update();
  this->CubeMapper->SetInput(this->CubeSource->GetOutput());

  vtkBoundingBox AABB(bds);

  // Set the radius on the sphere handles
  this->SizeHandles();
}

void vtkErSlicePlaneWidget::SizeHandles()
{
  double radius = this->vtk3DWidget::SizeHandles(this->HandleSizeFactor);
  
  // Set the height and radius of the cone
  this->ConeSource->SetHeight(2.0*radius);
  this->ConeSource->SetRadius(radius);
}


void vtkErSlicePlaneWidget::SelectRepresentation()
{
  if ( ! this->CurrentRenderer )
    {
    return;
    }

  if ( this->Representation == VTK_PLANE_OFF )
    {
    this->CurrentRenderer->RemoveActor(this->PlaneActor);
    }
  else if ( this->Representation == VTK_PLANE_OUTLINE )
    {
    this->CurrentRenderer->RemoveActor(this->PlaneActor);
    this->CurrentRenderer->AddActor(this->PlaneActor);
    this->PlaneMapper->SetInput( this->PlaneOutline );
    this->PlaneActor->GetProperty()->SetRepresentationToWireframe();
    }
  else if ( this->Representation == VTK_PLANE_SURFACE )
    {
    this->CurrentRenderer->RemoveActor(this->PlaneActor);
    this->CurrentRenderer->AddActor(this->PlaneActor);
    this->PlaneMapper->SetInput( this->PlaneSource->GetOutput() );
    this->PlaneActor->GetProperty()->SetRepresentationToSurface();
    }
  else //( this->Representation == VTK_PLANE_WIREFRAME )
    {
    this->CurrentRenderer->RemoveActor(this->PlaneActor);
    this->CurrentRenderer->AddActor(this->PlaneActor);
    this->PlaneMapper->SetInput( this->PlaneSource->GetOutput() );
    this->PlaneActor->GetProperty()->SetRepresentationToWireframe();
    }
}

void vtkErSlicePlaneWidget::SetOrigin(double x, double y, double z) 
{
  this->PlaneSource->SetOrigin(x,y,z);
  this->PositionHandles();
}

void vtkErSlicePlaneWidget::SetOrigin(double x[3]) 
{
  this->SetOrigin(x[0], x[1], x[2]);
}

double* vtkErSlicePlaneWidget::GetOrigin() 
{
  return this->PlaneSource->GetOrigin();
}

void vtkErSlicePlaneWidget::GetOrigin(double xyz[3]) 
{
  this->PlaneSource->GetOrigin(xyz);
}

// Description:
// Set the center of the plane.
void vtkErSlicePlaneWidget::SetCenter(double x, double y, double z) 
{
  this->PlaneSource->SetCenter(x, y, z);
  this->PositionHandles();
}

// Description:
// Set the center of the plane.
void vtkErSlicePlaneWidget::SetCenter(double c[3]) 
{
  this->SetCenter(c[0], c[1], c[2]);
}

// Description:
// Get the center of the plane.
double* vtkErSlicePlaneWidget::GetCenter() 
{
  return this->PlaneSource->GetCenter();
}

void vtkErSlicePlaneWidget::GetCenter(double xyz[3]) 
{
  this->PlaneSource->GetCenter(xyz);
}

// Description:
// Set the normal to the plane.
void vtkErSlicePlaneWidget::SetNormal(double x, double y, double z) 
{
  this->PlaneSource->SetNormal(x, y, z);
  this->PositionHandles();
}

// Description:
// Set the normal to the plane.
void vtkErSlicePlaneWidget::SetNormal(double n[3]) 
{
  this->SetNormal(n[0], n[1], n[2]);
}

// Description:
// Get the normal to the plane.
double* vtkErSlicePlaneWidget::GetNormal() 
{
  return this->PlaneSource->GetNormal();
}

void vtkErSlicePlaneWidget::GetNormal(double xyz[3]) 
{
  this->PlaneSource->GetNormal(xyz);
}

void vtkErSlicePlaneWidget::GetPolyData(vtkPolyData *pd)
{ 
  pd->ShallowCopy(this->PlaneSource->GetOutput()); 
}

vtkPolyDataAlgorithm *vtkErSlicePlaneWidget::GetPolyDataAlgorithm()
{
  return this->PlaneSource;
}

void vtkErSlicePlaneWidget::GetPlane(vtkPlane *plane)
{
  if ( plane == NULL )
    {
    return;
    }
  
  plane->SetNormal(this->GetNormal());
  plane->SetOrigin(this->GetCenter());
}

void vtkErSlicePlaneWidget::UpdatePlacement(void)
{
  this->PlaneSource->Update();
  this->PositionHandles();
}
