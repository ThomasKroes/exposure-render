/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include <vtkActor.h>
#include <vtkAssemblyNode.h>
#include <vtkAssemblyPath.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCellPicker.h>
#include <vtkCommand.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkMath.h>
#include <vtkObjectFactory.h>
#include <vtkPlanes.h>
#include <vtkErPointWidget.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSphereSource.h>

#include "vtkErAreaLightWidget.h"

//----------------------------------------------------------------------------
// This class is used to coordinate the interaction between the point widget
// at the center of the line and the line widget. When the line is selected
// (as compared to the handles), a point widget appears at the selection
// point, which can be manipulated in the usual way.
class vtkPWCallback : public vtkCommand
{
public:
  static vtkPWCallback *New()
    { return new vtkPWCallback; }
  virtual void Execute(vtkObject *vtkNotUsed(caller), unsigned long, void*)
    {
      double x[3];
      this->PointWidget->GetPosition(x);
//      this->LineWidget->SetLinePosition(x);
    }
  vtkPWCallback():LineWidget(0),PointWidget(0) {}
  vtkErAreaLightWidget  *LineWidget;
  vtkErPointWidget *PointWidget;
};

//----------------------------------------------------------------------------
// This class is used to coordinate the interaction between the point widget
// (point 1) and the line widget.
class vtkPW1Callback : public vtkCommand
{
public:
  static vtkPW1Callback *New()
    { return new vtkPW1Callback; }
  virtual void Execute(vtkObject *vtkNotUsed(caller), unsigned long, void*)
    {
      double x[3];
      this->PointWidget->GetPosition(x);
	  this->LineWidget->SetPosition(x);
    }
  vtkPW1Callback():LineWidget(0),PointWidget(0) {}
  vtkErAreaLightWidget  *LineWidget;
  vtkErPointWidget *PointWidget;
};

//----------------------------------------------------------------------------
// This class is used to coordinate the interaction between the point widget
// (point 2) and the line widget.
class vtkPW2Callback : public vtkCommand
{
public:
  static vtkPW2Callback *New()
    { return new vtkPW2Callback; }
  virtual void Execute(vtkObject *vtkNotUsed(caller), unsigned long, void*)
    {
      double x[3];
      this->PointWidget->GetPosition(x);
      this->LineWidget->SetTarget(x);
    }
  vtkPW2Callback():LineWidget(0),PointWidget(0) {}
  vtkErAreaLightWidget *LineWidget;
  vtkErPointWidget *PointWidget;
};

vtkStandardNewMacro(vtkErAreaLightWidget);

vtkErAreaLightWidget::vtkErAreaLightWidget()
{
	this->State = vtkErAreaLightWidget::Start;
	this->EventCallbackCommand->SetCallback(vtkErAreaLightWidget::ProcessEvents);

	this->LineSource = vtkLineSource::New();
	this->LineSource->SetResolution(5);
	this->LineMapper = vtkPolyDataMapper::New();
	this->LineMapper->SetInput(this->LineSource->GetOutput());
	this->LineActor = vtkActor::New();
	this->LineActor->SetMapper(this->LineMapper);

	this->PositionHandleGeometry = vtkSphereSource::New();
	this->PositionHandleGeometry->SetThetaResolution(32);
	this->PositionHandleGeometry->SetPhiResolution(16);
	this->PositionHandleMapper = vtkPolyDataMapper::New();
	this->PositionHandleMapper = vtkPolyDataMapper::New();
	this->PositionHandleMapper->SetInput(this->PositionHandleGeometry->GetOutput());
	this->PositionHandle = vtkActor::New();
	this->PositionHandle->SetMapper(this->PositionHandleMapper);

	this->TargetHandleGeometry = vtkSphereSource::New();
	this->TargetHandleGeometry->SetThetaResolution(32);
	this->TargetHandleGeometry->SetPhiResolution(16);
	this->TargetHandleMapper = vtkPolyDataMapper::New();
	this->TargetHandleMapper = vtkPolyDataMapper::New();
	this->TargetHandleMapper->SetInput(this->TargetHandleGeometry->GetOutput());
	this->TargetHandle = vtkActor::New();
	this->TargetHandle->SetMapper(this->TargetHandleMapper);

	this->HandlePicker = vtkCellPicker::New();
	this->HandlePicker->SetTolerance(0.001);
	this->HandlePicker->AddPickList(PositionHandle);
	this->HandlePicker->AddPickList(TargetHandle);
	this->HandlePicker->PickFromListOn();

	this->LinePicker = vtkCellPicker::New();
	this->LinePicker->SetTolerance(0.005);
	this->LinePicker->AddPickList(this->LineActor);
	this->LinePicker->PickFromListOn();

	this->CurrentHandle = NULL;

	this->CreateDefaultProperties();

	this->PointWidget  = vtkErPointWidget::New();
	this->PointWidget->AllOff();
	this->PointWidget->SetHotSpotSize(0.5);

	this->PointWidget1 = vtkErPointWidget::New();
	this->PointWidget1->AllOff();
	this->PointWidget1->SetHotSpotSize(0.5);

	this->PointWidget2 = vtkErPointWidget::New();
	this->PointWidget2->AllOff();
	this->PointWidget2->SetHotSpotSize(0.5);

	this->PWCallback = vtkPWCallback::New();
	this->PWCallback->LineWidget = this;
	this->PWCallback->PointWidget = this->PointWidget;
	this->PW1Callback = vtkPW1Callback::New();
	this->PW1Callback->LineWidget = this;
	this->PW1Callback->PointWidget = this->PointWidget1;
	this->PW2Callback = vtkPW2Callback::New();
	this->PW2Callback->LineWidget = this;
	this->PW2Callback->PointWidget = this->PointWidget2;

	// Very tricky, the point widgets watch for their own
	// interaction events.
	this->PointWidget->AddObserver(vtkCommand::InteractionEvent,
									this->PWCallback, 0.0);
	this->PointWidget1->AddObserver(vtkCommand::InteractionEvent,
									this->PW1Callback, 0.0);
	this->PointWidget2->AddObserver(vtkCommand::InteractionEvent,
									this->PW2Callback, 0.0);
	this->CurrentPointWidget = NULL;
}

vtkErAreaLightWidget::~vtkErAreaLightWidget()
{
}

void vtkErAreaLightWidget::SetEnabled(int Enabled)
{
	if (!this->Interactor)
	{
		vtkErrorMacro(<<"The interactor must be set prior to enabling/disabling widget");
		return;
	}

	if (Enabled)
	{
		vtkDebugMacro(<<"Enabling line widget");

		if (this->Enabled ) //already enabled, just return
		{
			return;
		}

		if (!this->CurrentRenderer)
		{
			this->SetCurrentRenderer(this->Interactor->FindPokedRenderer(this->Interactor->GetLastEventPosition()[0], this->Interactor->GetLastEventPosition()[1]));
			
			if (this->CurrentRenderer == NULL)
			{
				return;
			}
		}

		this->PointWidget->SetCurrentRenderer(this->CurrentRenderer);
		this->PointWidget1->SetCurrentRenderer(this->CurrentRenderer);
		this->PointWidget2->SetCurrentRenderer(this->CurrentRenderer);

		this->Enabled = 1;

		vtkRenderWindowInteractor *i = this->Interactor;
		i->AddObserver(vtkCommand::MouseMoveEvent,
		this->EventCallbackCommand, this->Priority);
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

		this->CurrentRenderer->AddActor(this->LineActor);
		this->LineActor->SetProperty(this->LineProperty);

		this->CurrentRenderer->AddActor(PositionHandle);
		this->CurrentRenderer->AddActor(TargetHandle);

		this->PositionHandle->SetProperty(this->HandleProperty);
		this->TargetHandle->SetProperty(this->HandleProperty);

		this->BuildRepresentation();
		this->SizeHandles();

		this->InvokeEvent(vtkCommand::EnableEvent,NULL);
	}
	else
	{
		vtkDebugMacro(<<"Disabling line widget");

		if (!this->Enabled)
			return;

		this->Enabled = 0;

		this->Interactor->RemoveObserver(this->EventCallbackCommand);

		this->CurrentRenderer->RemoveActor(this->LineActor);

		this->CurrentRenderer->RemoveActor(PositionHandle);
		this->CurrentRenderer->RemoveActor(TargetHandle);

		if (this->CurrentPointWidget)
		{
			this->CurrentPointWidget->EnabledOff();
		}

		this->CurrentHandle = NULL;
		this->InvokeEvent(vtkCommand::DisableEvent,NULL);
		this->SetCurrentRenderer(NULL);
	}

	this->Interactor->Render();
}

void vtkErAreaLightWidget::PlaceWidget(double Bounds[6])
{
}

void vtkErAreaLightWidget::CreateDefaultProperties(void)
{
	this->HandleProperty = vtkProperty::New();
	this->HandleProperty->SetColor(1,0.6,0.2);

	this->SelectedHandleProperty = vtkProperty::New();
	this->SelectedHandleProperty->SetColor(1,0.7,0.3);

	this->LineProperty = vtkProperty::New();
//	this->LineProperty->SetRepresentationToWireframe();
	this->LineProperty->SetAmbient(1.0);
	this->LineProperty->SetAmbientColor(1,0.6,0.2);
	this->LineProperty->SetLineWidth(1.0);
//	this->LineProperty->SetLineStipplePattern(0);

	this->SelectedLineProperty = vtkProperty::New();
//	this->SelectedLineProperty->SetRepresentationToWireframe();
	this->SelectedLineProperty->SetAmbient(1.0);
	this->SelectedLineProperty->SetAmbientColor(1,0.6,0.2);
	this->SelectedLineProperty->SetLineWidth(1.0);
}

void vtkErAreaLightWidget::BuildRepresentation()
{
  //int res = this->LineSource->GetResolution();
  double *pt1 = this->LineSource->GetPoint1();
  double *pt2 = this->LineSource->GetPoint2();

  this->PositionHandleGeometry->SetCenter(pt1);
  this->TargetHandleGeometry->SetCenter(pt2);
}

void vtkErAreaLightWidget::OnLeftButtonDown(void)
{
	int forward=0;

	int X = this->Interactor->GetEventPosition()[0];
	int Y = this->Interactor->GetEventPosition()[1];

	// Okay, make sure that the pick is in the current renderer
	if (!this->CurrentRenderer || !this->CurrentRenderer->IsInViewport(X, Y))
	{
		this->State = vtkErAreaLightWidget::Outside;
		return;
	}

  // Okay, we can process this. Try to pick handles first;
  // if no handles picked, then try to pick the line.
  vtkAssemblyPath *path;
  this->HandlePicker->Pick(X,Y,0.0,this->CurrentRenderer);
  path = this->HandlePicker->GetPath();
  if ( path != NULL )
    {
    this->EventCallbackCommand->SetAbortFlag(1);
    this->StartInteraction();
    this->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
    this->State = vtkErAreaLightWidget::MovingHandle;
    this->HighlightHandle(path->GetFirstNode()->GetViewProp());
    this->EnablePointWidget();
    forward = this->ForwardEvent(vtkCommand::LeftButtonPressEvent);
    }
  else
    {
    this->LinePicker->Pick(X,Y,0.0,this->CurrentRenderer);
    path = this->LinePicker->GetPath();
    if ( path != NULL )
      {
      this->EventCallbackCommand->SetAbortFlag(1);
      this->StartInteraction();
      this->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
      this->State = vtkErAreaLightWidget::MovingLine;
      this->HighlightLine(1);
      this->EnablePointWidget();
      forward = this->ForwardEvent(vtkCommand::LeftButtonPressEvent);
      }
    else
      {
      this->State = vtkErAreaLightWidget::Outside;
      this->HighlightHandle(NULL);
      return;
      }
    }

  if ( ! forward )
    {
    this->Interactor->Render();
    }
}

void vtkErAreaLightWidget::OnLeftButtonUp(void)
{
	if (this->State == vtkErAreaLightWidget::Outside || this->State == vtkErAreaLightWidget::Start)
	{
		return;
	}

  this->State = vtkErAreaLightWidget::Start;
  this->HighlightHandle(NULL);
  this->HighlightLine(0);

  this->SizeHandles();

  int forward = this->ForwardEvent(vtkCommand::LeftButtonReleaseEvent);
  this->DisablePointWidget();

  this->EventCallbackCommand->SetAbortFlag(1);
  this->EndInteraction();
  this->InvokeEvent(vtkCommand::EndInteractionEvent,NULL);
	if (!forward)
	{
	this->Interactor->Render();
	}
}

void vtkErAreaLightWidget::OnMiddleButtonDown()
{
	int forward=0;

	int X = this->Interactor->GetEventPosition()[0];
	int Y = this->Interactor->GetEventPosition()[1];

	// Okay, make sure that the pick is in the current renderer
	if (!this->CurrentRenderer || !this->CurrentRenderer->IsInViewport(X, Y))
	{
		this->State = vtkErAreaLightWidget::Outside;
		return;
	}

  // Okay, we can process this. Try to pick handles first;
  // if no handles picked, then pick the bounding box.
  vtkAssemblyPath *path;
  this->HandlePicker->Pick(X,Y,0.0,this->CurrentRenderer);
  path = this->HandlePicker->GetPath();
  if (path != NULL)
    {
    this->EventCallbackCommand->SetAbortFlag(1);
    this->StartInteraction();
    this->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
    this->State = vtkErAreaLightWidget::MovingLine;
    this->HighlightHandles(1);
    this->HighlightLine(1);
    this->EnablePointWidget();
    this->ForwardEvent(vtkCommand::LeftButtonPressEvent);
    }
  else
    {
    this->LinePicker->Pick(X,Y,0.0,this->CurrentRenderer);
    path = this->LinePicker->GetPath();
    if ( path != NULL )
      {
      this->EventCallbackCommand->SetAbortFlag(1);
      this->StartInteraction();
      this->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
      //The highlight methods set the LastPickPosition, so they are ordered
      this->HighlightHandles(1);
      this->HighlightLine(1);
      this->State = vtkErAreaLightWidget::MovingLine;
      this->EnablePointWidget();
      this->ForwardEvent(vtkCommand::LeftButtonPressEvent);
      }
    else
      {
      this->State = vtkErAreaLightWidget::Outside;
      return;
      }
    }

  if ( ! forward )
    {
    this->Interactor->Render();
    }
}

void vtkErAreaLightWidget::OnMiddleButtonUp()
{
  if (this->State == vtkErAreaLightWidget::Outside || this->State == vtkErAreaLightWidget::Start)
    {
    return;
    }

  this->State = vtkErAreaLightWidget::Start;
  this->HighlightLine(0);
  this->HighlightHandles(0);

  this->SizeHandles();

  int forward = this->ForwardEvent(vtkCommand::LeftButtonReleaseEvent);
  this->DisablePointWidget();

  this->EventCallbackCommand->SetAbortFlag(1);
  this->EndInteraction();
  this->InvokeEvent(vtkCommand::EndInteractionEvent,NULL);
  if ( ! forward )
    {
    this->Interactor->Render();
    }
}

void vtkErAreaLightWidget::OnRightButtonDown()
{
  int X = this->Interactor->GetEventPosition()[0];
  int Y = this->Interactor->GetEventPosition()[1];

  // Okay, make sure that the pick is in the current renderer
  if (!this->CurrentRenderer || !this->CurrentRenderer->IsInViewport(X, Y))
    {
    this->State = vtkErAreaLightWidget::Outside;
    return;
    }

  // Okay, we can process this. Try to pick handles first;
  // if no handles picked, then pick the bounding box.
  vtkAssemblyPath *path;
  this->HandlePicker->Pick(X,Y,0.0,this->CurrentRenderer);
  path = this->HandlePicker->GetPath();
  if ( path != NULL )
    {
    this->HighlightLine(1);
    this->HighlightHandles(1);
    this->State = vtkErAreaLightWidget::Scaling;
    }
  else
    {
    this->LinePicker->Pick(X,Y,0.0,this->CurrentRenderer);
    path = this->LinePicker->GetPath();
    if ( path != NULL )
      {
      this->HighlightHandles(1);
      this->HighlightLine(1);
      this->State = vtkErAreaLightWidget::Scaling;
      }
    else
      {
      this->State = vtkErAreaLightWidget::Outside;
      this->HighlightLine(0);
      return;
      }
    }

  this->EventCallbackCommand->SetAbortFlag(1);
  this->StartInteraction();
  this->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
  this->Interactor->Render();
}

void vtkErAreaLightWidget::OnRightButtonUp()
{
  if (this->State == vtkErAreaLightWidget::Outside || this->State == vtkErAreaLightWidget::Start)
    {
    return;
    }

  this->State = vtkErAreaLightWidget::Start;
  this->HighlightLine(0);
  this->HighlightHandles(0);

  this->SizeHandles();

  this->EventCallbackCommand->SetAbortFlag(1);
  this->EndInteraction();
  this->InvokeEvent(vtkCommand::EndInteractionEvent,NULL);
  this->Interactor->Render();
}

//----------------------------------------------------------------------------
void vtkErAreaLightWidget::OnMouseMove()
{
  // See whether we're active
  if ( this->State == vtkErAreaLightWidget::Outside ||
       this->State == vtkErAreaLightWidget::Start )
    {
    return;
    }

  int X = this->Interactor->GetEventPosition()[0];
  int Y = this->Interactor->GetEventPosition()[1];

  // Do different things depending on state
  // Calculations everybody does
  double focalPoint[4], pickPoint[4], prevPickPoint[4];
  double z;

  vtkCamera *camera = this->CurrentRenderer->GetActiveCamera();
  if ( ! camera )
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
  int forward=0;
  if ( this->State == vtkErAreaLightWidget::MovingHandle )
    {
    forward = this->ForwardEvent(vtkCommand::MouseMoveEvent);
    }
  else if ( this->State == vtkErAreaLightWidget::MovingLine )
    {
    forward = this->ForwardEvent(vtkCommand::MouseMoveEvent);
    }
  else if ( this->State == vtkErAreaLightWidget::Scaling )
    {
 //   this->Scale(prevPickPoint, pickPoint, X, Y);
    }

  // Interact, if desired
  this->EventCallbackCommand->SetAbortFlag(1);
  this->InvokeEvent(vtkCommand::InteractionEvent,NULL);
  if ( ! forward )
    {
    this->Interactor->Render();
    }
}

void vtkErAreaLightWidget::SizeHandles(void)
{
	double radius = this->vtk3DWidget::SizeHandles(1.0);
	this->PositionHandleGeometry->SetRadius(radius);
	this->TargetHandleGeometry->SetRadius(radius);
}

int vtkErAreaLightWidget::HighlightHandle(vtkProp* pProp)
{
	// first unhighlight anything picked
	if ( this->CurrentHandle )
	{
		this->CurrentHandle->SetProperty(this->HandleProperty);
	}

	// set the current handle
	this->CurrentHandle = static_cast<vtkActor *>(pProp);

	// find the current handle
	if ( this->CurrentHandle )
	{
		this->ValidPick = 1;
		this->HandlePicker->GetPickPosition(this->LastPickPosition);
		this->CurrentHandle->SetProperty(this->SelectedHandleProperty);
		return (this->CurrentHandle == this->PositionHandle ? 0 : 1);
	}
	return -1;
}

void vtkErAreaLightWidget::HighlightHandles(int Highlight)
{
	if (Highlight)
	{
		this->ValidPick = 1;
		this->HandlePicker->GetPickPosition(this->LastPickPosition);
		this->PositionHandle->SetProperty(this->SelectedHandleProperty);
		this->TargetHandle->SetProperty(this->SelectedHandleProperty);
	}
	else
	{
		this->PositionHandle->SetProperty(this->HandleProperty);
		this->TargetHandle->SetProperty(this->HandleProperty);
	}
}

int vtkErAreaLightWidget::ForwardEvent(unsigned long event)
{
	if (!this->CurrentPointWidget)
	{
		return 0;
	}

	this->CurrentPointWidget->ProcessEvents(this, event, this->CurrentPointWidget,NULL);

	return 1;
}

void vtkErAreaLightWidget::EnablePointWidget()
{
  // Set up the point widgets
  double x[3];
  if ( this->CurrentHandle ) //picking the handles
    {
    if ( this->CurrentHandle == this->PositionHandle)
      {
      this->CurrentPointWidget = this->PointWidget1;
      this->LineSource->GetPoint1(x);
      }
    else
      {
      this->CurrentPointWidget = this->PointWidget2;
      this->LineSource->GetPoint2(x);
      }
    }
  else //picking the line
    {
    this->CurrentPointWidget = this->PointWidget;
    this->LinePicker->GetPickPosition(x);
    this->LastPosition[0] = x[0];
    this->LastPosition[1] = x[1];
    this->LastPosition[2] = x[2];
    }

  double bounds[6];
  for (int i=0; i<3; i++)
    {
    bounds[2*i] = x[i] - 0.1*this->InitialLength;
    bounds[2*i+1] = x[i] + 0.1*this->InitialLength;
    }

  // Note: translation mode is disabled and enabled to control
  // the proper positioning of the bounding box.
  this->CurrentPointWidget->SetInteractor(this->Interactor);
  this->CurrentPointWidget->TranslationModeOff();
  this->CurrentPointWidget->SetPlaceFactor(1.0);
  this->CurrentPointWidget->PlaceWidget(bounds);
  this->CurrentPointWidget->TranslationModeOn();
  this->CurrentPointWidget->SetPosition(x);
  this->CurrentPointWidget->SetCurrentRenderer(this->CurrentRenderer);
  this->CurrentPointWidget->On();
}

void vtkErAreaLightWidget::DisablePointWidget()
{
  if (this->CurrentPointWidget)
    {
    this->CurrentPointWidget->Off();
    }
  this->CurrentPointWidget = NULL;
}

void vtkErAreaLightWidget::HighlightLine(int Highlight)
{
  if ( Highlight )
    {
    this->ValidPick = 1;
    this->LinePicker->GetPickPosition(this->LastPickPosition);
    this->LineActor->SetProperty(this->SelectedLineProperty);
    }
  else
    {
    this->LineActor->SetProperty(this->LineProperty);
    }
}

void vtkErAreaLightWidget::ProcessEvents(vtkObject* vtkNotUsed(object), unsigned long event, void* clientdata, void* vtkNotUsed(calldata))
{
  vtkErAreaLightWidget* self = reinterpret_cast<vtkErAreaLightWidget *>( clientdata );

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

void vtkErAreaLightWidget::SetPosition(double x, double y, double z)
{
  double xyz[3];
  xyz[0] = x; xyz[1] = y; xyz[2] = z;

  /*
  if ( this->ClampToBounds )
    {
    this->ClampPosition(xyz);
    this->PointWidget1->SetPosition(xyz);
    }
	*/
  this->PointWidget1->SetPosition(xyz);
  this->LineSource->SetPoint1(xyz);
  this->BuildRepresentation();
}

//----------------------------------------------------------------------------
void vtkErAreaLightWidget::SetTarget(double x, double y, double z)
{
  double xyz[3];
  xyz[0] = x; xyz[1] = y; xyz[2] = z;

  /*
  if ( this->ClampToBounds )
    {
    this->ClampPosition(xyz);
    this->PointWidget2->SetPosition(xyz);
    }
	*/

	this->PointWidget2->SetPosition(xyz);
	this->LineSource->SetPoint2(xyz);
	this->BuildRepresentation();
}