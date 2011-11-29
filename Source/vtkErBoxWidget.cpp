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

#include "Geometry.h"

#include "vtkErBoxWidget.h"

#define VTK_AVERAGE(a,b,c) \
  c[0] = (a[0] + b[0]) / 2.0; \
  c[1] = (a[1] + b[1]) / 2.0; \
  c[2] = (a[2] + b[2]) / 2.0;

vtkStandardNewMacro(vtkErBoxWidget);

vtkErBoxWidget::vtkErBoxWidget()
{
	this->State = vtkErBoxWidget::Start;
	this->EventCallbackCommand->SetCallback(vtkErBoxWidget::ProcessEvents);

	// Bounding box lines
	this->BoundingBoxActor				= vtkActor::New();
	this->BoundingBoxMapper				= vtkPolyDataMapper::New();
	this->BoundingBoxSource				= vtkCubeSource::New();
	this->BoundingBoxProperty			= vtkProperty::New();
	
	this->BoundingBoxActor->SetMapper(BoundingBoxMapper);
	this->BoundingBoxActor->SetProperty(this->BoundingBoxProperty);
	this->BoundingBoxMapper->SetInput(BoundingBoxSource->GetOutput());
	this->BoundingBoxProperty->SetRepresentationToWireframe();

	// Bounding box points and labels
	this->BoundingBoxPoints				= vtkPoints::New();
	this->BoundingBoxPointPolyData		= vtkPolyData::New();
	this->BoundingBoxPointMapper		= vtkPolyDataMapper::New();
	this->BoundingBoxPointActor			= vtkActor::New();
	this->BoundingBoxPointProperty		= vtkProperty::New();
	this->BoundingBoxPointLabelMapper	= vtkLabeledDataMapper::New();
	this->BoundingBoxPointLabelActor	= vtkActor2D::New();
	this->BoundingBoxPointLabelProperty	= vtkProperty2D::New();

	this->BoundingBoxPoints->SetNumberOfPoints(8);
	this->BoundingBoxPointMapper->SetInput(this->BoundingBoxPointPolyData);
	this->BoundingBoxPointActor->SetProperty(this->BoundingBoxPointProperty);
	this->BoundingBoxPointActor->SetMapper(this->BoundingBoxPointMapper);
	this->BoundingBoxPointProperty->SetPointSize(100);
	this->BoundingBoxPointProperty->SetColor(0.9, 0.6, 0.1);
	this->BoundingBoxPointLabelActor->SetMapper(this->BoundingBoxPointLabelMapper);
	this->BoundingBoxPointLabelActor->SetProperty(this->BoundingBoxPointLabelProperty);

	// Widgets
	this->DefaultPositions	= vtkPoints::New();
	this->DefaultNormals	= vtkPoints::New();
	this->DefaultUp			= vtkPoints::New();

	this->DefaultPositions->SetNumberOfPoints(6);
	this->DefaultNormals->SetNumberOfPoints(6);
	this->DefaultUp->SetNumberOfPoints(6);

	for (int i = 0; i < 6; i++)
		this->PointWidget[i] = vtkErSlicePlaneWidget::New();
}

vtkErBoxWidget::~vtkErBoxWidget()
{
}

void vtkErBoxWidget::SetVolume(vtkVolume* pVolume)
{
	if (pVolume == NULL)
	{
		vtkErrorMacro("This widget needs a valid volume!");
		return;
	}

	this->Volume = pVolume;

	this->BoundingBoxSource->SetBounds(pVolume->GetMapper()->GetBounds());

	this->PlaceWidget(pVolume->GetMapper()->GetBounds());
}

void vtkErBoxWidget::SetEnabled(int enabling)
{
	if (!this->Interactor)
	{
		vtkErrorMacro(<<"The interactor must be set prior to enabling/disabling widget");
		return;
	}

	if (enabling)
    {
		vtkDebugMacro(<<"Enabling plane widget");

		if (this->Enabled)
			return;
    
		if (!this->CurrentRenderer)
		{
			this->SetCurrentRenderer(this->Interactor->FindPokedRenderer(this->Interactor->GetLastEventPosition()[0], this->Interactor->GetLastEventPosition()[1]));
			
			if (this->CurrentRenderer == NULL)
				return;
		}

		this->Enabled = 1;

		// Listen to the following events
		vtkRenderWindowInteractor* i = this->Interactor;

		i->AddObserver(vtkCommand::MouseMoveEvent, this->EventCallbackCommand, this->Priority);
		i->AddObserver(vtkCommand::LeftButtonPressEvent, this->EventCallbackCommand, this->Priority);
		i->AddObserver(vtkCommand::LeftButtonReleaseEvent, this->EventCallbackCommand, this->Priority);
		i->AddObserver(vtkCommand::MiddleButtonPressEvent, this->EventCallbackCommand, this->Priority);
		i->AddObserver(vtkCommand::MiddleButtonReleaseEvent, this->EventCallbackCommand, this->Priority);
		i->AddObserver(vtkCommand::RightButtonPressEvent, this->EventCallbackCommand, this->Priority);
		i->AddObserver(vtkCommand::RightButtonReleaseEvent, this->EventCallbackCommand, this->Priority);

		this->CurrentRenderer->AddActor(this->BoundingBoxActor);
		this->CurrentRenderer->AddActor(this->BoundingBoxPointActor);
		this->CurrentRenderer->AddActor(this->BoundingBoxPointLabelActor);

		for (int i = 0; i < 6; i++)
		{
			this->PointWidget[i]->SetInteractor(this->Interactor);
	//		this->PointWidget[i]->TranslationModeOff();
//			this->PointWidget[i]->SetPlaceFactor(1.0);
//			this->PointWidget[i]->PlaceWidget(this->Volume->GetBounds());
	//		this->PointWidget[i]->TranslationModeOn();
	//		this->PointWidget[i]->SetPosition(100, 100, 100);
			this->PointWidget[i]->SetCurrentRenderer(this->CurrentRenderer);
			this->PointWidget[i]->On();
		}

		this->InvokeEvent(vtkCommand::EnableEvent,NULL);
    }
	else
    {
		vtkDebugMacro(<<"Disabling plane widget");

		if (!this->Enabled)
			return;
    
		this->Enabled = 0;

		this->Interactor->RemoveObserver(this->EventCallbackCommand);

		this->CurrentRenderer->RemoveActor(this->BoundingBoxActor);
		this->CurrentRenderer->RemoveActor(this->BoundingBoxPointActor);
		this->CurrentRenderer->RemoveActor(this->BoundingBoxPointLabelActor);

		for (int i = 0; i < 6; i++)
		{
			this->PointWidget[i]->Off();
		}

		this->InvokeEvent(vtkCommand::DisableEvent,NULL);
		this->SetCurrentRenderer(NULL);
    }

	this->Interactor->Render();
}

void vtkErBoxWidget::ProcessEvents(vtkObject* vtkNotUsed(object), unsigned long event, void* clientdata, void* vtkNotUsed(calldata))
{
	vtkErBoxWidget* self = reinterpret_cast<vtkErBoxWidget *>( clientdata );

	switch (event)
	{
		case vtkCommand::LeftButtonPressEvent:		self->OnLeftButtonDown();		break;
		case vtkCommand::LeftButtonReleaseEvent:	self->OnLeftButtonUp();			break;
		case vtkCommand::MiddleButtonPressEvent:	self->OnMiddleButtonDown();		break;
		case vtkCommand::MiddleButtonReleaseEvent:	self->OnMiddleButtonUp();		break;
		case vtkCommand::RightButtonPressEvent:		self->OnRightButtonDown();		break;
		case vtkCommand::RightButtonReleaseEvent:	self->OnRightButtonUp();		break;
		case vtkCommand::MouseMoveEvent:			self->OnMouseMove();			break;
	}
}

void vtkErBoxWidget::PrintSelf(ostream& os, vtkIndent indent)
{
}

void vtkErBoxWidget::PositionHandles()
{
}

void vtkErBoxWidget::OnLeftButtonDown()
{
}

void vtkErBoxWidget::OnLeftButtonUp()
{
}

void vtkErBoxWidget::OnMiddleButtonDown()
{
}

void vtkErBoxWidget::OnMiddleButtonUp()
{
}

void vtkErBoxWidget::OnRightButtonDown()
{
}

void vtkErBoxWidget::OnRightButtonUp()
{
}

void vtkErBoxWidget::OnMouseMove()
{
}

void vtkErBoxWidget::CreateDefaultProperties()
{
}

void vtkErBoxWidget::PlaceWidget(double Bounds[6])
{
	vtkBoundingBox AABB(Bounds);

	this->BoundingBoxPoints->SetPoint(0, Bounds[0], Bounds[2], Bounds[4]);
	this->BoundingBoxPoints->SetPoint(1, Bounds[1], Bounds[2], Bounds[4]);
	this->BoundingBoxPoints->SetPoint(2, Bounds[0], Bounds[3], Bounds[4]);
	this->BoundingBoxPoints->SetPoint(3, Bounds[1], Bounds[3], Bounds[4]);
	this->BoundingBoxPoints->SetPoint(4, Bounds[0], Bounds[2], Bounds[5]);
	this->BoundingBoxPoints->SetPoint(5, Bounds[1], Bounds[2], Bounds[5]);
	this->BoundingBoxPoints->SetPoint(6, Bounds[0], Bounds[3], Bounds[5]);
	this->BoundingBoxPoints->SetPoint(7, Bounds[1], Bounds[3], Bounds[5]);

	this->BoundingBoxPointPolyData->SetPoints(this->BoundingBoxPoints);
	this->BoundingBoxPointPolyData->Update();

	this->BoundingBoxPointPolyData->GetPointData()->SetScalars(this->BoundingBoxPoints->GetData());

	this->BoundingBoxPointLabelMapper->SetLabelModeToLabelScalars();
	this->BoundingBoxPointLabelMapper->SetInput(this->BoundingBoxPointPolyData);
	this->BoundingBoxPointLabelMapper->SetLabelFormat("%0.1f");
	this->BoundingBoxPointLabelMapper->Update();

	const double HalfX = 0.5 * (Bounds[1] + Bounds[0]);
	const double HalfY = 0.5 * (Bounds[3] + Bounds[2]);
	const double HalfZ = 0.5 * (Bounds[5] + Bounds[4]);

	this->DefaultPositions->SetPoint(0, HalfX, HalfY, Bounds[4]);
	this->DefaultPositions->SetPoint(1, HalfX, HalfY, Bounds[5]);
	this->DefaultPositions->SetPoint(2, HalfX, Bounds[2], HalfZ);
	this->DefaultPositions->SetPoint(3, HalfX, Bounds[3], HalfZ);
	this->DefaultPositions->SetPoint(4, Bounds[0], HalfY, HalfZ);
	this->DefaultPositions->SetPoint(5, Bounds[1], HalfY, HalfZ);

	this->DefaultNormals->SetPoint(0, 0, 0, 1);
	this->DefaultNormals->SetPoint(1, 0, 0, -1);
	this->DefaultNormals->SetPoint(2, 0, 1, 0);
	this->DefaultNormals->SetPoint(3, 0, -1, 0);
	this->DefaultNormals->SetPoint(4, 1, 0, 0);
	this->DefaultNormals->SetPoint(5, -1, 0, 0);

	this->DefaultUp->SetPoint(0, 0, 1, 0);
	this->DefaultUp->SetPoint(1, 0, 1, 0);
	this->DefaultUp->SetPoint(2, 1, 0, 0);
	this->DefaultUp->SetPoint(3, 1, 0, 0);
	this->DefaultUp->SetPoint(4, 0, 1, 0);
	this->DefaultUp->SetPoint(5, 0, 1, 0);

	double Lengths[3];

	AABB.GetLengths(Lengths);

	const double MinLength = min(Lengths[0], min(Lengths[1], Lengths[2]));

	for (int i = 0; i < 6; i++)
	{
		this->PointWidget[i]->PlaceWidget(Bounds);
		this->PointWidget[i]->SetCenter(this->DefaultPositions->GetPoint(i));
		this->PointWidget[i]->SetNormal(this->DefaultNormals->GetPoint(i));
		this->PointWidget[i]->SetUp(this->DefaultUp->GetPoint(i));
		this->PointWidget[i]->SetPlaceFactor(1);
		this->PointWidget[i]->UpdatePlacement();
		this->PointWidget[i]->SetSize(0.75 * MinLength, 0.75 * MinLength);
	}
}

void vtkErBoxWidget::UpdatePlacement(void)
{
}

void vtkErBoxWidget::GetPlanes(vtkPlanes* pPlanes)
{
	if (!pPlanes)
		return;
	  
//	this->ComputeNormals();

	vtkSmartPointer<vtkPoints> pts = vtkPoints::New(VTK_DOUBLE);
	pts->SetNumberOfPoints(6);
  
	vtkSmartPointer<vtkDoubleArray> normals = vtkDoubleArray::New();
	normals->SetNumberOfComponents(3);
	normals->SetNumberOfTuples(6);
  
	// Set the normals and coordinate values
	double factor = 1.0;//(this->InsideOut ? -1.0 : 1.0);
	
	for (int i = 0; i < 6; i++)
	{
		pts->SetPoint(i, this->PointWidget[i]->GetCenter());
		normals->SetTuple3(i, factor * this->PointWidget[i]->GetNormal()[0], factor * this->PointWidget[i]->GetNormal()[1], factor * this->PointWidget[i]->GetNormal()[2]);
	}
    
	pPlanes->SetPoints(pts);
	pPlanes->SetNormals(normals);
}

vtkErSlicePlaneWidget* vtkErBoxWidget::GetSlicePlaneWidget(int Index)
{
	if (Index < 0 || Index >= 6)
	{
		vtkErrorMacro("Cannot return slice plane widget, index is out of bounds!");
		return NULL;
	}

	return this->PointWidget[Index];
}