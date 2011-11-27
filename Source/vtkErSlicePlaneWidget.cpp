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

#include "vtkErSlicePlaneWidget.h"
#include "vtkErSlicePlane.h"

vtkStandardNewMacro(vtkErSlicePlaneWidget);

vtkErSlicePlaneWidget::vtkErSlicePlaneWidget()
{
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

	// Construct the poly data representing the hex
	this->HexPolyData	= vtkPolyData::New();
	this->HexMapper		= vtkPolyDataMapper::New();
	this->HexActor		= vtkActor::New();

	this->HexMapper->SetInput(HexPolyData);
	this->HexActor->SetMapper(this->HexMapper);

	// Construct initial points
	this->Points = vtkPoints::New(VTK_DOUBLE);
	this->Points->SetNumberOfPoints(15);//8 corners; 6 faces; 1 center
	this->HexPolyData->SetPoints(this->Points);
  
	// Construct connectivity for the faces. These are used to perform
	// the picking.
	vtkIdType pts[4];
	vtkCellArray *cells = vtkCellArray::New();
	cells->Allocate(cells->EstimateSize(6,4));
	pts[0] = 3; pts[1] = 0; pts[2] = 4; pts[3] = 7;
	cells->InsertNextCell(4,pts);
	pts[0] = 1; pts[1] = 2; pts[2] = 6; pts[3] = 5;
	cells->InsertNextCell(4,pts);
	pts[0] = 0; pts[1] = 1; pts[2] = 5; pts[3] = 4;
	cells->InsertNextCell(4,pts);
	pts[0] = 2; pts[1] = 3; pts[2] = 7; pts[3] = 6;
	cells->InsertNextCell(4,pts);
	pts[0] = 0; pts[1] = 3; pts[2] = 2; pts[3] = 1;
	cells->InsertNextCell(4,pts);
	pts[0] = 4; pts[1] = 5; pts[2] = 6; pts[3] = 7;
	cells->InsertNextCell(4,pts);
	this->HexPolyData->SetPolys(cells);
	cells->Delete();
	this->HexPolyData->BuildCells();
  
	// The face of the hexahedra
	cells = vtkCellArray::New();
	cells->Allocate(cells->EstimateSize(1,4));
	cells->InsertNextCell(4,pts); //temporary, replaced later
	this->HexFacePolyData = vtkPolyData::New();
	this->HexFacePolyData->SetPoints(this->Points);
	this->HexFacePolyData->SetPolys(cells);
	this->HexFaceMapper = vtkPolyDataMapper::New();
	this->HexFaceMapper->SetInput(HexFacePolyData);
	this->HexFace = vtkActor::New();
	this->HexFace->SetMapper(this->HexFaceMapper);
	cells->Delete();

	//Manage the picking stuff
	this->HandlePicker = vtkCellPicker::New();
	this->HandlePicker->SetTolerance(0.001);
	
	for (int i = 0; i < 7; i++)
	{
//		this->HandlePicker->AddPickList(this->Handle[i]);
	}
	
	this->HandlePicker->PickFromListOn();

	this->HexPicker = vtkCellPicker::New();
	this->HexPicker->SetTolerance(0.001);
	this->HexPicker->AddPickList(HexActor);
	this->HexPicker->PickFromListOn();
  
	this->CurrentHandle = NULL;
}

vtkErSlicePlaneWidget::~vtkErSlicePlaneWidget()
{
}

void vtkErSlicePlaneWidget::SetVolume(vtkVolume* pVolume)
{
	if (pVolume == NULL)
	{
		vtkErrorMacro("This widget needs a valid volume!");
		return;
	}

	this->Volume = pVolume;

	this->BoundingBoxSource->SetBounds(pVolume->GetMapper()->GetBounds());

	double* pBounds = pVolume->GetMapper()->GetBounds();

	this->BoundingBoxPoints->SetPoint(0, pBounds[0], pBounds[2], pBounds[4]);
	this->BoundingBoxPoints->SetPoint(1, pBounds[1], pBounds[2], pBounds[4]);
	this->BoundingBoxPoints->SetPoint(2, pBounds[0], pBounds[3], pBounds[4]);
	this->BoundingBoxPoints->SetPoint(3, pBounds[1], pBounds[3], pBounds[4]);
	this->BoundingBoxPoints->SetPoint(4, pBounds[0], pBounds[2], pBounds[5]);
	this->BoundingBoxPoints->SetPoint(5, pBounds[1], pBounds[2], pBounds[5]);
	this->BoundingBoxPoints->SetPoint(6, pBounds[0], pBounds[3], pBounds[5]);
	this->BoundingBoxPoints->SetPoint(7, pBounds[1], pBounds[3], pBounds[5]);

	this->BoundingBoxPointPolyData->SetPoints(this->BoundingBoxPoints);
	this->BoundingBoxPointPolyData->Update();

	this->BoundingBoxPointPolyData->GetPointData()->SetScalars(this->BoundingBoxPoints->GetData());

	this->BoundingBoxPointLabelMapper->SetLabelModeToLabelScalars();
	this->BoundingBoxPointLabelMapper->SetInput(this->BoundingBoxPointPolyData);
	this->BoundingBoxPointLabelMapper->SetLabelFormat("%0.1f");
	this->BoundingBoxPointLabelMapper->Update();
}

void vtkErSlicePlaneWidget::SetEnabled(int enabling)
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

		this->InvokeEvent(vtkCommand::EnableEvent,NULL);

		this->CurrentRenderer->AddActor(this->BoundingBoxActor);
		this->CurrentRenderer->AddActor(this->BoundingBoxPointActor);
		this->CurrentRenderer->AddActor(this->BoundingBoxPointLabelActor);
    }
	else
    {
		vtkDebugMacro(<<"Disabling plane widget");

		if (!this->Enabled)
			return;
    
		this->Enabled = 0;

		this->Interactor->RemoveObserver(this->EventCallbackCommand);

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
}

void vtkErSlicePlaneWidget::PositionHandles()
{
}

void vtkErSlicePlaneWidget::OnLeftButtonDown()
{
}

void vtkErSlicePlaneWidget::OnLeftButtonUp()
{
}

void vtkErSlicePlaneWidget::OnMiddleButtonDown()
{
}

void vtkErSlicePlaneWidget::OnMiddleButtonUp()
{
}

void vtkErSlicePlaneWidget::OnRightButtonDown()
{
}

void vtkErSlicePlaneWidget::OnRightButtonUp()
{
}

void vtkErSlicePlaneWidget::OnMouseMove()
{
}

void vtkErSlicePlaneWidget::CreateDefaultProperties()
{
}

void vtkErSlicePlaneWidget::PlaceWidget(double bds[6])
{
}

void vtkErSlicePlaneWidget::UpdatePlacement(void)
{
}
