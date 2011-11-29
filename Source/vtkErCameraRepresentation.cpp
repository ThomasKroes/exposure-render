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

#include "vtkErCameraRepresentation.h"

vtkStandardNewMacro(vtkErCameraRepresentation);

vtkCxxSetObjectMacro(vtkErCameraRepresentation, Camera, vtkCamera);

vtkErCameraRepresentation::vtkErCameraRepresentation()
{
  this->Camera = NULL;
  
  // Set up the 
  double size[2];
  this->GetSize(size);
  this->Position2Coordinate->SetValue(0.04*size[0], 0.04*size[1]);
  this->ProportionalResize = 1;
  this->Moving = 1;
  this->ShowBorder = vtkBorderRepresentation::BORDER_ON;

  // Create the geometry in canonical coordinates
  this->Points = vtkPoints::New();
  this->Points->SetDataTypeToDouble();
  this->Points->SetNumberOfPoints(25);
  this->Points->SetPoint(0, 0.0, 0.0, 0.0);
  this->Points->SetPoint(1, 6.0, 0.0, 0.0);
  this->Points->SetPoint(2, 6.0, 2.0, 0.0);
  this->Points->SetPoint(3, 0.0, 2.0, 0.0);
  this->Points->SetPoint(4, 0.375, 0.25, 0.0);
  this->Points->SetPoint(5, 1.0, 0.25, 0.0);
  this->Points->SetPoint(6, 1.0, 1.75, 0.0);
  this->Points->SetPoint(7, 0.375, 1.75, 0.0);
  this->Points->SetPoint(8, 1.0, 0.875, 0.0);
  this->Points->SetPoint(9, 1.25, 0.75, 0.0);
  this->Points->SetPoint(10, 1.5, 0.75, 0.0);
  this->Points->SetPoint(11, 1.5, 1.25, 0.0);
  this->Points->SetPoint(12, 1.25, 1.25, 0.0);
  this->Points->SetPoint(13, 1.0, 1.125, 0.0);
  this->Points->SetPoint(14, 2.5, 0.5, 0.0);
  this->Points->SetPoint(15, 3.5, 1.0, 0.0);
  this->Points->SetPoint(16, 2.5, 1.5, 0.0);
  this->Points->SetPoint(17, 4.625, 0.375, 0.0);
  this->Points->SetPoint(18, 5.625, 0.375, 0.0);
  this->Points->SetPoint(19, 5.75, 0.5, 0.0);
  this->Points->SetPoint(20, 5.75, 1.5, 0.0);
  this->Points->SetPoint(21, 5.625, 1.625, 0.0);
  this->Points->SetPoint(22, 4.625, 1.625, 0.0);
  this->Points->SetPoint(23, 4.5, 1.5, 0.0);
  this->Points->SetPoint(24, 4.5, 0.5, 0.0);

  vtkCellArray *cells = vtkCellArray::New();
  cells->InsertNextCell(4); //camera body
  cells->InsertCellPoint(4);
  cells->InsertCellPoint(5);
  cells->InsertCellPoint(6);
  cells->InsertCellPoint(7);
  cells->InsertNextCell(6); //camera lens
  cells->InsertCellPoint(8);
  cells->InsertCellPoint(9);
  cells->InsertCellPoint(10);
  cells->InsertCellPoint(11);
  cells->InsertCellPoint(12);
  cells->InsertCellPoint(13);
  cells->InsertNextCell(3); //play button
  cells->InsertCellPoint(14);
  cells->InsertCellPoint(15);
  cells->InsertCellPoint(16);
  cells->InsertNextCell(4); //part of delete button
  cells->InsertCellPoint(17);
  cells->InsertCellPoint(20);
  cells->InsertCellPoint(21);
  cells->InsertCellPoint(24);
  cells->InsertNextCell(4); //part of delete button
  cells->InsertCellPoint(18);
  cells->InsertCellPoint(19);
  cells->InsertCellPoint(22);
  cells->InsertCellPoint(23);
  this->PolyData = vtkPolyData::New();
  this->PolyData->SetPoints(this->Points);
  this->PolyData->SetPolys(cells);
  cells->Delete();

  this->TransformFilter = vtkTransformPolyDataFilter::New();
  this->TransformFilter->SetTransform(this->BWTransform);
  this->TransformFilter->SetInput(this->PolyData);

  this->Mapper = vtkPolyDataMapper2D::New();
  this->Mapper->SetInput(this->TransformFilter->GetOutput());
  this->Property = vtkProperty2D::New();
  this->Actor = vtkActor2D::New();
  this->Actor->SetMapper(this->Mapper);
  this->Actor->SetProperty(this->Property);
}

vtkErCameraRepresentation::~vtkErCameraRepresentation()
{
  this->SetCamera(0);
  
  this->Points->Delete();
  this->TransformFilter->Delete();
  this->PolyData->Delete();
  this->Mapper->Delete();
  this->Property->Delete();
  this->Actor->Delete();
}

void vtkErCameraRepresentation::BuildRepresentation()
{
  // Note that the transform is updated by the superclass
  this->Superclass::BuildRepresentation();
}

void vtkErCameraRepresentation::GetActors2D(vtkPropCollection *pc)
{
  pc->AddItem(this->Actor);
  this->Superclass::GetActors2D(pc);
}

void vtkErCameraRepresentation::ReleaseGraphicsResources(vtkWindow *w)
{
  this->Actor->ReleaseGraphicsResources(w);
  this->Superclass::ReleaseGraphicsResources(w);
}

int vtkErCameraRepresentation::RenderOverlay(vtkViewport *w)
{
  int count = this->Superclass::RenderOverlay(w);
  count += this->Actor->RenderOverlay(w);
  return count;
}

int vtkErCameraRepresentation::RenderOpaqueGeometry(vtkViewport *w)
{
  int count = this->Superclass::RenderOpaqueGeometry(w);
  count += this->Actor->RenderOpaqueGeometry(w);
  return count;
}

int vtkErCameraRepresentation::RenderTranslucentPolygonalGeometry(vtkViewport *w)
{
  int count = this->Superclass::RenderTranslucentPolygonalGeometry(w);
  count += this->Actor->RenderTranslucentPolygonalGeometry(w);
  return count;
}

int vtkErCameraRepresentation::HasTranslucentPolygonalGeometry()
{
  int result = this->Superclass::HasTranslucentPolygonalGeometry();
  result |= this->Actor->HasTranslucentPolygonalGeometry();
  return result;
}


