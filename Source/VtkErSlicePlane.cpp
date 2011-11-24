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

#include "vtkErSlicePlane.h"

#include <vtkObjectFactory.h>

vtkStandardNewMacro(vtkErSlicePlane);

vtkErSlicePlane::vtkErSlicePlane(void)
{
	SetEnabled(true);
};

vtkErSlicePlane::~vtkErSlicePlane(void)
{
};

double* vtkErSlicePlane::GetBounds(void)
{
	return NULL;
}

int vtkErSlicePlane::RenderOpaqueGeometry(vtkViewport *viewport)
{
	this->UpdateViewProps();
  
	int result=0;
  
	if (this->PlaneActor != NULL && this->PlaneActor->GetMapper() != NULL)
	{
		result = this->PlaneActor->RenderOpaqueGeometry(viewport);
	}
  
	return result;
}

int vtkErSlicePlane::HasTranslucentPolygonalGeometry()
{
	return false;
}

void vtkErSlicePlane::UpdateViewProps()
{
	if (this->PlaneSource == NULL)
		PlaneSource = vtkPlaneSource::New();

	if (this->PlaneMapper == NULL)
		PlaneMapper	= vtkPolyDataMapper::New();

	if (this->PlaneActor == NULL)
		PlaneActor	= vtkActor::New();

	this->PlaneSource->SetPoint1(100, -100, 0);
	this->PlaneSource->SetPoint2(-100, 100, 0);
	this->PlaneSource->SetOrigin(0, 0, 0);

	this->PlaneMapper->SetInputConnection(this->PlaneSource->GetOutputPort());
	this->PlaneActor->SetMapper(this->PlaneMapper);
//	this->ConeMapper->SetScalarVisibility(0);
}