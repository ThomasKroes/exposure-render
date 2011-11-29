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

#include "vtkErPlaneSource.h"

vtkStandardNewMacro(vtkErPlaneSource);

// Construct plane perpendicular to z-axis, resolution 1x1, width and height
// 1.0, and centered at the origin.
vtkErPlaneSource::vtkErPlaneSource()
{
	this->XResolution = 1;
	this->YResolution = 1;

	this->Position[0] = 0.0;
	this->Position[1] = 0.0;
	this->Position[2] = 0.0;

	this->Normal[0] = 1.0;
	this->Normal[1] = 0.0;
	this->Normal[2] = 0.0;

	this->Up[0] = 0.0;
	this->Up[1] = 1.0;
	this->Up[2] = 0.0;

	this->Size[0] = 10.0;
	this->Size[1] = 10.0;

	this->SetNumberOfInputPorts(0);
}

// Set the number of x-y subdivisions in the plane.
void vtkErPlaneSource::SetResolution(const int xR, const int yR)
{
  if ( xR != this->XResolution || yR != this->YResolution )
    {
    this->XResolution = xR;
    this->YResolution = yR;

    this->XResolution = (this->XResolution > 0 ? this->XResolution : 1);
    this->YResolution = (this->YResolution > 0 ? this->YResolution : 1);

    this->Modified();
    }
}

int vtkErPlaneSource::RequestData(vtkInformation* vtkNotUsed(request), vtkInformationVector** vtkNotUsed(inputVector), vtkInformationVector* outputVector)
{
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  // get the ouptut
  vtkPolyData *output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

  double x[3], tc[2], v1[3], v2[3];
  vtkIdType pts[4];
  int i, j, ii;
  int numPts;
  int numPolys;
  
	// Compute coordinate system
	vtkMath::Cross(this->Normal, this->Up, v1);
	vtkMath::Cross(this->Normal, v1, v2);

	// Points
	numPts	= (this->XResolution + 1) * (this->YResolution + 1);
	numPolys	= this->XResolution * this->YResolution;

	vtkSmartPointer<vtkPoints> newPoints = vtkPoints::New();
	newPoints->Allocate(numPts);
	
	// Normals
	vtkSmartPointer<vtkFloatArray> newNormals = vtkFloatArray::New();
	newNormals->SetNumberOfComponents(3);
	newNormals->Allocate(3*numPts);

	// Texture coordinates
	vtkSmartPointer<vtkFloatArray> newTCoords = vtkFloatArray::New();
	newTCoords->SetNumberOfComponents(2);
	newTCoords->Allocate(2*numPts);
	
	vtkSmartPointer<vtkCellArray> newPolys = vtkCellArray::New();
	newPolys->Allocate(newPolys->EstimateSize(numPolys,4));

	const double HalfSize[2]	= { 0.5 * this->Size[0], 0.5 * this->Size[1] };
	const double Origin[2]		= { -HalfSize[0], -HalfSize[1] };
	const double Delta[2]		= { this->Size[0] / (double)this->XResolution, this->Size[1] / (double)this->YResolution };

	// Generate points and point data
	for (numPts = 0, i = 0; i < (this->YResolution + 1); i++)
	{
		tc[1] = Origin[1] + ((double)i * Delta[1]);
		
		for (j = 0; j < (this->XResolution + 1); j++)
		{
			tc[0] = Origin[0] + ((double)j * Delta[0]);

			for (ii = 0; ii < 3; ii++)
			{
				x[ii] = this->Position[ii] + tc[0]*v1[ii] + tc[1]*v2[ii];
			}

			newPoints->InsertPoint(numPts,x);
			newTCoords->InsertTuple(numPts,tc);
			newNormals->InsertTuple(numPts++,this->Normal);
		}
	}

	// Generate polygon connectivity
	for (i=0; i<this->YResolution; i++)
	{
		for (j=0; j<this->XResolution; j++)
		{
			pts[0] = j + i*(this->XResolution+1);
			pts[1] = pts[0] + 1;
			pts[2] = pts[0] + this->XResolution + 2;
			pts[3] = pts[0] + this->XResolution + 1;
			newPolys->InsertNextCell(4,pts);
		}
	}

	output->SetPoints(newPoints);

	newNormals->SetName("Normals");
	output->GetPointData()->SetNormals(newNormals);

	newTCoords->SetName("TextureCoordinates");
	output->GetPointData()->SetTCoords(newTCoords);

	output->SetPolys(newPolys);

	return 1;
}

void vtkErPlaneSource::SetUp(double X, double Y, double Z)
{
	this->Up[0] = X;
	this->Up[1] = Y;
	this->Up[2] = Z;

	this->Modified();
}

void vtkErPlaneSource::SetSize(double Width, double Height)
{
	if (Width < 0)
	{
		vtkErrorMacro("Plane width cannot be negative!");
		return;
	}

	if (Height < 0)
	{
		vtkErrorMacro("Plane height cannot be negative!");
		return;
	}

	this->Size[0] = Width;
	this->Size[1] = Height;

	this->Modified();
}

