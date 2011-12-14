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

#include "vtkErAreaLight.h"

vtkStandardNewMacro(vtkErAreaLight);

vtkErAreaLight::vtkErAreaLight()
{
	this->TransformMatrix = vtkTransform::New();
	this->TransformMatrix->Identity();
}

vtkErAreaLight::~vtkErAreaLight()
{
}

vtkTransform* vtkErAreaLight::GetTransform()
{
	double N[3];
	double U[3];
	double V[3];

	vtkMath::Subtract(this->GetFocalPoint(), this->GetPosition(), N);
	vtkMath::Normalize(N);

	vtkMath::Cross(N, this->GetUp(), U);
	vtkMath::Normalize(U);

	vtkMath::Cross(N, U, V);
	vtkMath::Normalize(V);
	
	/*
	TransformMatrix->SetElement(0, 0, U[0]);
	TransformMatrix->SetElement(1, 0, U[1]);
	TransformMatrix->SetElement(2, 0, U[2]);

	TransformMatrix->SetElement(0, 1, V[0]);
	TransformMatrix->SetElement(1, 1, V[1]);
	TransformMatrix->SetElement(2, 1, V[2]);

	TransformMatrix->SetElement(0, 2, N[0]);
	TransformMatrix->SetElement(1, 2, N[1]);
	TransformMatrix->SetElement(2, 2, N[2]);

	TransformMatrix->SetElement(3, 0, this->GetPosition()[0]);
	TransformMatrix->SetElement(3, 1, this->GetPosition()[1]);
	TransformMatrix->SetElement(3, 2, this->GetPosition()[2]);

	TransformMatrix->SetElement(3, 3, 1);

	TransformMatrix->SetElement(0, 3, 0);
	TransformMatrix->SetElement(1, 3, 0);
	TransformMatrix->SetElement(2, 3, 0);
	
	
	TransformMatrix->Identity();
	TransformMatrix->GetMatrix()->SetElement(0, 3, 1000);
	TransformMatrix->GetMatrix()->SetElement(1, 3, 0);
	TransformMatrix->GetMatrix()->SetElement(2, 3, 0);
*/
//	TransformMatrix->Translate(this->GetPosition()[0], this->GetPosition()[1], this->GetPosition()[2]);
//	TransformMatrix->Scale(this->Scale);

	return TransformMatrix;
}