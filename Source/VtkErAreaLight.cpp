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

#include <vtkCamera.h>

vtkStandardNewMacro(vtkErAreaLight);

vtkErAreaLight::vtkErAreaLight()
{
	this->Transform = vtkTransform::New();
	this->Transform->Identity();

	this->Up[0] = 0;
	this->Up[1] = 1;
	this->Up[2] = 0;

	this->Size[0] = 100;
	this->Size[1] = 100;
	this->Size[2] = 100;

	this->Camera = NULL;

	this->Type			= vtkErAreaLight::DefaultType();
	this->ShapeType		= vtkErAreaLight::DefaultShapeType();
	this->OneSided		= vtkErAreaLight::DefaultOneSided();
	this->Elevation		= vtkErAreaLight::DefaultElevation();
	this->Azimuth		= vtkErAreaLight::DefaultAzimuth();
	this->Distance		= vtkErAreaLight::DefaultDistance();
	this->Offset		= vtkErAreaLight::DefaultOffset();
	this->InnerRadius	= vtkErAreaLight::DefaultInnerRadius();
	this->OuterRadius	= vtkErAreaLight::DefaultOuterRadius();
}

vtkErAreaLight::~vtkErAreaLight()
{
}

vtkMatrix4x4* vtkErAreaLight::GetTransformMatrix()
{
	this->Transform->Identity();

	switch (this->Type)
	{
		case 0:
//		case 2:
		{
			const double Elevation = vtkMath::RadiansFromDegrees(this->Elevation);
			const double Azimuth   = vtkMath::RadiansFromDegrees(this->Azimuth);

			const double Direction[3] = { cos(Elevation) * sin(Azimuth), sin(Elevation), cos(Elevation) * cos(Azimuth) };

			const double Position[3] = { this->FocalPoint[0] + this->Distance * Direction[0], this->FocalPoint[1] + this->Distance * Direction[1], this->FocalPoint[2] + this->Distance * Direction[2] };

			double U[3], V[3], W[3];

			vtkMath::Subtract(this->FocalPoint, Position, W);
			vtkMath::Normalize(W);

			vtkMath::Cross(W, this->Up, U);
			vtkMath::Normalize(U);

			vtkMath::Cross(U, W, V);
			vtkMath::Normalize(V);

			this->Transform->Translate(Position);

			this->Transform->GetMatrix()->SetElement(0, 0, U[0]);
			this->Transform->GetMatrix()->SetElement(1, 0, U[1]);
			this->Transform->GetMatrix()->SetElement(2, 0, U[2]);

			this->Transform->GetMatrix()->SetElement(0, 1, V[0]);
			this->Transform->GetMatrix()->SetElement(1, 1, V[1]);
			this->Transform->GetMatrix()->SetElement(2, 1, V[2]);

			this->Transform->GetMatrix()->SetElement(0, 2, W[0]);
			this->Transform->GetMatrix()->SetElement(1, 2, W[1]);
			this->Transform->GetMatrix()->SetElement(2, 2, W[2]);

			this->Transform->Scale(1, 1, 1);

			if (this->Type == 0)
				return this->Transform->GetMatrix();

			vtkSmartPointer<vtkTransform> CameraTransform = vtkTransform::New();

			CameraTransform->Identity();

			vtkMath::Subtract(this->Camera->GetFocalPoint(), this->Camera->GetPosition(), W);
			vtkMath::Normalize(W);

			vtkMath::Cross(W, this->Camera->GetViewUp(), U);
			vtkMath::Normalize(U);

			vtkMath::Cross(U, W, V);
			vtkMath::Normalize(V);

			CameraTransform->Translate(this->Camera->GetPosition());

			CameraTransform->GetMatrix()->SetElement(0, 0, U[0]);
			CameraTransform->GetMatrix()->SetElement(1, 0, U[1]);
			CameraTransform->GetMatrix()->SetElement(2, 0, U[2]);

			CameraTransform->GetMatrix()->SetElement(0, 1, V[0]);
			CameraTransform->GetMatrix()->SetElement(1, 1, V[1]);
			CameraTransform->GetMatrix()->SetElement(2, 1, V[2]);

			CameraTransform->GetMatrix()->SetElement(0, 2, W[0]);
			CameraTransform->GetMatrix()->SetElement(1, 2, W[1]);
			CameraTransform->GetMatrix()->SetElement(2, 2, W[2]);

			this->Transform->PostMultiply();

			// vtkMatrix4x4::Multiply4x4(this->Transform->GetMatrix(), CameraTransform->GetMatrix(), this->Transform->GetMatrix());
			this->Transform->Concatenate(CameraTransform->GetMatrix());
				

			return CameraTransform->GetMatrix();//this->Transform->GetMatrix();
		}

		case 1:
		{
			double U[3], V[3], W[3];

			vtkMath::Subtract(this->FocalPoint, this->Position, W);
			vtkMath::Normalize(W);

			vtkMath::Cross(W, this->Up, U);
			vtkMath::Normalize(U);

			vtkMath::Cross(U, W, V);
			vtkMath::Normalize(V);

			this->Transform->Translate(this->Position);

			this->Transform->GetMatrix()->SetElement(0, 0, U[0]);
			this->Transform->GetMatrix()->SetElement(1, 0, U[1]);
			this->Transform->GetMatrix()->SetElement(2, 0, U[2]);

			this->Transform->GetMatrix()->SetElement(0, 1, V[0]);
			this->Transform->GetMatrix()->SetElement(1, 1, V[1]);
			this->Transform->GetMatrix()->SetElement(2, 1, V[2]);

			this->Transform->GetMatrix()->SetElement(0, 2, W[0]);
			this->Transform->GetMatrix()->SetElement(1, 2, W[1]);
			this->Transform->GetMatrix()->SetElement(2, 2, W[2]);

//			this->Transform->Scale(this->Scale);

			return this->Transform->GetMatrix();
		}
	}

	return this->Transform->GetMatrix();
}

double vtkErAreaLight::GetArea() const
{
	switch (this->Type)
	{
		case 0:
		{
			return this->Size[0] * this->Size[1];
		}

		case 1:
		{
			return vtkMath::Pi() * (this->OuterRadius * this->OuterRadius);
		}

		case 2:
		{
			return (vtkMath::Pi() * (this->OuterRadius * this->OuterRadius)) - (vtkMath::Pi() * (this->InnerRadius * this->InnerRadius));
		}

		case 3:
		{
			return (2 * this->Size[0] * this->Size[1]) + (2 * this->Size[0] * this->Size[2]) + (2 * this->Size[1] * this->Size[2]);
		}

		case 4:
		{
			return 4 * vtkMath::Pi() * (this->OuterRadius * this->OuterRadius);
		}
	}

	return 0;
}

int vtkErAreaLight::DefaultType()
{
	return 0;
}

int vtkErAreaLight::DefaultShapeType()
{
	return 0;
}

bool vtkErAreaLight::DefaultOneSided()
{
	return false;
}

double vtkErAreaLight::DefaultElevation()
{
	return 0;
}

double vtkErAreaLight::DefaultAzimuth()
{	
	return 0;
}

double vtkErAreaLight::DefaultDistance()
{
	return 1000;
}

double vtkErAreaLight::DefaultOffset()
{
	return 0;
}

double vtkErAreaLight::DefaultInnerRadius()
{
	return 50;
}

double vtkErAreaLight::DefaultOuterRadius()
{
	return 100;
}