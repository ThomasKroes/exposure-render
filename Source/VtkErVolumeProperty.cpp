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

#include "vtkErVolumeProperty.h"

#include <vtkObjectFactory.h>

vtkCxxRevisionMacro(vtkErVolumeProperty, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkErVolumeProperty);

vtkErVolumeProperty::vtkErVolumeProperty()
{
	this->Opacity		= vtkPiecewiseFunction::New();
	this->Diffuse[0]	= vtkPiecewiseFunction::New();
	this->Diffuse[1]	= vtkPiecewiseFunction::New();
	this->Diffuse[2]	= vtkPiecewiseFunction::New();
	this->Specular[0]	= vtkPiecewiseFunction::New();
	this->Specular[1]	= vtkPiecewiseFunction::New();
	this->Specular[2]	= vtkPiecewiseFunction::New();
	this->Roughness		= vtkPiecewiseFunction::New();
	this->Emission[0]	= vtkPiecewiseFunction::New();
	this->Emission[1]	= vtkPiecewiseFunction::New();
	this->Emission[2]	= vtkPiecewiseFunction::New();
}

vtkErVolumeProperty::~vtkErVolumeProperty()
{
}

void vtkErVolumeProperty::SetOpacity(vtkPiecewiseFunction* pPiecewiseFunction)
{
	if (!pPiecewiseFunction)
	{
		vtkErrorMacro("Opacity piecewise function is NULL!")
		return;
	}

	if (this->Opacity != pPiecewiseFunction)
	{
		this->Opacity = pPiecewiseFunction;
		this->Modified();
	}
}

vtkPiecewiseFunction* vtkErVolumeProperty::GetOpacity(void)
{
	return Opacity.GetPointer();
}

void vtkErVolumeProperty::SetDiffuse(int Index, vtkPiecewiseFunction* pPiecewiseFunction)
{
	if (Index < 0|| Index > 3)
	{
		vtkErrorMacro("Diffuse color index out of bounds!")
		return;
	}

	if (!pPiecewiseFunction)
	{
		vtkErrorMacro("Diffuse color piecewise function is NULL!")
		return;
	}

	if (this->Diffuse[Index] != pPiecewiseFunction)
	{
		this->Diffuse[Index] = pPiecewiseFunction;
		this->Modified();
	}
}

vtkPiecewiseFunction* vtkErVolumeProperty::GetDiffuse(int Index)
{
	if (Index < 0|| Index > 3)
	{
		vtkErrorMacro("Diffuse color index out of bounds!")
		return NULL;
	}

	return Diffuse[Index];
}

void vtkErVolumeProperty::SetSpecular(int Index, vtkPiecewiseFunction* pPiecewiseFunction)
{
	if (Index < 0|| Index > 3)
	{
		vtkErrorMacro("Specular color index out of bounds!")
		return;
	}

	if (!pPiecewiseFunction)
	{
		vtkErrorMacro("Specular color piecewise function is NULL!")
		return;
	}

	if (this->Specular[Index] != pPiecewiseFunction)
	{
		this->Specular[Index] = pPiecewiseFunction;
		this->Modified();
	}
}

vtkPiecewiseFunction* vtkErVolumeProperty::GetSpecular(int Index)
{
	if (Index < 0|| Index > 3)
	{
		vtkErrorMacro("Specular color index out of bounds!")
		return NULL;
	}

	return Specular[Index];
}

void vtkErVolumeProperty::SetRoughness(vtkPiecewiseFunction* pPiecewiseFunction)
{
	if (!pPiecewiseFunction)
	{
		vtkErrorMacro("Roughness piecewise function is NULL!")
		return;
	}

	if (this->Roughness != pPiecewiseFunction)
	{
		this->Roughness = pPiecewiseFunction;
		this->Modified();
	}
}

vtkPiecewiseFunction* vtkErVolumeProperty::GetRoughness(void)
{
	return Roughness.GetPointer();
}

void vtkErVolumeProperty::SetEmission(int Index, vtkPiecewiseFunction* pPiecewiseFunction)
{
	if (Index < 0|| Index > 3)
	{
		vtkErrorMacro("Emission color index out of bounds!")
		return;
	}

	if (!pPiecewiseFunction)
	{
		vtkErrorMacro("Emission color piecewise function is NULL!")
		return;
	}

	if (this->Emission[Index] != pPiecewiseFunction)
	{
		this->Emission[Index] = pPiecewiseFunction;
		this->Modified();
	}
}

vtkPiecewiseFunction* vtkErVolumeProperty::GetEmission(int Index)
{
	if (Index < 0|| Index > 3)
	{
		vtkErrorMacro("Emission color index out of bounds!")
		return NULL;
	}

	return Emission[Index];
}

void vtkErVolumeProperty::Default(double Min, double Max)
{
	this->Opacity->RemoveAllPoints();
	this->Diffuse[0]->RemoveAllPoints();
	this->Diffuse[1]->RemoveAllPoints();
	this->Diffuse[2]->RemoveAllPoints();
	this->Specular[0]->RemoveAllPoints();
	this->Specular[1]->RemoveAllPoints();
	this->Specular[2]->RemoveAllPoints();
	this->Roughness->RemoveAllPoints();
	this->Emission[0]->RemoveAllPoints();
	this->Emission[1]->RemoveAllPoints();
	this->Emission[2]->RemoveAllPoints();

	Opacity->AddPoint(Min, 0, 0.5, 0);
	Opacity->AddPoint(Max, 1, 0.5, 0);

	for (int i = 0; i < 3; i++)
	{
		Diffuse[0]->AddPoint(Min, 0.5, 0.5, 0);
		Diffuse[0]->AddPoint(Max, 0.5, 0.5, 0);
	}

	for (int i = 0; i < 3; i++)
	{
		Specular[0]->AddPoint(Min, 0.5, 0.5, 0);
		Specular[0]->AddPoint(Max, 0.5, 0.5, 0);
	}

	Roughness->AddPoint(Min, 1, 0.5, 0);
	Roughness->AddPoint(Max, 1, 0.5, 0);

	for (int i = 0; i < 3; i++)
	{
		Emission[0]->AddPoint(Min, 0.5, 0.5, 0);
		Emission[0]->AddPoint(Max, 0.5, 0.5, 0);
	}
}