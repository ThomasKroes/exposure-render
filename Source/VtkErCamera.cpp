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

#include "vtkErCamera.h"

vtkStandardNewMacro(vtkErCamera);

vtkErCamera::vtkErCamera(void)
{
	Default();
}

vtkErCamera::~vtkErCamera(void)
{
}

void vtkErCamera::Default(void)
{
	SetFocalDisk(vtkErCamera::DefaultFocalDisk());
	SetFocalDistance(vtkErCamera::DefaultFocalDistance());
	SetNoApertureBlades(vtkErCamera::DefaultNoApertureBlades());
	SetApertureBias(vtkErCamera::DefaultApertureBias());
	SetExposure(vtkErCamera::DefaultExposure());
}

void vtkErCamera::SetViewFront(void)
{
	if (!Renderer)
	{
		vtkErrorMacro("Unable to set front view, VTK Renderer is NULL!");
		return;
	}

	double* pBounds = Renderer->ComputeVisiblePropBounds();

	vtkBoundingBox AABB(pBounds);

	double Center[3];

	AABB.GetCenter(Center);

	SetFocalPoint(Center[0], Center[1], Center[2]);

	const double Distance = (0.5 * AABB.GetMaxLength()) / tan(vtkMath::RadiansFromDegrees(0.5 * GetViewAngle()));
	
	SetPosition(Center[0], Center[1], Center[2] + Distance);
	SetViewUp(0, 1, 0);

	this->Modified();
}

void vtkErCamera::SetViewBack(void)
{
	if (!Renderer)
	{
		vtkErrorMacro("Unable to set front view, VTK Renderer is NULL!");
		return;
	}

	double* pBounds = Renderer->ComputeVisiblePropBounds();

	vtkBoundingBox AABB(pBounds);

	double Center[3];

	AABB.GetCenter(Center);

	SetFocalPoint(Center[0], Center[1], Center[2]);

	const double Distance = (0.5 * AABB.GetMaxLength()) / tan(vtkMath::RadiansFromDegrees(0.5 * GetViewAngle()));
	
	SetPosition(Center[0], Center[1], Center[2] - Distance);
	SetViewUp(0, 1, 0);

	this->Modified();
}

void vtkErCamera::SetViewLeft(void)
{
	if (!Renderer)
	{
		vtkErrorMacro("Unable to set front view, VTK Renderer is NULL!");
		return;
	}

	double* pBounds = Renderer->ComputeVisiblePropBounds();

	vtkBoundingBox AABB(pBounds);

	double Center[3];

	AABB.GetCenter(Center);

	SetFocalPoint(Center[0], Center[1], Center[2]);

	const double Distance = (0.5 * AABB.GetMaxLength()) / tan(vtkMath::RadiansFromDegrees(0.5 * GetViewAngle()));
	
	SetPosition(Center[0] - Distance, Center[1], Center[2]);
	SetViewUp(0, 1, 0);

	this->Modified();
}

void vtkErCamera::SetViewRight(void)
{
	if (!Renderer)
	{
		vtkErrorMacro("Unable to set front view, VTK Renderer is NULL!");
		return;
	}

	double* pBounds = Renderer->ComputeVisiblePropBounds();

	vtkBoundingBox AABB(pBounds);

	double Center[3];

	AABB.GetCenter(Center);

	SetFocalPoint(Center[0], Center[1], Center[2]);

	const double Distance = (0.5 * AABB.GetMaxLength()) / tan(vtkMath::RadiansFromDegrees(0.5 * GetViewAngle()));
	
	SetPosition(Center[0] + Distance, Center[1], Center[2]);
	SetViewUp(0, 1, 0);

	this->Modified();
}

void vtkErCamera::SetViewTop(void)
{
	if (!Renderer)
	{
		vtkErrorMacro("Unable to set front view, VTK Renderer is NULL!");
		return;
	}

	double* pBounds = Renderer->ComputeVisiblePropBounds();

	vtkBoundingBox AABB(pBounds);

	double Center[3];

	AABB.GetCenter(Center);

	SetFocalPoint(Center[0], Center[1], Center[2]);

	const double Distance = (0.5 * AABB.GetMaxLength()) / tan(vtkMath::RadiansFromDegrees(0.5 * GetViewAngle()));
	
	SetPosition(Center[0], Center[1] + Distance, Center[2]);
	SetViewUp(0, 0, -1);

	this->Modified();
}

void vtkErCamera::SetViewBottom(void)
{
	if (!Renderer)
	{
		vtkErrorMacro("Unable to set front view, VTK Renderer is NULL!");
		return;
	}

	double* pBounds = Renderer->ComputeVisiblePropBounds();

	vtkBoundingBox AABB(pBounds);

	double Center[3] = { 0.5 * (AABB.GetMaxPoint()[0] - AABB.GetMinPoint()[0]), 0.5 * (AABB.GetMaxPoint()[1] - AABB.GetMinPoint()[1]), 0.5 * (AABB.GetMaxPoint()[2] - AABB.GetMinPoint()[2]) };

	AABB.GetCenter(Center);

	SetFocalPoint(Center[0], Center[1], Center[2]);

	const double Distance = (0.5 * AABB.GetMaxLength()) / tan(vtkMath::RadiansFromDegrees(0.5 * GetViewAngle()));
	
	SetPosition(Center[0], Center[1] - Distance, Center[2]);
	SetViewUp(0, 0, -1);

	this->Modified();
}

double vtkErCamera::DefaultFocalDisk(void)
{
	return 0.001;
}

double vtkErCamera::DefaultFocalDistance(void)
{
	return 1.0;
}

double vtkErCamera::DefaultNoApertureBlades(void)
{
	return 6;
}

double vtkErCamera::DefaultBladesAngle(void)
{
	return 0.0;
}

double vtkErCamera::DefaultApertureBias(void)
{
	return 0.5;
}

double vtkErCamera::DefaultExposure(void)
{
	return 50.0;
}