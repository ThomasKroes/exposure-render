/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "VtkErLight.h"

#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkMatrix4x4.h>

class vtkCamera;

class VTK_ER_CORE_EXPORT vtkErAreaLight : public vtkErLight
{
public:
	vtkTypeMacro(vtkErAreaLight, vtkErLight);
	static vtkErAreaLight* New();

	// Light types:
	// 0 - Spherical positioning (focal point + elevation + azimuth + distance)
	// 1 - Target positioning (focal point + origin + up vector)
	// 2 - Camera light (focal point + elevation + azimuth + distance)
	vtkGetMacro(Type, int);
	vtkSetMacro(Type, int);

	// Shape types:
	// 0 - Plane
	// 1 - Disk
	// 2 - Ring
	// 3 - Box
	// 4 - Sphere
	// 5 - Cylinder
	vtkGetMacro(ShapeType, int);
	vtkSetMacro(ShapeType, int);

	vtkGetMacro(OneSided, bool);
	vtkSetMacro(OneSided, bool);

	vtkSetVector3Macro(Up, double);
	vtkGetVector3Macro(Up, double);

	vtkSetVector3Macro(Size, double);
	vtkGetVector3Macro(Size, double);

	vtkMatrix4x4* GetTransformMatrix();

	vtkGetMacro(Elevation, double);
	vtkSetMacro(Elevation, double);

	vtkGetMacro(Azimuth, double);
	vtkSetMacro(Azimuth, double);

	vtkGetMacro(Distance, double);
	vtkSetMacro(Distance, double);

	vtkGetMacro(Offset, double);
	vtkSetMacro(Offset, double);

	vtkGetMacro(InnerRadius, double);
	vtkSetMacro(InnerRadius, double);

	vtkGetMacro(OuterRadius, double);
	vtkSetMacro(OuterRadius, double);

	vtkGetMacro(Camera, vtkCamera*);
	vtkSetMacro(Camera, vtkCamera*);

	double GetArea() const;

	static int		DefaultType();
	static int		DefaultShapeType();
	static bool		DefaultOneSided();
	static double	DefaultElevation();
	static double	DefaultAzimuth();
	static double	DefaultDistance();
	static double	DefaultOffset();
	static double	DefaultInnerRadius();
	static double	DefaultOuterRadius();

protected:
	vtkErAreaLight(void);
	virtual ~vtkErAreaLight(void);

	int								Type;
	int								ShapeType;
	bool							OneSided;
	double							Up[3];
	double							Size[3];
	double							Elevation;
	double							Azimuth;
	double							Distance;
	double							Offset;
	double							InnerRadius;
	double							OuterRadius;
	vtkCamera*						Camera;
	vtkSmartPointer<vtkTransform>	Transform;
};