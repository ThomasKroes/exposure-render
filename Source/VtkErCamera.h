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

#include "Geometry.h"

class vtkRenderer;

class EXPOSURE_RENDER_DLL vtkErCamera : public vtkOpenGLCamera
{
	vtkTypeRevisionMacro(vtkErCamera, vtkCamera);
	static vtkErCamera *New();

	vtkGetMacro(Renderer, vtkRenderer*);
	vtkSetMacro(Renderer, vtkRenderer*);

	vtkGetMacro(FocalDistance, double);
	vtkSetMacro(FocalDistance, double);

	vtkGetMacro(NoApertureBlades, int);
	vtkSetMacro(NoApertureBlades, int);

	vtkGetMacro(BladesAngle, double);
	vtkSetMacro(BladesAngle, double);

	vtkGetMacro(ApertureBias, double);
	vtkSetMacro(ApertureBias, double);

	vtkGetMacro(Exposure, double);
	vtkSetMacro(Exposure, double);

	void Default(void);

	void SetViewFront(void);
	void SetViewBack(void);
	void SetViewLeft(void);
	void SetViewRight(void);
	void SetViewTop(void);
	void SetViewBottom(void);

	static double DefaultFocalDisk(void);
	static double DefaultFocalDistance(void);
	static double DefaultNoApertureBlades(void);
	static double DefaultBladesAngle(void);
	static double DefaultApertureBias(void);
	static double DefaultExposure(void);

protected:
	vtkErCamera();
	virtual ~vtkErCamera();

	vtkRenderer*	Renderer;
	double			FocalDistance;
	int				NoApertureBlades;
	int				BladesAngle;
	double			ApertureBias;
	double			Exposure;
};