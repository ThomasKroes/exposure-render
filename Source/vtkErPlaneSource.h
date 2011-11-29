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

#include "vtkErCoreDll.h"

#include <vtkPolyDataAlgorithm.h>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkMath.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkTransform.h>

class VTK_ER_CORE_EXPORT vtkErPlaneSource : public vtkPolyDataAlgorithm 
{
public:
	vtkTypeMacro(vtkErPlaneSource,vtkPolyDataAlgorithm);
	static vtkErPlaneSource *New();

	vtkSetMacro(XResolution, int);
	vtkGetMacro(XResolution, int);

	vtkSetMacro(YResolution, int);
	vtkGetMacro(YResolution, int);

	void SetResolution(const int xR, const int yR);
	void GetResolution(int& xR,int& yR) {
	xR=this->XResolution; yR=this->YResolution;};

	vtkSetVector3Macro(Position, double);
	vtkGetVector3Macro(Position, double);

	vtkSetVector3Macro(Normal, double);
	vtkGetVector3Macro(Normal, double);

	vtkGetVector3Macro(Up, double);
	void SetUp(double X, double Y, double Z);

	vtkGetVector2Macro(Size, double);
	void SetSize(double Width, double Height);

protected:
	vtkErPlaneSource();
	~vtkErPlaneSource() {};

	int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*);

	int XResolution;
	int YResolution;

	double Position[3];
	double Normal[3];
	double Up[3];
	double Size[3];

private:
	vtkErPlaneSource(const vtkErPlaneSource&);
	void operator=(const vtkErPlaneSource&);
};
