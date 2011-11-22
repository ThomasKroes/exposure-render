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

#include <vtkVolumeProperty.h>
#include <vtkSmartPointer.h>
#include <vtkPiecewiseFunction.h>

class VTK_ER_CORE_EXPORT vtkErVolumeProperty : public vtkVolumeProperty
{
public:
	vtkTypeMacro(vtkErVolumeProperty, vtkVolumeProperty);
	static vtkErVolumeProperty *New();

	void SetOpacity(vtkPiecewiseFunction* pPiecewiseFunction);
	vtkPiecewiseFunction* GetOpacity(void);

	void SetDiffuse(int Index, vtkPiecewiseFunction* pPiecewiseFunction);
	vtkPiecewiseFunction* GetDiffuse(int Index);

	void SetSpecular(int Index, vtkPiecewiseFunction* pPiecewiseFunction);
	vtkPiecewiseFunction* GetSpecular(int Index);

	void SetGlossiness(vtkPiecewiseFunction* pPiecewiseFunction);
	vtkPiecewiseFunction* GetGlossiness(void);

	void SetEmission(int Index, vtkPiecewiseFunction* pPiecewiseFunction);
	vtkPiecewiseFunction* GetEmission(int Index);

	vtkGetMacro(Dirty, bool);
	vtkSetMacro(Dirty, bool);
	
	vtkGetMacro(DensityScale, double);
	void SetDensityScale(double DensityScale);
	
	vtkGetMacro(StepSizeFactorPrimary, double);
	void SetStepSizeFactorPrimary(double StepSizeFactorPrimary);
	
	vtkGetMacro(StepSizeFactorSecondary, double);
	void SetStepSizeFactorSecondary(double StepSizeFactorSecondary);

	vtkGetMacro(GradientDeltaFactor, double);
	void SetGradientDeltaFactor(double GradientDeltaFactor);
	
	vtkGetMacro(GradientFactor, double);
	void SetGradientFactor(double GradientFactor);

	vtkGetMacro(ShadingType, int);
	void SetShadingType(int ShadingType);

	void Default(double Min, double Max);

	static double DefaultDensityScale(void);
	static double DefaultStepSizeFactorPrimary(void);
	static double DefaultStepSizeFactorSecondary(void);
	static double DefaultGradientDeltaFactor(void);
	static double DefaultGradientFactor(void);
	static int DefaultShadingType(void);

protected:
	vtkErVolumeProperty();
	virtual ~vtkErVolumeProperty();

	vtkSmartPointer<vtkPiecewiseFunction>	Opacity;
	vtkSmartPointer<vtkPiecewiseFunction>	Diffuse[3];
	vtkSmartPointer<vtkPiecewiseFunction>	Specular[3];
	vtkSmartPointer<vtkPiecewiseFunction>	Glossiness;
	vtkSmartPointer<vtkPiecewiseFunction>	Emission[3];
	bool									Dirty;
	double									DensityScale;
	double									StepSizeFactorPrimary;
	double									StepSizeFactorSecondary;
	double									GradientDeltaFactor;
	double									GradientFactor;
	int										ShadingType;
};
