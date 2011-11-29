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
#include <vtkAppendPolyData.h>
#include <vtkConeSource.h>
#include <vtkCylinderSource.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>

class VTK_ER_CORE_EXPORT vtkErArrowSource : public vtkPolyDataAlgorithm
{
public:
	vtkTypeMacro(vtkErArrowSource,vtkPolyDataAlgorithm);
	static vtkErArrowSource *New();
    
	vtkGetMacro(ShaftRadius, double);
	vtkSetMacro(ShaftRadius, double);
  
	vtkGetMacro(ShaftLength, double);
	vtkSetMacro(ShaftLength, double);

	vtkGetMacro(ShaftResolution, int);
	vtkSetClampMacro(ShaftResolution, int, 0, 128);
  
	vtkGetMacro(TipRadius, double);
	vtkSetMacro(TipRadius, double);
  
	vtkGetMacro(TipLength, double);
	vtkSetMacro(TipLength, double);

	vtkGetMacro(TipResolution, int);
	vtkSetClampMacro(TipResolution, int, 0, 128);

	static double DefaultShaftRadius(void);
	static double DefaultShaftLength(void);
	static int DefaultShaftResolution(void);
	static double DefaultTipRadius(void);
	static double DefaultTipLength(void);
	static int DefaultTipResolution(void);

protected:
	vtkErArrowSource();
	~vtkErArrowSource() {};

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

	double	ShaftRadius;
	double	ShaftLength;
	int		ShaftResolution;
	double	TipRadius;
	double	TipLength;
	int		TipResolution;

private:
	vtkErArrowSource(const vtkErArrowSource&);
	void operator=(const vtkErArrowSource&);
};
