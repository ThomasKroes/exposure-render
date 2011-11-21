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

#include "Dll.h"

class EXPOSURE_RENDER_DLL vtkErArrowSource : public vtkPolyDataAlgorithm
{
public:
  // Description
  // Construct cone with angle of 45 degrees.
  static vtkErArrowSource *New();

  vtkTypeMacro(vtkErArrowSource,vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);
    
  // Description:
  // Set the length, and radius of the tip.  They default to 0.35 and 0.1
  vtkSetClampMacro(TipLength,double,0.0,1.0);
  vtkGetMacro(TipLength,double);
  vtkSetClampMacro(TipRadius,double,0.0,10.0);
  vtkGetMacro(TipRadius,double);
  
  // Description:
  // Set the resolution of the tip.  The tip behaves the same as a cone.
  // Resoultion 1 gives a single triangle, 2 gives two crossed triangles.
  vtkSetClampMacro(TipResolution,int,1,128);
  vtkGetMacro(TipResolution,int);

  // Description:
  // Set the radius of the shaft.  Defaults to 0.03.
  vtkSetClampMacro(ShaftRadius,double,0.0,5.0);
  vtkGetMacro(ShaftRadius,double);

  // Description:
  // Set the resolution of the shaft.  2 gives a rectangle.
  // I would like to extend the cone to produce a line,
  // but this is not an option now.
  vtkSetClampMacro(ShaftResolution,int,0,128);
  vtkGetMacro(ShaftResolution,int);

  // Description:
  // Inverts the arrow direction. When set to true, base is at (1, 0, 0) while the
  // tip is at (0, 0, 0). The default is false, i.e. base at (0, 0, 0) and the tip
  // at (1, 0, 0).
  vtkBooleanMacro(Invert, bool);
  vtkSetMacro(Invert, bool);
  vtkGetMacro(Invert, bool);

  vtkSetMacro(ShaftLength, double);
  vtkGetMacro(ShaftLength, double);

  vtkSetMacro(Offset, double);
  vtkGetMacro(Offset, double);

protected:
	vtkErArrowSource();
	~vtkErArrowSource() {};

	//BTX
	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

	double	Offset;
	double	TipLength;
	double	TipRadius;
	int		TipResolution;
	double	ShaftLength;
	double	ShaftRadius;
	int		ShaftResolution;
	bool	Invert;
	//ETX

private:
  vtkErArrowSource(const vtkErArrowSource&); // Not implemented.
  void operator=(const vtkErArrowSource&); // Not implemented.
};