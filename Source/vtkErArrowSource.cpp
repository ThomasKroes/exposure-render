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

#include "vtkErArrowSource.h"

vtkStandardNewMacro(vtkErArrowSource);

vtkErArrowSource::vtkErArrowSource()
{
	this->SetShaftRadius(vtkErArrowSource::DefaultShaftRadius());
	this->SetShaftLength(vtkErArrowSource::DefaultShaftLength());
	this->SetShaftResolution(vtkErArrowSource::DefaultShaftResolution());
	this->SetTipRadius(vtkErArrowSource::DefaultTipRadius());
	this->SetTipLength(vtkErArrowSource::DefaultTipLength());
	this->SetTipResolution(vtkErArrowSource::DefaultTipResolution());
	
	this->SetNumberOfInputPorts(0);
}

int vtkErArrowSource::RequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **vtkNotUsed(inputVector),
  vtkInformationVector *outputVector)
{
  // get the info object
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  // get the ouptut
  vtkPolyData *output = vtkPolyData::SafeDownCast(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));

  int piece, numPieces, ghostLevel;
  vtkCylinderSource *cyl = vtkCylinderSource::New();
  vtkTransform *trans0 = vtkTransform::New();
  vtkTransformFilter *tf0 = vtkTransformFilter::New();
  vtkConeSource *cone = vtkConeSource::New();
  vtkTransform *trans1 = vtkTransform::New();
  vtkTransform *trans2 = vtkTransform::New();
  vtkTransformFilter *tf1 = vtkTransformFilter::New();
  vtkTransformFilter *tf2 = vtkTransformFilter::New();
  vtkAppendPolyData *append = vtkAppendPolyData::New();

  piece = output->GetUpdatePiece();
  numPieces = output->GetUpdateNumberOfPieces();
  ghostLevel = output->GetUpdateGhostLevel();

  cyl->SetResolution(this->ShaftResolution);
  cyl->SetRadius(this->ShaftRadius);
  cyl->SetHeight(1.0 - this->TipLength);
  cyl->SetCenter(0, (1.0-this->TipLength)*0.5, 0.0);
  cyl->CappingOn();

  trans0->RotateZ(-90.0);
  tf0->SetTransform(trans0);
  tf0->SetInput(cyl->GetOutput());

  cone->SetResolution(this->TipResolution);
  cone->SetHeight(this->TipLength);
  cone->SetRadius(this->TipRadius);

  trans1->Translate(1.0-this->TipLength*0.5, 0.0, 0.0);
  tf1->SetTransform(trans1);
  tf1->SetInput(cone->GetOutput());

  append->AddInput(tf0->GetPolyDataOutput());
  append->AddInput(tf1->GetPolyDataOutput());

 // used only when this->Invert is true.
 trans2->Translate(1, 0, 0);
 trans2->Scale(-1, 1, 1);
 tf2->SetTransform(trans2);
 tf2->SetInputConnection(append->GetOutputPort());

  if (output->GetUpdatePiece() == 0 && numPieces > 0)
    {
      append->Update();
      output->ShallowCopy(append->GetOutput());
    }
  output->SetUpdatePiece(piece);
  output->SetUpdateNumberOfPieces(numPieces);
  output->SetUpdateGhostLevel(ghostLevel);

  cone->Delete();
  trans0->Delete();
  tf0->Delete();
  cyl->Delete();
  trans1->Delete();
  tf1->Delete();
  append->Delete();
  tf2->Delete();
  trans2->Delete();

  return 1;
}

double vtkErArrowSource::DefaultShaftRadius(void)
{
	return 10;
}

double vtkErArrowSource::DefaultShaftLength(void)
{
	return 90;
}

int vtkErArrowSource::DefaultShaftResolution(void)
{
	return 16;
}

double vtkErArrowSource::DefaultTipRadius(void)
{
	return 15;
}

double vtkErArrowSource::DefaultTipLength(void)
{
	return 20;
}

int vtkErArrowSource::DefaultTipResolution(void)
{
	return 16;
}