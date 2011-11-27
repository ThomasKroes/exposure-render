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

#include "VtkErVolumeInfo.h"
#include "VtkErRenderInfo.h"
#include "vtkErSlicePlaneWidget.h"

#include <vtkVolumeMapper.h>
#include <vtkSmartPointer.h>
#include <vtkVolumeProperty.h>
#include <vtkCommand.h>

class vtkErVolumeMapper;

class vtkErResetCommand : public vtkCommand
{
public:
	static vtkErResetCommand* New() { return new vtkErResetCommand; };

	virtual void Execute(vtkObject*, unsigned long, void *);
	void SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper);

protected:
	vtkErResetCommand() { this->VolumeMapper = NULL; };
	~vtkErResetCommand() {};

	vtkErVolumeMapper*	VolumeMapper;
};

class vtkErUpdateSlicingCommand : public vtkCommand
{
public:
	static vtkErUpdateSlicingCommand* New() { return new vtkErUpdateSlicingCommand; };

	virtual void Execute(vtkObject*, unsigned long, void *);
	void SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper);

protected:
	vtkErUpdateSlicingCommand() { this->VolumeMapper = NULL; };
	~vtkErUpdateSlicingCommand() {};

	vtkErVolumeMapper*	VolumeMapper;
};

class VTK_ER_CORE_EXPORT vtkErVolumeMapper : public vtkVolumeMapper
{
public:
	vtkTypeMacro(vtkErVolumeMapper, vtkVolumeMapper);
    static vtkErVolumeMapper* New();

	vtkErVolumeMapper operator=(const vtkErVolumeMapper&);
    vtkErVolumeMapper(const vtkErVolumeMapper&);
    virtual void SetInput( vtkImageData * );
//	virtual void SetInput( vtkDataSet * );
    virtual void Render(vtkRenderer *, vtkVolume *);
	virtual int FillInputPortInformation(int, vtkInformation*);

   vtkImageData* GetOutput() { return NULL;  }

	void PrintSelf(ostream& os, vtkIndent indent);

 //   void UpdateOutputResolution(unsigned int width, unsigned int height, bool TypeChanged = false);

	vtkGetMacro(UseCustomRenderSize, bool);
	vtkSetMacro(UseCustomRenderSize, bool);

	vtkGetVector2Macro(CustomRenderSize, int);
	vtkSetVector2Macro(CustomRenderSize, int);

//	void DoIt(void* pData);
	
	vtkSmartPointer<vtkErVolumeInfo>	m_CudaVolumeInfo;
	vtkSmartPointer<vtkErRenderInfo>	m_CudaRenderInfo;

	vtkErVolumeMapper();
    virtual ~vtkErVolumeMapper();

	void UploadVolumeProperty(vtkVolumeProperty* pVolumeProperty);

	CHostBuffer2D<ColorRGBAuc>	m_Host;
//	CCudaView	m_CudaView;
	unsigned int TextureID;

	bool	UseCustomRenderSize;
	int		CustomRenderSize[2];

	vtkGetMacro(SliceWidget, vtkErSliceBoxWidget*);
//	vtkSetMacro(SliceWidget, vtkErSlicePlaneWidget*);
	void SetSliceWidget(vtkErSliceBoxWidget* pSliceWidget);

	void Reset();

protected:
	vtkErSliceBoxWidget*						SliceWidget;
	vtkSmartPointer<vtkErResetCommand>			ResetCallBack;
	vtkSmartPointer<vtkErUpdateSlicingCommand>	UpdateSlicingCommand;
};

// http://www.na-mic.org/svn/Slicer3/branches/cuda/Modules/VolumeRenderingCuda/