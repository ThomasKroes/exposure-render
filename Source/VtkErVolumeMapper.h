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

// #include "View.cuh"

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkConeSource.h>
#include <vtkRenderWindow.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkImageImport.h>
#include <vtkImageActor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkVolumeMapper.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkImageData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPointData.h>
#include <vtkVolume.h>

// http://www.na-mic.org/svn/Slicer3/branches/cuda/Modules/VolumeRenderingCuda/

#include "VtkErVolumeInfo.h"
#include "VtkErRenderInfo.h"

class vtkVolumeProperty;

class EXPOSURE_RENDER_DLL vtkVolumeCudaMapper : public vtkVolumeMapper
{
public:
//  vtkTypeRevisionMacro(vtkVolumeCudaMapper,vtkVolumeMapper);
	vtkTypeMacro(vtkVolumeCudaMapper,vtkVolumeMapper);
    static vtkVolumeCudaMapper *New();
vtkVolumeCudaMapper operator=(const vtkVolumeCudaMapper&);
    vtkVolumeCudaMapper(const vtkVolumeCudaMapper&);
    virtual void SetInput( vtkImageData * );
//	virtual void SetInput( vtkDataSet * );
    virtual void Render(vtkRenderer *, vtkVolume *);
	virtual int FillInputPortInformation(int, vtkInformation*);

   vtkImageData* GetOutput() { return NULL;  }

	void PrintSelf(ostream& os, vtkIndent indent);

    void UpdateOutputResolution(unsigned int width, unsigned int height, bool TypeChanged = false);

protected:
	vtkSmartPointer<vtkErVolumeInfo>	m_CudaVolumeInfo;
	vtkSmartPointer<vtkErRenderInfo>	m_CudaRenderInfo;

	vtkVolumeCudaMapper();
    virtual ~vtkVolumeCudaMapper();

	void UploadVolumeProperty(vtkVolumeProperty* pVolumeProperty);

	CHostBuffer2D<ColorRGBAuc>	m_Host;
//	CCudaView	m_CudaView;
	unsigned int TextureID;
};
/**/