/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include <vtkMetaImageReader.h>
#include <vtkVolumeProperty.h>
#include <vtkObjectFactory.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>

#include <vtkgl.h>

#include "VtkCudaVolumeMapper.h"

#include "Core.cuh"

#include "Geometry.h"

//vtkCxxRevisionMacro(vtkVolumeCudaMapper, "$Revision: 1.8 $");
vtkStandardNewMacro(vtkVolumeCudaMapper);



vtkVolumeCudaMapper::vtkVolumeCudaMapper()
{
	m_CudaVolumeInfo	= vtkCudaVolumeInfo::New();
	m_CudaRenderInfo	= vtkCudaRenderInfo::New();

	SetCudaDevice(0);

	

	glGenTextures(1, &TextureID);
}  

vtkVolumeCudaMapper::~vtkVolumeCudaMapper()
{
}

void vtkVolumeCudaMapper::SetInput(vtkImageData* input)
{
	this->Superclass::SetInput(input);


}

void vtkVolumeCudaMapper::Render(vtkRenderer* pRenderer, vtkVolume* pVolume)
{
	
	if (pVolume)
		UploadVolumeProperty(pVolume->GetProperty());

    int* pWindowSize = pRenderer->GetRenderWindow()->GetSize();

	/**/
	m_CudaRenderInfo->SetRenderer(pRenderer);
	
	m_CudaVolumeInfo->SetInputData(this->GetInput());
//	m_CudaVolumeInfo->SetVolume(pVolume);
	m_CudaVolumeInfo->Update();
//	return;
//	m_CudaRenderInfo->Bind();
	
	m_Host.Resize(CResolution2D(pWindowSize[0], pWindowSize[1]));

	m_CudaRenderInfo->GetRenderInfo()->m_NoIterations += 1;

	RenderEstimate(m_CudaVolumeInfo->GetVolumeInfo(), m_CudaRenderInfo->GetRenderInfo(), &m_CudaRenderInfo->m_Lighting, &m_CudaRenderInfo->m_FrameBuffer);

	cudaMemcpy(m_Host.GetPtr(), m_CudaRenderInfo->m_FrameBuffer.m_EstimateRgbaLdr.GetPtr(), m_Host.GetSize(), cudaMemcpyDeviceToHost);
	
    glBindTexture(GL_TEXTURE_2D, TextureID);
	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	/**/
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, pWindowSize[0], pWindowSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, m_Host.GetPtr());
 //   glBindTexture(GL_TEXTURE_2D, 0);
//	lPushAttrib(GL_ENABLE_BIT);
    
	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, pWindowSize[0], pWindowSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, m_Host.GetPtr());
	glBindTexture(GL_TEXTURE_2D, TextureID);
	glEnable(GL_TEXTURE_2D);
	glColor3f(1.0f, 1.0f, 1.0f);

    pRenderer->SetDisplayPoint(0,0,0.5);
    pRenderer->DisplayToWorld();
    double coordinatesA[4];
    pRenderer->GetWorldPoint(coordinatesA);

    pRenderer->SetDisplayPoint(pWindowSize[0],0,0.5);
    pRenderer->DisplayToWorld();
    double coordinatesB[4];
    pRenderer->GetWorldPoint(coordinatesB);

    pRenderer->SetDisplayPoint(pWindowSize[0],pWindowSize[1],0.5);
    pRenderer->DisplayToWorld();
    double coordinatesC[4];
    pRenderer->GetWorldPoint(coordinatesC);

    pRenderer->SetDisplayPoint(0,pWindowSize[1],0.5);
    pRenderer->DisplayToWorld();
    double coordinatesD[4];
    pRenderer->GetWorldPoint(coordinatesD);
	
	
    glPushAttrib(GL_BLEND);
    glEnable(GL_BLEND);
    glPushAttrib(GL_LIGHTING);
    glDisable(GL_LIGHTING);

    glBegin(GL_QUADS);
    glTexCoord2i(1,1);
    glVertex4dv(coordinatesA);
    glTexCoord2i(0,1);
    glVertex4dv(coordinatesB);
    glTexCoord2i(0,0);
    glVertex4dv(coordinatesC);
    glTexCoord2i(1,0);
    glVertex4dv(coordinatesD);
    glEnd();
    glPopAttrib();
    glPopAttrib();
    
//	Modified();

	return;
}

void vtkVolumeCudaMapper::PrintSelf(ostream& os, vtkIndent indent)
{
	vtkVolumeMapper::PrintSelf(os, indent);
}

int vtkVolumeCudaMapper::FillInputPortInformation(int port, vtkInformation* info)
{
	return 0;
}

void vtkVolumeCudaMapper::UploadVolumeProperty(vtkVolumeProperty* pVolumeProperty)
{
	if (pVolumeProperty == NULL)
	{
		vtkErrorMacro("Volume property cannot be null");
		return;
	}

	int N = 128;

	float* pOpacity		= new float[N];
	float* pDiffuse		= new float[N * 3];
	float* pSpecular	= new float[N * 3];
	float* pRoughness	= new float[N];
	float* pEmission	= new float[N * 3];
	
	double* pRange = GetDataSetInput()->GetScalarRange();

	pVolumeProperty->GetScalarOpacity()->GetTable(pRange[0], pRange[1], N, pOpacity);
	pVolumeProperty->GetRGBTransferFunction()->GetTable(pRange[0], pRange[1], N, pDiffuse);
	pVolumeProperty->GetRGBTransferFunction()->GetTable(pRange[0], pRange[1], N, pSpecular);

	BindTransferFunctions1D(pOpacity, pDiffuse, pSpecular, pRoughness, pEmission, N);

	delete pOpacity;
	delete pDiffuse;
	delete pSpecular;
	delete pRoughness;
	delete pEmission;
}