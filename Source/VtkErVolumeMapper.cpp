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
#include <vtkMultiThreader.h>

#include <vtkgl.h>

#include "VtkErVolumeMapper.h"
#include "VtkErVolumeProperty.h"

#include "Core.cuh"

#include "Geometry.h"

#include "vtkImageActor.h"

//vtkCxxRevisionMacro(vtkErVolumeMapper, "$Revision: 1.8 $");
vtkStandardNewMacro(vtkErVolumeMapper);

/*
// this function runs in an alternate thread to asyncronously generate matrices
static void *vtkTrackerSimulatorRecordThread(vtkMultiThreader::ThreadInfo* pData)
{
	vtkErVolumeMapper* pVolumeMapper = (vtkErVolumeMapper*)(pData->UserData);

	if (!pVolumeMapper)
		return NULL;

//	RenderEstimate(pVolumeMapper->m_CudaVolumeInfo->GetVolumeInfo(), pVolumeMapper->m_CudaRenderInfo->GetRenderInfo(), &pVolumeMapper->m_CudaRenderInfo->m_Lighting, &pVolumeMapper->m_CudaRenderInfo->m_FrameBuffer);

	return NULL;
}
*/

vtkErVolumeMapper::vtkErVolumeMapper()
{
	m_CudaVolumeInfo	= vtkErVolumeInfo::New();
	m_CudaRenderInfo	= vtkErRenderInfo::New();

	SetCudaDevice(0);

//	SetUseCustomRenderSize(false);
//	SetCustomRenderSize(34, 34);

	glGenTextures(1, &TextureID);

//	MultiThreader = vtkMultiThreader::New();

//	MultiThreader->SpawnThread((vtkThreadFunctionType)vtkTrackerSimulatorRecordThread, this);

}  

vtkErVolumeMapper::~vtkErVolumeMapper()
{
}

void vtkErVolumeMapper::SetInput(vtkImageData* input)
{
	this->Superclass::SetInput(input);


}

void vtkErVolumeMapper::Render(vtkRenderer* pRenderer, vtkVolume* pVolume)
{
	if (!pVolume)
		return;
	
	UploadVolumeProperty(pVolume->GetProperty());

	int RenderSize[2];

	int* pWindowSize = pRenderer->GetRenderWindow()->GetSize();

	RenderSize[0] = pWindowSize[0];
	RenderSize[1] = pWindowSize[1];
	
	m_CudaRenderInfo->SetRenderer(pRenderer);
	m_CudaRenderInfo->Update();
	m_CudaVolumeInfo->SetVolume(pVolume);
	m_CudaVolumeInfo->SetInputData(this->GetInput());
	m_CudaVolumeInfo->SetVolume(pVolume);
	m_CudaVolumeInfo->Update();
	
	m_Host.Resize(CResolution2D(RenderSize[0], RenderSize[1]));

	m_CudaRenderInfo->GetRenderInfo()->m_NoIterations += 1;

	RenderEstimate(m_CudaVolumeInfo->GetVolumeInfo(), m_CudaRenderInfo->GetRenderInfo(), &m_CudaRenderInfo->m_Lighting, &m_CudaRenderInfo->m_FrameBuffer);

	cudaMemcpy(m_Host.GetPtr(), m_CudaRenderInfo->m_FrameBuffer.m_EstimateRgbaLdr.GetPtr(), m_Host.GetSize(), cudaMemcpyDeviceToHost);
	
	glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, TextureID);
    
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, RenderSize[0], RenderSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, m_Host.GetPtr());
	glBindTexture(GL_TEXTURE_2D, TextureID);

//	glColor4f(1.0f, 1.0f, 1.0f, 0.5f);
	
	double d = 0.5;
	
    pRenderer->SetDisplayPoint(0,0,d);
    pRenderer->DisplayToWorld();
    double coordinatesA[4];
    pRenderer->GetWorldPoint(coordinatesA);

    pRenderer->SetDisplayPoint(RenderSize[0],0,d);
    pRenderer->DisplayToWorld();
    double coordinatesB[4];
    pRenderer->GetWorldPoint(coordinatesB);

    pRenderer->SetDisplayPoint(RenderSize[0], RenderSize[1],d);
    pRenderer->DisplayToWorld();
    double coordinatesC[4];
    pRenderer->GetWorldPoint(coordinatesC);

    pRenderer->SetDisplayPoint(0,RenderSize[1],d);
    pRenderer->DisplayToWorld();
    double coordinatesD[4];
    pRenderer->GetWorldPoint(coordinatesD);
	
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
}

void vtkErVolumeMapper::PrintSelf(ostream& os, vtkIndent indent)
{
	vtkVolumeMapper::PrintSelf(os, indent);
}

int vtkErVolumeMapper::FillInputPortInformation(int port, vtkInformation* info)
{
	return 0;
}

void vtkErVolumeMapper::UploadVolumeProperty(vtkVolumeProperty* pVolumeProperty)
{
	if (pVolumeProperty == NULL)
	{
		vtkErrorMacro("Volume property is NULL!");
		return;
	}

	vtkErVolumeProperty* pErVolumeProperty = dynamic_cast<vtkErVolumeProperty*>(pVolumeProperty);

	double* pRange = GetDataSetInput()->GetScalarRange();

	int N = 128;

	float	Opacity[128];
	float	Diffuse[3][128];
	float	Specular[3][128];
	float	Glossiness[128];
	float	Emission[3][128];

	if (pErVolumeProperty == NULL)
	{
		vtkErrorMacro("Incompatible volume property (reverting to default property), use vtkErVolumeProperty!");
		
		vtkSmartPointer<vtkErVolumeProperty> ErVolumeProperty = vtkErVolumeProperty::New();

		ErVolumeProperty->Default(pRange[0], pRange[1]);

		ErVolumeProperty->GetOpacity()->GetTable(pRange[0], pRange[1], N, Opacity);
		ErVolumeProperty->GetDiffuse(0)->GetTable(pRange[0], pRange[1], N, Diffuse[0]);
		ErVolumeProperty->GetDiffuse(1)->GetTable(pRange[0], pRange[1], N, Diffuse[1]);
		ErVolumeProperty->GetDiffuse(2)->GetTable(pRange[0], pRange[1], N, Diffuse[2]);
		ErVolumeProperty->GetSpecular(0)->GetTable(pRange[0], pRange[1], N, Specular[0]);
		ErVolumeProperty->GetSpecular(1)->GetTable(pRange[0], pRange[1], N, Specular[1]);
		ErVolumeProperty->GetSpecular(2)->GetTable(pRange[0], pRange[1], N, Specular[2]);
		ErVolumeProperty->GetGlossiness()->GetTable(pRange[0], pRange[1], N, Glossiness);
		ErVolumeProperty->GetEmission(0)->GetTable(pRange[0], pRange[1], N, Emission[0]);
		ErVolumeProperty->GetEmission(1)->GetTable(pRange[0], pRange[1], N, Emission[1]);
		ErVolumeProperty->GetEmission(2)->GetTable(pRange[0], pRange[1], N, Emission[2]);
	}
	else
	{
		if (pErVolumeProperty->GetDirty())
		{
			m_CudaRenderInfo->Reset();
			pErVolumeProperty->SetDirty(false);
		}

		pErVolumeProperty->GetOpacity()->GetTable(pRange[0], pRange[1], N, Opacity);
		pErVolumeProperty->GetDiffuse(0)->GetTable(pRange[0], pRange[1], N, Diffuse[0]);
		pErVolumeProperty->GetDiffuse(1)->GetTable(pRange[0], pRange[1], N, Diffuse[1]);
		pErVolumeProperty->GetDiffuse(2)->GetTable(pRange[0], pRange[1], N, Diffuse[2]);
		pErVolumeProperty->GetSpecular(0)->GetTable(pRange[0], pRange[1], N, Specular[0]);
		pErVolumeProperty->GetSpecular(1)->GetTable(pRange[0], pRange[1], N, Specular[1]);
		pErVolumeProperty->GetSpecular(2)->GetTable(pRange[0], pRange[1], N, Specular[2]);
		pErVolumeProperty->GetGlossiness()->GetTable(pRange[0], pRange[1], N, Glossiness);
		pErVolumeProperty->GetEmission(0)->GetTable(pRange[0], pRange[1], N, Emission[0]);
		pErVolumeProperty->GetEmission(1)->GetTable(pRange[0], pRange[1], N, Emission[1]);
		pErVolumeProperty->GetEmission(2)->GetTable(pRange[0], pRange[1], N, Emission[2]);
	}

	BindTransferFunctions1D(Opacity, Diffuse, Specular, Glossiness, Emission, N);
}
