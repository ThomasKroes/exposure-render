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

#include "VtkErVolumeMapper.h"
#include "VtkErVolumeProperty.h"

#include "Core.cuh"

#include "Geometry.h"

//vtkCxxRevisionMacro(vtkErVolumeMapper, "$Revision: 1.8 $");
vtkStandardNewMacro(vtkErVolumeMapper);

vtkErVolumeMapper::vtkErVolumeMapper()
{
	m_CudaVolumeInfo	= vtkErVolumeInfo::New();
	m_CudaRenderInfo	= vtkErRenderInfo::New();

	SetCudaDevice(0);

	

	glGenTextures(1, &TextureID);
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
	/**/
	if (pVolume)
		UploadVolumeProperty(pVolume->GetProperty());

    int* pWindowSize = pRenderer->GetRenderWindow()->GetSize();

	
	m_CudaRenderInfo->SetRenderer(pRenderer);
	m_CudaVolumeInfo->SetVolume(pVolume);
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

//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, pWindowSize[0], pWindowSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, m_Host.GetPtr());
 //   glBindTexture(GL_TEXTURE_2D, 0);
//	lPushAttrib(GL_ENABLE_BIT);
    
	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, pWindowSize[0], pWindowSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, m_Host.GetPtr());
	glBindTexture(GL_TEXTURE_2D, TextureID);
	glEnable(GL_TEXTURE_2D);
	glColor3f(1.0f, 1.0f, 1.0f);

	double d = 0.5;

    pRenderer->SetDisplayPoint(0,0,d);
    pRenderer->DisplayToWorld();
    double coordinatesA[4];
    pRenderer->GetWorldPoint(coordinatesA);

    pRenderer->SetDisplayPoint(pWindowSize[0],0,d);
    pRenderer->DisplayToWorld();
    double coordinatesB[4];
    pRenderer->GetWorldPoint(coordinatesB);

    pRenderer->SetDisplayPoint(pWindowSize[0],pWindowSize[1],d);
    pRenderer->DisplayToWorld();
    double coordinatesC[4];
    pRenderer->GetWorldPoint(coordinatesC);

    pRenderer->SetDisplayPoint(0,pWindowSize[1],d);
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

	return;
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
		vtkErrorMacro("Incompatible volume property, use vtkErVolumeProperty!");
		
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