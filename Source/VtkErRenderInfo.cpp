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

#include "vtkErRenderInfo.h"
#include "vtkErAreaLight.h"
#include "vtkErBackgroundLight.h"
#include "vtkErCamera.h"

#include <vtkObjectFactory.h>
#include <vtkTransform.h>
#include <vtkPerspectiveTransform.h>
#include <vtkCommand.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>

#include "Core.cuh"

class vtkErCameraCallbackCommand : public vtkCommand
{
public:
  static vtkErCameraCallbackCommand *New()
    { return new vtkErCameraCallbackCommand; };
  vtkCamera *Self;

  vtkErRenderInfo*	m_pCudaRenderInfo;

  void Execute(vtkObject *, unsigned long, void *)
    {
		m_pCudaRenderInfo->Reset();

    }
protected:
  vtkErCameraCallbackCommand() { this->Self = NULL; };
  ~vtkErCameraCallbackCommand() {};
};

class vtkRendererObserver : public vtkCommand
{
public:
  static vtkRendererObserver* New (void) { return new vtkRendererObserver; }

  void Execute (vtkObject* aCaller, unsigned long aEID, void* aCallData)
    {
      vtkRenderer* renderer = static_cast<vtkRenderer*>(aCaller);
//	  renderer->Up();
//	  renderer->();
//	  Log("asd");
    }

protected:
  vtkRendererObserver (void){}

  char          Buffer[256];

private:
  vtkRendererObserver (const vtkRendererObserver&);
  void operator= (const vtkRendererObserver&);
};

vtkCxxRevisionMacro(vtkErRenderInfo, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkErRenderInfo);

vtkErRenderInfo::vtkErRenderInfo()
{
	Renderer = NULL;
}

vtkErRenderInfo::~vtkErRenderInfo()
{
	Renderer = NULL;
}

void vtkErRenderInfo::SetRenderer(vtkRenderer* pRenderer)
{
	if (Renderer == NULL)
	{
		vtkRendererObserver* renObserver = vtkRendererObserver::New();

		pRenderer->AddObserver(vtkCommand::EndEvent, renObserver);

		vtkErCameraCallbackCommand* pCallBack = vtkErCameraCallbackCommand::New();
		pCallBack->m_pCudaRenderInfo = this;

		pRenderer->GetActiveCamera()->AddObserver(vtkCommand::ModifiedEvent, pCallBack);
	}
	
	Renderer = pRenderer;

	Update();
}

void vtkErRenderInfo::Update()
{
    if (this->Renderer != NULL)
    {
        vtkRenderWindow* pRenderWindow = Renderer->GetRenderWindow();

        int* pRenderSize = pRenderWindow->GetSize();

		m_FrameBuffer.Resize(CResolution2D(pRenderSize[0], pRenderSize[1]));

		if (pRenderSize[0] != this->RendererInfo.m_FilmWidth || pRenderSize[1] != this->RendererInfo.m_FilmHeight)
        {
            RendererInfo.m_FilmWidth	= pRenderSize[0];
            RendererInfo.m_FilmHeight	= pRenderSize[1];
			RendererInfo.m_FilmNoPixels	= RendererInfo.m_FilmWidth * RendererInfo.m_FilmHeight;
        }
        
		vtkCamera* pCamera = this->Renderer->GetActiveCamera();

		RendererInfo.m_Camera.m_Pos.x		= 0.001f * pCamera->GetPosition()[0];
        RendererInfo.m_Camera.m_Pos.y		= 0.001f * pCamera->GetPosition()[1];
        RendererInfo.m_Camera.m_Pos.z		= 0.001f * pCamera->GetPosition()[2];
		
		RendererInfo.m_Camera.m_Target.x	= 0.001f * pCamera->GetFocalPoint()[0];
        RendererInfo.m_Camera.m_Target.y	= 0.001f * pCamera->GetFocalPoint()[1];
        RendererInfo.m_Camera.m_Target.z	= 0.001f * pCamera->GetFocalPoint()[2];

        RendererInfo.m_Camera.m_Up.x		= pCamera->GetViewUp()[0];
        RendererInfo.m_Camera.m_Up.y		= pCamera->GetViewUp()[1];
        RendererInfo.m_Camera.m_Up.z		= pCamera->GetViewUp()[2];

		RendererInfo.m_Camera.m_N			= Normalize(RendererInfo.m_Camera.m_Target - RendererInfo.m_Camera.m_Pos);
		RendererInfo.m_Camera.m_U			= -Normalize(Cross(RendererInfo.m_Camera.m_Up, RendererInfo.m_Camera.m_N));
		RendererInfo.m_Camera.m_V			= -Normalize(Cross(RendererInfo.m_Camera.m_N, RendererInfo.m_Camera.m_U));

		vtkErCamera* pErCamera = dynamic_cast<vtkErCamera*>(pCamera);

		if (pErCamera)
		{
			RendererInfo.m_Camera.m_FocalDistance = pErCamera->GetFocalDistance();
		}
		else
		{
			RendererInfo.m_Camera.m_FocalDistance = vtkErCamera::DefaultFocalDistance();
		}

        double ClippingRange[2];

        pCamera->GetClippingRange(ClippingRange);

        this->RendererInfo.m_Camera.m_ClipNear.x	= (float)ClippingRange[0];
        this->RendererInfo.m_Camera.m_ClipFar.y		= (float)ClippingRange[1];

		
		float Scale = 0.0f;

		Scale = tanf((0.5f * pCamera->GetViewAngle() / RAD_F));

		const float AspectRatio = (float)RendererInfo.m_FilmHeight / (float)RendererInfo.m_FilmWidth;

		if (AspectRatio > 1.0f)
		{
			RendererInfo.m_Camera.m_Screen[0][0] = -Scale;
			RendererInfo.m_Camera.m_Screen[0][1] = Scale;
			RendererInfo.m_Camera.m_Screen[1][0] = -Scale * AspectRatio;
			RendererInfo.m_Camera.m_Screen[1][1] = Scale * AspectRatio;
		}
		else
		{
			RendererInfo.m_Camera.m_Screen[0][0] = -Scale / AspectRatio;
			RendererInfo.m_Camera.m_Screen[0][1] = Scale / AspectRatio;
			RendererInfo.m_Camera.m_Screen[1][0] = -Scale;
			RendererInfo.m_Camera.m_Screen[1][1] = Scale;
		}

		RendererInfo.m_Camera.m_InvScreen.x = (RendererInfo.m_Camera.m_Screen[0][1] - RendererInfo.m_Camera.m_Screen[0][0]) / (float)RendererInfo.m_FilmWidth;
		RendererInfo.m_Camera.m_InvScreen.y = (RendererInfo.m_Camera.m_Screen[1][1] - RendererInfo.m_Camera.m_Screen[1][0]) / (float)RendererInfo.m_FilmHeight;

		RendererInfo.m_Camera.m_ApertureSize = (float)pCamera->GetFocalDisk();

		RendererInfo.m_FilterWidth = 2;

		RendererInfo.m_FilterWeights[0] = 1.0f;
		RendererInfo.m_FilterWeights[1] = 0.5f;
		RendererInfo.m_FilterWeights[2] = 0.1f;
		RendererInfo.m_FilterWeights[3] = 1.01f;

		RendererInfo.m_Gamma		= 2.2;
		RendererInfo.m_InvGamma		= 1.0f / RendererInfo.m_Gamma;
		RendererInfo.m_Exposure		= 10.0f;
		RendererInfo.m_InvExposure	= 1.0f / RendererInfo.m_Exposure;
//		RendererInfo.m_NoIterations = 1.0f;

		RendererInfo.m_Denoise.m_Enabled			= false;
		RendererInfo.m_Denoise.m_Noise				= 0.05f;
		RendererInfo.m_Denoise.m_LerpC				= 0.01f;
		RendererInfo.m_Denoise.m_WindowRadius		= 6.0f;
		RendererInfo.m_Denoise.m_WindowArea			= (2.0f * RendererInfo.m_Denoise.m_WindowRadius + 1.0f) * (2.0f * RendererInfo.m_Denoise.m_WindowRadius + 1.0f);
		RendererInfo.m_Denoise.m_InvWindowArea		= 1.0f / RendererInfo.m_Denoise.m_WindowArea;
		RendererInfo.m_Denoise.m_WeightThreshold	= 0.1f;
		RendererInfo.m_Denoise.m_LerpThreshold		= 0.0f;

		RendererInfo.m_Shadows = true;

		vtkLightCollection* pLights = Renderer->GetLights();

		m_Lighting.m_NoLights = pLights->GetNumberOfItems();

		 
 
		pLights->InitTraversal();
		vtkLight* pLight = pLights->GetNextItem();

		int count = 0;

		while (pLight != 0)
		{
			vtkErAreaLight* pAreaLight = dynamic_cast<vtkErAreaLight*>(pLight);
			vtkErBackgroundLight* pBackgroundLight = dynamic_cast<vtkErBackgroundLight*>(pLight);

			if (pAreaLight && pAreaLight->GetEnabled())
			{
				m_Lighting.m_Type[count] = 1;

				m_Lighting.m_P[count].x = pLight->GetPosition()[0];
				m_Lighting.m_P[count].y = pLight->GetPosition()[1];
				m_Lighting.m_P[count].z = pLight->GetPosition()[2];

				ColorXYZf Color;

				Color.FromRGB(pLight->GetDiffuseColor()[0], pLight->GetDiffuseColor()[1], pLight->GetDiffuseColor()[2]);

				m_Lighting.m_Color[count].x = Color[0];
				m_Lighting.m_Color[count].y = Color[1];
				m_Lighting.m_Color[count].z = Color[2];

				count++;
			}

			if (pBackgroundLight && pBackgroundLight->GetEnabled())
			{
				m_Lighting.m_Type[count] = 2;

				ColorXYZf Color;

				Color.FromRGB(pLight->GetDiffuseColor()[0], pLight->GetDiffuseColor()[1], pLight->GetDiffuseColor()[2]);

				m_Lighting.m_Color[count].x = Color[0];
				m_Lighting.m_Color[count].y = Color[1];
				m_Lighting.m_Color[count].z = Color[2];

				count++;
			}

			pLight = pLights->GetNextItem();
		}

    }
}

void vtkErRenderInfo::Reset()
{
	RendererInfo.m_NoIterations = 0;
}