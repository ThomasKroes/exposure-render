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

#include "vtkErRenderInfo.h"
#include "vtkErAreaLight.h"
#include "vtkErBackgroundLight.h"
#include "vtkErCamera.h"

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
        

		/*
		RendererInfo.m_FilterWidth = 2;

		RendererInfo.m_FilterWeights[0] = 1.0f;
		RendererInfo.m_FilterWeights[1] = 0.5f;
		RendererInfo.m_FilterWeights[2] = 0.1f;
		RendererInfo.m_FilterWeights[3] = 1.01f;

		RendererInfo.m_Gamma		= 2.2;
		RendererInfo.m_InvGamma		= 1.0f / RendererInfo.m_Gamma;
		
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
		*/
		

//		Renderer->GetRenderWindow()->GetInteractor()->get
    }
}

void vtkErRenderInfo::Reset()
{
	/*
	RendererInfo.m_NoIterations = 0;
	m_FrameBuffer.m_DisplayEstimateRgbLdr.Reset();
	m_FrameBuffer.m_EstimateRgbaLdr.Reset();
	m_FrameBuffer.m_FrameBlurXyza.Reset();
	m_FrameBuffer.m_FrameEstimateXyza.Reset();
	*/
}
