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

#include "VtkErVolumeMapper.h"
#include "VtkErVolumeProperty.h"
#include "vtkErAreaLight.h"
#include "VtkErBackgroundLight.h"
#include "VtkErCamera.h"

#include "Core.cuh"
#include "cutil_math.h"

#include "Geometry.h"

#include "vtkImageActor.h"
#include "vtkgl.h"
#include "vtkCommand.h"

#include <vtkImageData.h>
#include <vtkTimerLog.h>
#include <vtkImageReslice.h>
#include <vtkCamera.h>
#include <vtkTransform.h>

void vtkErResetCommand::Execute(vtkObject*, unsigned long, void *)
{
	this->VolumeMapper->Reset();
}

void vtkErResetCommand::SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper)
{
	if (pVolumeMapper == NULL)
		return;

	this->VolumeMapper = pVolumeMapper;
};

void vtkErUpdateSlicingCommand::Execute(vtkObject*, unsigned long, void *)
{
	if (this->VolumeMapper == NULL)
		return;

	if (!this->VolumeMapper->GetSliceWidget())
		return;

	this->VolumeMapper->Slicing.m_NoSlices = 6;

	for (int i = 0; i < 6; i++)
	{
		this->VolumeMapper->Slicing.m_Position[i].x		= this->VolumeMapper->GetSliceWidget()->GetSlicePlaneWidget(i)->GetCenter()[0];
		this->VolumeMapper->Slicing.m_Position[i].y		= this->VolumeMapper->GetSliceWidget()->GetSlicePlaneWidget(i)->GetCenter()[1];
		this->VolumeMapper->Slicing.m_Position[i].z		= this->VolumeMapper->GetSliceWidget()->GetSlicePlaneWidget(i)->GetCenter()[2];
		this->VolumeMapper->Slicing.m_Normal[i].x		= this->VolumeMapper->GetSliceWidget()->GetSlicePlaneWidget(i)->GetNormal()[0];
		this->VolumeMapper->Slicing.m_Normal[i].y		= this->VolumeMapper->GetSliceWidget()->GetSlicePlaneWidget(i)->GetNormal()[1];
		this->VolumeMapper->Slicing.m_Normal[i].z		= this->VolumeMapper->GetSliceWidget()->GetSlicePlaneWidget(i)->GetNormal()[2];
	}

	this->VolumeMapper->Reset();
}

void vtkErUpdateSlicingCommand::SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper)
{
	if (pVolumeMapper == NULL)
		return;

	this->VolumeMapper = pVolumeMapper;
};

void vtkErUpdateLightingCommand::Execute(vtkObject*, unsigned long, void*)
{
	if (this->VolumeMapper == NULL)
		return;

	vtkLightCollection* pLights = this->VolumeMapper->GetLights();

	this->VolumeMapper->Lighting.m_NoLights = pLights->GetNumberOfItems();
	
	vtkDebugWithObjectMacro(this->VolumeMapper, "Processing lighting changes");

	pLights->InitTraversal();
	
	vtkLight* pLight = pLights->GetNextItem();

	int count = 0;

	while (pLight != 0)
	{
		vtkErAreaLight*			pErAreaLight		= dynamic_cast<vtkErAreaLight*>(pLight);
		vtkErBackgroundLight*	pErBackgroundLight	= dynamic_cast<vtkErBackgroundLight*>(pLight);

		Light& L = this->VolumeMapper->Lighting.m_Lights[count];

		if (pErAreaLight || pErBackgroundLight)
		{
			// ER area light
			if (pErAreaLight && pErAreaLight->GetEnabled())
			{
				L.m_Type		= 0;
				L.m_OneSided	= pErAreaLight->GetOneSided();

				ColorXYZf Color;

				Color.FromRGB(pErAreaLight->GetColor()[0], pErAreaLight->GetColor()[1], pErAreaLight->GetColor()[2]);

				L.m_Color.x = Color[0] * pErAreaLight->GetIntensity();
				L.m_Color.y = Color[1] * pErAreaLight->GetIntensity();
				L.m_Color.z = Color[2] * pErAreaLight->GetIntensity();
				
				vtkSmartPointer<vtkMatrix4x4> TM = vtkMatrix4x4::New(), InvTM = vtkMatrix4x4::New();

				TM->DeepCopy(pErAreaLight->GetTransformMatrix());
				InvTM->DeepCopy(TM);

				InvTM->Invert();

				L.m_ShapeType = pErAreaLight->GetShapeType(); 

				for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						L.m_TM.NN[i][j]		= (float)TM->GetElement(i, j);
						L.m_InvTM.NN[i][j]	= (float)InvTM->GetElement(i, j);
					}
				}

				L.m_Size			= make_float3(pErAreaLight->GetSize()[0], pErAreaLight->GetSize()[1], pErAreaLight->GetSize()[2]);
				L.m_InnerRadius		= pErAreaLight->GetInnerRadius();
				L.m_OuterRadius		= pErAreaLight->GetOuterRadius();
				L.m_Area			= pErAreaLight->GetArea();

				count++;
			}

			// ER background light
			if (pErBackgroundLight && pErBackgroundLight->GetEnabled())
			{
				L.m_Type = 1;

				ColorXYZf Color;

				Color.FromRGB(pErBackgroundLight->GetDiffuseColor()[0], pErBackgroundLight->GetDiffuseColor()[1], pErBackgroundLight->GetDiffuseColor()[2]);

				L.m_Color.x = Color[0] * pErAreaLight->GetIntensity();
				L.m_Color.y = Color[1] * pErAreaLight->GetIntensity();
				L.m_Color.z = Color[2] * pErAreaLight->GetIntensity();

				count++;
			}
		}
		else
		{
			// VTK light
			L.m_Type = 0;

//			L.m_P.x = pLight->GetPosition()[0];
//			L.m_P.y = pLight->GetPosition()[1];
//			L.m_P.z = pLight->GetPosition()[2];

			ColorXYZf Color;

			Color.FromRGB(pErAreaLight->GetDiffuseColor()[0], pErAreaLight->GetDiffuseColor()[1], pErAreaLight->GetDiffuseColor()[2]);

			L.m_Color.x = Color[0] * pErAreaLight->GetIntensity();
			L.m_Color.y = Color[1] * pErAreaLight->GetIntensity();
			L.m_Color.z = Color[2] * pErAreaLight->GetIntensity();
				
			L.m_ShapeType = pErAreaLight->GetShapeType(); 

			count++;
		}

		pLight = pLights->GetNextItem();
	}

	this->VolumeMapper->Reset();
}

void vtkErUpdateLightingCommand::SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper)
{
	if (pVolumeMapper == NULL)
		return;

	this->VolumeMapper = pVolumeMapper;
};

void vtkErUpdateCameraCommand::Execute(vtkObject*, unsigned long, void*)
{
	if (this->VolumeMapper == NULL)
		return;

	vtkRenderer* pRenderer = this->VolumeMapper->Renderer;

	if (pRenderer == NULL)
		return;

	vtkRenderWindow* pRenderWindow = pRenderer->GetRenderWindow();

	if (pRenderWindow == NULL)
		return;

	int* pRenderSize = pRenderWindow->GetSize();

	this->VolumeMapper->Camera.m_FilmWidth		= pRenderSize[0];
	this->VolumeMapper->Camera.m_FilmHeight		= pRenderSize[1];
	this->VolumeMapper->Camera.m_FilmNoPixels	= this->VolumeMapper->Camera.m_FilmWidth * this->VolumeMapper->Camera.m_FilmHeight;

	vtkCamera* pCamera = pRenderer->GetActiveCamera();

	this->VolumeMapper->Camera.m_Pos.x		= pCamera->GetPosition()[0];
	this->VolumeMapper->Camera.m_Pos.y		= pCamera->GetPosition()[1];
	this->VolumeMapper->Camera.m_Pos.z		= pCamera->GetPosition()[2];
		
	this->VolumeMapper->Camera.m_Target.x	= pCamera->GetFocalPoint()[0];
	this->VolumeMapper->Camera.m_Target.y	= pCamera->GetFocalPoint()[1];
	this->VolumeMapper->Camera.m_Target.z	= pCamera->GetFocalPoint()[2];

	this->VolumeMapper->Camera.m_Up.x		= pCamera->GetViewUp()[0];
	this->VolumeMapper->Camera.m_Up.y		= pCamera->GetViewUp()[1];
	this->VolumeMapper->Camera.m_Up.z		= pCamera->GetViewUp()[2];

	this->VolumeMapper->Camera.m_N			= normalize(this->VolumeMapper->Camera.m_Target - this->VolumeMapper->Camera.m_Pos);
	this->VolumeMapper->Camera.m_U			= -normalize(cross(this->VolumeMapper->Camera.m_Up, this->VolumeMapper->Camera.m_N));
	this->VolumeMapper->Camera.m_V			= -normalize(cross(this->VolumeMapper->Camera.m_N, this->VolumeMapper->Camera.m_U));

	pCamera->GetViewTransformMatrix();

	vtkErCamera* pErCamera = dynamic_cast<vtkErCamera*>(pCamera);

	if (pErCamera)
	{
		this->VolumeMapper->Camera.m_FocalDistance	= pErCamera->GetFocalDistance();
		this->VolumeMapper->Camera.m_Exposure		= pErCamera->GetExposure();
	}
	else
	{
		this->VolumeMapper->Camera.m_FocalDistance	= vtkErCamera::DefaultFocalDistance();
		this->VolumeMapper->Camera.m_Exposure		= vtkErCamera::DefaultExposure();
	}

	this->VolumeMapper->Camera.m_InvExposure = 1.0f / this->VolumeMapper->Camera.m_Exposure;

	double ClippingRange[2];

	pCamera->GetClippingRange(ClippingRange);

	this->VolumeMapper->Camera.m_ClipNear.x	= (float)ClippingRange[0];
	this->VolumeMapper->Camera.m_ClipFar.y	= (float)ClippingRange[1];

	this->VolumeMapper->Camera.m_Gamma		= 2.2f;
	this->VolumeMapper->Camera.m_InvGamma	= 1.0f / this->VolumeMapper->Camera.m_Gamma;

	float Scale = 0.0f;

	Scale = tanf((0.5f * pCamera->GetViewAngle() / RAD_F));

	const float AspectRatio = (float)this->VolumeMapper->Camera.m_FilmHeight / (float)this->VolumeMapper->Camera.m_FilmWidth;

	if (AspectRatio > 1.0f)
	{
		this->VolumeMapper->Camera.m_Screen[0][0] = -Scale;
		this->VolumeMapper->Camera.m_Screen[0][1] = Scale;
		this->VolumeMapper->Camera.m_Screen[1][0] = -Scale * AspectRatio;
		this->VolumeMapper->Camera.m_Screen[1][1] = Scale * AspectRatio;
	}
	else
	{
		this->VolumeMapper->Camera.m_Screen[0][0] = -Scale / AspectRatio;
		this->VolumeMapper->Camera.m_Screen[0][1] = Scale / AspectRatio;
		this->VolumeMapper->Camera.m_Screen[1][0] = -Scale;
		this->VolumeMapper->Camera.m_Screen[1][1] = Scale;
	}

	this->VolumeMapper->Camera.m_InvScreen.x = (this->VolumeMapper->Camera.m_Screen[0][1] - this->VolumeMapper->Camera.m_Screen[0][0]) / (float)this->VolumeMapper->Camera.m_FilmWidth;
	this->VolumeMapper->Camera.m_InvScreen.y = (this->VolumeMapper->Camera.m_Screen[1][1] - this->VolumeMapper->Camera.m_Screen[1][0]) / (float)this->VolumeMapper->Camera.m_FilmHeight;

	this->VolumeMapper->Camera.m_ApertureSize = (float)pCamera->GetFocalDisk();

	this->VolumeMapper->Reset();
}

void vtkErUpdateCameraCommand::SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper)
{
	if (pVolumeMapper == NULL)
		return;

	this->VolumeMapper = pVolumeMapper;
};

void vtkErUpdateBlurCommand::Execute(vtkObject*, unsigned long, void *)
{
	if (this->VolumeMapper == NULL)
		return;

	this->VolumeMapper->Blur.m_FilterWidth			= 1;
	this->VolumeMapper->Blur.m_FilterWeights[0]		= 0.7f;
	this->VolumeMapper->Blur.m_FilterWeights[1]		= 0.4f;
	this->VolumeMapper->Blur.m_FilterWeights[2]		= 0.1f;
	this->VolumeMapper->Blur.m_FilterWeights[3]		= 1.01f;

	this->VolumeMapper->Reset();
}

void vtkErUpdateBlurCommand::SetVolumeMapper(vtkErVolumeMapper* pVolumeMapper)
{
	if (pVolumeMapper == NULL)
		return;

	this->VolumeMapper = pVolumeMapper;
};

vtkStandardNewMacro(vtkErNoiseReduction);

vtkStandardNewMacro(vtkErVolumeMapper);

vtkErVolumeMapper::vtkErVolumeMapper(void)
{
	SetCudaDevice(0);

	glGenTextures(1, &TextureID);

	this->ResetCallBack = vtkErResetCommand::New();
	this->ResetCallBack->SetVolumeMapper(this);

	this->UpdateSlicingCommand = vtkErUpdateSlicingCommand::New();
	this->UpdateSlicingCommand->SetVolumeMapper(this);

	this->UpdateLightingCommand = vtkErUpdateLightingCommand::New();
	this->UpdateLightingCommand->SetVolumeMapper(this);

	this->UpdateCameraCommand = vtkErUpdateCameraCommand::New();
	this->UpdateCameraCommand->SetVolumeMapper(this);
	
	this->UpdateBlur = vtkErUpdateBlurCommand::New();
	this->UpdateBlur->SetVolumeMapper(this);

	this->Lights = vtkLightCollection::New();

	this->Renderer			= NULL;
	this->ActiveCamera		= NULL;
	this->VolumeProperty	= NULL;

	this->BorderWidget		= vtkTextActor::New();


//	this->DebugOn();

//	vtkDebugMacro("SDds");
}  

vtkErVolumeMapper::~vtkErVolumeMapper(void)
{
}

void vtkErVolumeMapper::SetInput(vtkImageData* pImageData)
{
	this->Superclass::SetInput(pImageData);

    if (pImageData == NULL)
    {
    }
    else if (pImageData != this->Intensity)
    {
		vtkSmartPointer<vtkImageCast> ImageCast = vtkImageCast::New();
		
		ImageCast->SetInput(pImageData);

		ImageCast->SetOutputScalarTypeToFloat();
		ImageCast->Update();

		this->Intensity = ImageCast->GetOutput();

		int* pResolution = this->Intensity->GetExtent();
		
		double* pBounds = this->Intensity->GetBounds();

		this->Volume.m_Size.x = pBounds[1] - pBounds[0];
		this->Volume.m_Size.y = pBounds[3] - pBounds[2];
		this->Volume.m_Size.z = pBounds[5] - pBounds[4];

		this->Volume.m_InvSize.x = 1.0f / this->Volume.m_Size.x;
		this->Volume.m_InvSize.y = 1.0f / this->Volume.m_Size.y;
		this->Volume.m_InvSize.z = 1.0f / this->Volume.m_Size.z;

		this->Volume.m_Extent.x	= pResolution[1] - pResolution[0];
		this->Volume.m_Extent.y	= pResolution[3] - pResolution[2];
		this->Volume.m_Extent.z	= pResolution[5] - pResolution[4];
		
		this->Volume.m_InvExtent.x	= this->Volume.m_Extent.x != 0 ? 1.0f / (float)this->Volume.m_Extent.x : 0.0f;
		this->Volume.m_InvExtent.y	= this->Volume.m_Extent.y != 0 ? 1.0f / (float)this->Volume.m_Extent.y : 0.0f;
		this->Volume.m_InvExtent.z	= this->Volume.m_Extent.z != 0 ? 1.0f / (float)this->Volume.m_Extent.z : 0.0f;

		cudaExtent Extent;

		Extent.width	= pResolution[1] + 1;
		Extent.height	= pResolution[3] + 1;
		Extent.depth	= pResolution[5] + 1;

		double* pIntensityRange = this->Intensity->GetScalarRange();
		
		this->Volume.m_IntensityMin			= (float)pIntensityRange[0];
		this->Volume.m_IntensityMax			= (float)pIntensityRange[1];
		this->Volume.m_IntensityRange		= this->Volume.m_IntensityMax - this->Volume.m_IntensityMin;
		this->Volume.m_IntensityInvRange	= 1.0f / this->Volume.m_IntensityRange;

		double* pSpacing = this->Intensity->GetSpacing();

		this->Volume.m_Spacing.x = (float)pSpacing[0];
		this->Volume.m_Spacing.y = (float)pSpacing[1];
		this->Volume.m_Spacing.z = (float)pSpacing[2];

		this->Volume.m_InvSpacing.x = this->Volume.m_Spacing.x != 0.0f ? 1.0f / this->Volume.m_Spacing.x : 0.0f;
		this->Volume.m_InvSpacing.y = this->Volume.m_Spacing.y != 0.0f ? 1.0f / this->Volume.m_Spacing.y : 0.0f;
		this->Volume.m_InvSpacing.z = this->Volume.m_Spacing.z != 0.0f ? 1.0f / this->Volume.m_Spacing.z : 0.0f;

		this->Volume.m_MinAABB.x	= pBounds[0];
		this->Volume.m_MinAABB.y	= pBounds[2];
		this->Volume.m_MinAABB.z	= pBounds[4];

		this->Volume.m_InvMinAABB.x	= this->Volume.m_MinAABB.x != 0.0f ? 1.0f / this->Volume.m_MinAABB.x : 0.0f;
		this->Volume.m_InvMinAABB.y	= this->Volume.m_MinAABB.y != 0.0f ? 1.0f / this->Volume.m_MinAABB.y : 0.0f;
		this->Volume.m_InvMinAABB.z	= this->Volume.m_MinAABB.z != 0.0f ? 1.0f / this->Volume.m_MinAABB.z : 0.0f;

		this->Volume.m_MaxAABB.x	= pBounds[1];
		this->Volume.m_MaxAABB.y	= pBounds[3];
		this->Volume.m_MaxAABB.z	= pBounds[5];

		this->Volume.m_InvMaxAABB.x	= this->Volume.m_MaxAABB.x != 0.0f ? 1.0f / this->Volume.m_MaxAABB.x : 0.0f;
		this->Volume.m_InvMaxAABB.y	= this->Volume.m_MaxAABB.y != 0.0f ? 1.0f / this->Volume.m_MaxAABB.y : 0.0f;
		this->Volume.m_InvMaxAABB.z	= this->Volume.m_MaxAABB.z != 0.0f ? 1.0f / this->Volume.m_MaxAABB.z : 0.0f;

		this->Volume.m_GradientDelta		= 1.0f;
		this->Volume.m_InvGradientDelta		= 1.0f;
		this->Volume.m_GradientDeltaX.x		= this->Volume.m_GradientDelta;
		this->Volume.m_GradientDeltaX.y		= 0.0f;
		this->Volume.m_GradientDeltaX.z		= 0.0f;
		this->Volume.m_GradientDeltaY.x		= 0.0f;
		this->Volume.m_GradientDeltaY.y		= this->Volume.m_GradientDelta;
		this->Volume.m_GradientDeltaY.z		= 0.0f;
		this->Volume.m_GradientDeltaZ.x		= 0.0f;
		this->Volume.m_GradientDeltaZ.y		= 0.0f;
		this->Volume.m_GradientDeltaZ.z		= this->Volume.m_GradientDelta;

		vtkErVolumeProperty* pErVolumeProperty = dynamic_cast<vtkErVolumeProperty*>(this->VolumeProperty);

		if (pErVolumeProperty)
		{
			this->Volume.m_DensityScale		= pErVolumeProperty->GetDensityScale();
			this->Volume.m_StepSize			= pErVolumeProperty->GetStepSizeFactorPrimary() * (this->Volume.m_MaxAABB.x / this->Volume.m_Extent.x);
			this->Volume.m_StepSizeShadow	= this->Volume.m_StepSize * pErVolumeProperty->GetStepSizeFactorSecondary();
			this->Volume.m_GradientDelta	= pErVolumeProperty->GetGradientDeltaFactor() * this->Volume.m_Spacing.x;
			this->Volume.m_InvGradientDelta	= 1.0f / this->Volume.m_GradientDelta;
			this->Volume.m_GradientFactor	= pErVolumeProperty->GetGradientFactor();
			this->Volume.m_ShadingType		= pErVolumeProperty->GetShadingType();
		}
		else
		{
			this->Volume.m_DensityScale		= vtkErVolumeProperty::DefaultDensityScale();
			this->Volume.m_StepSize			= vtkErVolumeProperty::DefaultStepSizeFactorPrimary() * (this->Volume.m_MaxAABB.x / this->Volume.m_Extent.x);
			this->Volume.m_StepSizeShadow	= vtkErVolumeProperty::DefaultStepSizeFactorSecondary() * this->Volume.m_StepSize;
			this->Volume.m_GradientDelta	= vtkErVolumeProperty::DefaultGradientDeltaFactor() * this->Volume.m_Spacing.x;
			this->Volume.m_InvGradientDelta	= 1.0f / this->Volume.m_GradientDelta;
			this->Volume.m_GradientFactor	= vtkErVolumeProperty::DefaultGradientFactor();
			this->Volume.m_ShadingType		= vtkErVolumeProperty::DefaultShadingType();
		}

		this->Volume.m_StepSize			= this->Volume.m_Spacing.x;
		this->Volume.m_StepSizeShadow	= this->Volume.m_Spacing.x;

		this->Volume.m_GradientDeltaX.x = this->Volume.m_GradientDelta;
		this->Volume.m_GradientDeltaX.y = 0.0f;
		this->Volume.m_GradientDeltaX.z = 0.0f;

		this->Volume.m_GradientDeltaY.x = 0.0f;
		this->Volume.m_GradientDeltaY.y = this->Volume.m_GradientDelta;
		this->Volume.m_GradientDeltaY.z = 0.0f;

		this->Volume.m_GradientDeltaZ.x = 0.0f;
		this->Volume.m_GradientDeltaZ.y = 0.0f;
		this->Volume.m_GradientDeltaZ.z = this->Volume.m_GradientDelta;

		BindIntensityBuffer((float*)this->Intensity->GetScalarPointer(), Extent);
    }
}

void vtkErVolumeMapper::Render(vtkRenderer* pRenderer, vtkVolume* pVolume)
{
	if (pRenderer == NULL || pVolume == NULL)
		return;
	
	if (this->Renderer != pRenderer)
	{
		this->Renderer = pRenderer;
	}

	if (this->ActiveCamera != pRenderer->GetActiveCamera())
	{
		this->ActiveCamera = pRenderer->GetActiveCamera();
		this->ActiveCamera->AddObserver(vtkCommand::ModifiedEvent, this->UpdateCameraCommand, 0.0f);	
	}

	if (this->VolumeProperty != pVolume->GetProperty())
	{
		this->VolumeProperty = pVolume->GetProperty();

		this->VolumeProperty->AddObserver(vtkCommand::ModifiedEvent, this->UpdateBlur, 0.0f);
		this->VolumeProperty->Modified();
	}

	char FPS[255];

	sprintf(FPS, "FPS: %0.5f", 1.0 / this->TimeToDraw); 
	
	this->BorderWidget->SetInput(FPS);
	vtkTextProperty* tprop = this->BorderWidget->GetTextProperty();
	tprop->SetFontFamilyToCourier();
		tprop->BoldOn();
//		tprop->ShadowOn();
		tprop->SetLineSpacing(1.0);
		tprop->SetFontSize(11);
		tprop->SetColor(0.1, 0.1, 0.1);
		tprop->SetShadowOffset(1,1);
	this->BorderWidget->SetDisplayPosition(10, 10);
	pRenderer->AddActor2D(this->BorderWidget);

//	this->BorderWidget->SetInteractor(this->Renderer->GetRenderWindow()->GetInteractor());
//	this->BorderWidget->SetEnabled(1);
	
	// Start the timer to time the length of this render
	vtkSmartPointer<vtkTimerLog> Timer = vtkTimerLog::New();
	Timer->StartTimer();

	UploadVolumeProperty(pVolume->GetProperty());

	int* pRenderSize = pRenderer->GetRenderWindow()->GetSize();

	this->Host.Resize(CResolution2D(pRenderSize[0], pRenderSize[1]));
	this->FrameBuffer.Resize(CResolution2D(pRenderSize[0], pRenderSize[1]));

	Scattering.m_NoIterations += 1;

	RenderEstimate(&this->Volume, &this->Camera, &this->Lighting, &this->Slicing, &this->Denoise, &this->Scattering, &this->Blur, &this->FrameBuffer);

	cudaMemcpy(this->Host.GetPtr(), this->FrameBuffer.m_EstimateRgbaLdr.GetPtr(), this->Host.GetSize(), cudaMemcpyDeviceToHost);
	
	glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, TextureID);
    
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, pRenderSize[0], pRenderSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, this->Host.GetPtr());
	glBindTexture(GL_TEXTURE_2D, TextureID);

	double d = 0.5;
	
    pRenderer->SetDisplayPoint(0,0,d);
    pRenderer->DisplayToWorld();
    double coordinatesA[4];
    pRenderer->GetWorldPoint(coordinatesA);

    pRenderer->SetDisplayPoint(pRenderSize[0],0,d);
    pRenderer->DisplayToWorld();
    double coordinatesB[4];
    pRenderer->GetWorldPoint(coordinatesB);

    pRenderer->SetDisplayPoint(pRenderSize[0], pRenderSize[1],d);
    pRenderer->DisplayToWorld();
    double coordinatesC[4];
    pRenderer->GetWorldPoint(coordinatesC);

    pRenderer->SetDisplayPoint(0,pRenderSize[1],d);
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
	/**/

	// Stop the timer
	Timer->StopTimer();
	
	const double Time = Timer->GetElapsedTime();

	this->TimeToDraw = Time;
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
	float	IOR[128];
	float	Emission[3][128];

	if (pErVolumeProperty == NULL)
	{
		vtkErrorMacro("Incompatible volume property (reverting to default property), use vtkErVolumeProperty!");
		
		/*
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
		*/
	}
	else
	{
		if (pErVolumeProperty->GetDirty())
		{
			this->Reset();
			pErVolumeProperty->SetDirty(false);
		}

		this->Volume.m_ShadingType = pErVolumeProperty->GetShadingType();

		pErVolumeProperty->GetOpacity()->GetTable(pRange[0], pRange[1], N, Opacity);
		pErVolumeProperty->GetDiffuse(0)->GetTable(pRange[0], pRange[1], N, Diffuse[0]);
		pErVolumeProperty->GetDiffuse(1)->GetTable(pRange[0], pRange[1], N, Diffuse[1]);
		pErVolumeProperty->GetDiffuse(2)->GetTable(pRange[0], pRange[1], N, Diffuse[2]);
		pErVolumeProperty->GetSpecular(0)->GetTable(pRange[0], pRange[1], N, Specular[0]);
		pErVolumeProperty->GetSpecular(1)->GetTable(pRange[0], pRange[1], N, Specular[1]);
		pErVolumeProperty->GetSpecular(2)->GetTable(pRange[0], pRange[1], N, Specular[2]);
		pErVolumeProperty->GetGlossiness()->GetTable(pRange[0], pRange[1], N, Glossiness);
		pErVolumeProperty->GetIOR()->GetTable(pRange[0], pRange[1], N, IOR);
		pErVolumeProperty->GetEmission(0)->GetTable(pRange[0], pRange[1], N, Emission[0]);
		pErVolumeProperty->GetEmission(1)->GetTable(pRange[0], pRange[1], N, Emission[1]);
		pErVolumeProperty->GetEmission(2)->GetTable(pRange[0], pRange[1], N, Emission[2]);
	}

	BindTransferFunctions1D(Opacity, Diffuse, Specular, Glossiness, IOR, Emission, N);
}

void vtkErVolumeMapper::SetSliceWidget(vtkErBoxWidget* pSliceWidget)
{
	if (pSliceWidget == NULL)
	{
		vtkErrorMacro("Slice widget is NULL!");
		return;
	}

	this->SliceWidget = pSliceWidget;

	for (int i = 0; i < 6; i++)
		pSliceWidget->GetSlicePlaneWidget(i)->AddObserver(vtkCommand::InteractionEvent, this->UpdateSlicingCommand, 0.0);
}

void vtkErVolumeMapper::Reset()
{
	this->Scattering.m_NoIterations = 0;
}

void vtkErVolumeMapper::AddLight(vtkLight* pLight)
{
	if (pLight == NULL)
	{
		vtkErrorMacro("Supplied light pointer is NULL!");
		return;
	}

	this->Lights->AddItem(pLight);

	pLight->AddObserver(vtkCommand::AnyEvent, this->UpdateLightingCommand, 0.0);
	pLight->Modified();

	// Adding a light invalidates the rendering, so restart
	this->Reset();
}

void vtkErVolumeMapper::RemoveLight(vtkLight* pLight)
{
	if (pLight == NULL)
	{
		vtkErrorMacro("Supplied light pointer is NULL!");
		return;
	}

	this->Lights->RemoveItem(pLight);

	// Adding a light invalidates the rendering, so restart
	this->Reset();
}

vtkLightCollection* vtkErVolumeMapper::GetLights()
{
	return this->Lights.GetPointer();
}

vtkErNoiseReduction* vtkErVolumeMapper::GetNoiseReduction()
{
	return this->NoiseReduction.GetPointer();
}

