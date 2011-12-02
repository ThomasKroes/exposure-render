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

#include "vtkErVolumeInfo.h"
#include "vtkErVolumeProperty.h"

#include "Core.cuh"

#include <vtkBoundingBox.h>
#include <vtkImageReslice.h>

vtkCxxRevisionMacro(vtkErVolumeInfo, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkErVolumeInfo);

vtkErVolumeInfo::vtkErVolumeInfo() :
	m_pIntensity(NULL),
	m_pGradientMagnitude(NULL),
	Volume(NULL)
{
}

vtkErVolumeInfo::~vtkErVolumeInfo()
{
}

void vtkErVolumeInfo::SetInputData(vtkImageData* pInputData)
{
	if (m_pIntensity != NULL)
		return;

    if (pInputData == NULL)
    {
//        this->CudaInputBuffer.Free();
    }
    else if (pInputData != m_pIntensity)
    {
		/*
		vtkSmartPointer<vtkImageReslice> Reslicer = vtkImageReslice::New();

		Reslicer->SetInput(pInputData);
		Reslicer->SetInterpolate(1);
		Reslicer->SetOutputExtent(0, 127, 0, 127, 0, 127);
		Reslicer->Update();

		vtkSmartPointer<vtkImageCast> ImageCast = vtkImageCast::New();
		
		ImageCast->SetInput(pInputData);//Reslicer->GetOutput());

		ImageCast->SetOutputScalarTypeToShort();
		ImageCast->Update();

		m_pIntensity = ImageCast->GetOutput();

		vtkSmartPointer<vtkImageGradientMagnitude> GradientMagnitude = vtkImageGradientMagnitude::New();

		GradientMagnitude->SetDimensionality(3);
		GradientMagnitude->SetInput(m_pIntensity);
		GradientMagnitude->Update();
		
		m_pGradientMagnitude = GradientMagnitude->GetOutput();
	
		int* pResolution = m_pIntensity->GetExtent();
		
		double* pBounds = m_pIntensity->GetBounds();

		m_VolumeInfo.m_Size.x = 1.0f;//pBounds[1] - pBounds[0];
		m_VolumeInfo.m_Size.y = 1.0f;//pBounds[3] - pBounds[2];
		m_VolumeInfo.m_Size.z = 1.0f;//pBounds[5] - pBounds[4];

		m_VolumeInfo.m_InvSize.x = 1.0f / m_VolumeInfo.m_Size.x;
		m_VolumeInfo.m_InvSize.y = 1.0f / m_VolumeInfo.m_Size.y;
		m_VolumeInfo.m_InvSize.z = 1.0f / m_VolumeInfo.m_Size.z;

		m_VolumeInfo.m_Extent.x	= pResolution[1] - pResolution[0];
		m_VolumeInfo.m_Extent.y	= pResolution[3] - pResolution[2];
		m_VolumeInfo.m_Extent.z	= pResolution[5] - pResolution[4];
		
		m_VolumeInfo.m_InvExtent.x	= m_VolumeInfo.m_Extent.x != 0 ? 1.0f / (float)m_VolumeInfo.m_Extent.x : 0.0f;
		m_VolumeInfo.m_InvExtent.y	= m_VolumeInfo.m_Extent.y != 0 ? 1.0f / (float)m_VolumeInfo.m_Extent.y : 0.0f;
		m_VolumeInfo.m_InvExtent.z	= m_VolumeInfo.m_Extent.z != 0 ? 1.0f / (float)m_VolumeInfo.m_Extent.z : 0.0f;

		cudaExtent Extent;

		Extent.width	= pResolution[1] + 1;
		Extent.height	= pResolution[3] + 1;
		Extent.depth	= pResolution[5] + 1;

		double* pIntensityRange = m_pIntensity->GetScalarRange();
		
		m_VolumeInfo.m_IntensityMin			= (float)pIntensityRange[0];
		m_VolumeInfo.m_IntensityMax			= (float)pIntensityRange[1];
		m_VolumeInfo.m_IntensityRange		= m_VolumeInfo.m_IntensityMax - m_VolumeInfo.m_IntensityMin;
		m_VolumeInfo.m_IntensityInvRange	= 1.0f / m_VolumeInfo.m_IntensityRange;

		double* pSpacing = m_pIntensity->GetSpacing();

		m_VolumeInfo.m_Spacing.x = (float)pSpacing[0];
		m_VolumeInfo.m_Spacing.y = (float)pSpacing[1];
		m_VolumeInfo.m_Spacing.z = (float)pSpacing[2];

		m_VolumeInfo.m_InvSpacing.x = m_VolumeInfo.m_Spacing.x != 0.0f ? 1.0f / m_VolumeInfo.m_Spacing.x : 0.0f;
		m_VolumeInfo.m_InvSpacing.y = m_VolumeInfo.m_Spacing.y != 0.0f ? 1.0f / m_VolumeInfo.m_Spacing.y : 0.0f;
		m_VolumeInfo.m_InvSpacing.z = m_VolumeInfo.m_Spacing.z != 0.0f ? 1.0f / m_VolumeInfo.m_Spacing.z : 0.0f;

		m_VolumeInfo.m_MinAABB.x	= 0.0f;//pBounds[0];
		m_VolumeInfo.m_MinAABB.y	= 0.0f;//pBounds[2];
		m_VolumeInfo.m_MinAABB.z	= 0.0f;//pBounds[4];

		m_VolumeInfo.m_InvMinAABB.x	= m_VolumeInfo.m_MinAABB.x != 0.0f ? 1.0f / m_VolumeInfo.m_MinAABB.x : 0.0f;
		m_VolumeInfo.m_InvMinAABB.y	= m_VolumeInfo.m_MinAABB.y != 0.0f ? 1.0f / m_VolumeInfo.m_MinAABB.y : 0.0f;
		m_VolumeInfo.m_InvMinAABB.z	= m_VolumeInfo.m_MinAABB.z != 0.0f ? 1.0f / m_VolumeInfo.m_MinAABB.z : 0.0f;

		m_VolumeInfo.m_MaxAABB.x	= 1.0f;//pBounds[1];
		m_VolumeInfo.m_MaxAABB.y	= 1.0f;//pBounds[3];
		m_VolumeInfo.m_MaxAABB.z	= 1.0f;//pBounds[5];

		m_VolumeInfo.m_InvMaxAABB.x	= m_VolumeInfo.m_MaxAABB.x != 0.0f ? 1.0f / m_VolumeInfo.m_MaxAABB.x : 0.0f;
		m_VolumeInfo.m_InvMaxAABB.y	= m_VolumeInfo.m_MaxAABB.y != 0.0f ? 1.0f / m_VolumeInfo.m_MaxAABB.y : 0.0f;
		m_VolumeInfo.m_InvMaxAABB.z	= m_VolumeInfo.m_MaxAABB.z != 0.0f ? 1.0f / m_VolumeInfo.m_MaxAABB.z : 0.0f;

		m_VolumeInfo.m_GradientDelta		= 1.0f;
		m_VolumeInfo.m_InvGradientDelta		= 1.0f;
		m_VolumeInfo.m_GradientDeltaX.x		= m_VolumeInfo.m_GradientDelta;
		m_VolumeInfo.m_GradientDeltaX.y		= 0.0f;
		m_VolumeInfo.m_GradientDeltaX.z		= 0.0f;
		m_VolumeInfo.m_GradientDeltaY.x		= 0.0f;
		m_VolumeInfo.m_GradientDeltaY.y		= m_VolumeInfo.m_GradientDelta;
		m_VolumeInfo.m_GradientDeltaY.z		= 0.0f;
		m_VolumeInfo.m_GradientDeltaZ.x		= 0.0f;
		m_VolumeInfo.m_GradientDeltaZ.y		= 0.0f;
		m_VolumeInfo.m_GradientDeltaZ.z		= m_VolumeInfo.m_GradientDelta;

		

		BindIntensityBuffer((short*)m_pGradientMagnitude->GetScalarPointer(), Extent);
		BindGradientMagnitudeBuffer((short*)m_pGradientMagnitude->GetScalarPointer(), Extent);
		CreateExtinctionVolume();

		m_VolumeInfo.m_MacroCellSize.x		= 1.0f / ((float)extinctionSize.width);
		m_VolumeInfo.m_MacroCellSize.y		= 1.0f / ((float)extinctionSize.height);
		m_VolumeInfo.m_MacroCellSize.z		= 1.0f / ((float)extinctionSize.depth);
		*/
    }
}

void vtkErVolumeInfo::Update()
{
	/*
	vtkErVolumeProperty* pErVolumeProperty = dynamic_cast<vtkErVolumeProperty*>(GetVolume()->GetProperty());

	if (pErVolumeProperty)
	{
		m_VolumeInfo.m_DensityScale		= pErVolumeProperty->GetDensityScale();
		m_VolumeInfo.m_StepSize			= pErVolumeProperty->GetStepSizeFactorPrimary() * (m_VolumeInfo.m_MaxAABB.x / m_VolumeInfo.m_Extent.x);
		m_VolumeInfo.m_StepSizeShadow	= m_VolumeInfo.m_StepSize * pErVolumeProperty->GetStepSizeFactorSecondary();
		m_VolumeInfo.m_GradientDelta	= pErVolumeProperty->GetGradientDeltaFactor() * m_VolumeInfo.m_Spacing.x;
		m_VolumeInfo.m_InvGradientDelta	= 1.0f / m_VolumeInfo.m_GradientDelta;
		m_VolumeInfo.m_GradientFactor	= pErVolumeProperty->GetGradientFactor();
		m_VolumeInfo.m_ShadingType		= pErVolumeProperty->GetShadingType();
	}
	else
	{
		m_VolumeInfo.m_DensityScale		= vtkErVolumeProperty::DefaultDensityScale();
		m_VolumeInfo.m_StepSize			= vtkErVolumeProperty::DefaultStepSizeFactorPrimary() * (m_VolumeInfo.m_MaxAABB.x / m_VolumeInfo.m_Extent.x);
		m_VolumeInfo.m_StepSizeShadow	= vtkErVolumeProperty::DefaultStepSizeFactorSecondary() * m_VolumeInfo.m_StepSize;
		m_VolumeInfo.m_GradientDelta	= vtkErVolumeProperty::DefaultGradientDeltaFactor() * m_VolumeInfo.m_Spacing.x;
		m_VolumeInfo.m_InvGradientDelta	= 1.0f / m_VolumeInfo.m_GradientDelta;
		m_VolumeInfo.m_GradientFactor	= vtkErVolumeProperty::DefaultGradientFactor();
		m_VolumeInfo.m_ShadingType		= vtkErVolumeProperty::DefaultShadingType();
	}

	m_VolumeInfo.m_GradientDeltaX.x = m_VolumeInfo.m_GradientDelta;
	m_VolumeInfo.m_GradientDeltaX.y = 0.0f;
	m_VolumeInfo.m_GradientDeltaX.z = 0.0f;

	m_VolumeInfo.m_GradientDeltaY.x = 0.0f;
	m_VolumeInfo.m_GradientDeltaY.y = m_VolumeInfo.m_GradientDelta;
	m_VolumeInfo.m_GradientDeltaY.z = 0.0f;

	m_VolumeInfo.m_GradientDeltaZ.x = 0.0f;
	m_VolumeInfo.m_GradientDeltaZ.y = 0.0f;
	m_VolumeInfo.m_GradientDeltaZ.z = m_VolumeInfo.m_GradientDelta;
	*/
}


void vtkErVolumeInfo::CreateExtinctionVolume()
{
	/*
	vtkDebugMacro("Generating extinction volume");

	

	extinctionSize.width	= ceilf((float)m_VolumeInfo.m_Extent.x / 8.0f);
	extinctionSize.height	= ceilf((float)m_VolumeInfo.m_Extent.y / 8.0f);
	extinctionSize.depth	= ceilf((float)m_VolumeInfo.m_Extent.z / 8.0f);

	vtkSmartPointer<vtkImageData> Vol = vtkImageData::New();

	// first we allocate the data
//	Vol->SetOrigin( origin );
//	Vol->SetSpacing( spacing );
	Vol->SetDimensions(extinctionSize.width, extinctionSize.height, extinctionSize.depth);
	Vol->SetScalarTypeToUnsignedShort(); // the data

	Vol->SetNumberOfScalarComponents(1);
	Vol->AllocateScalars();




//	vtkErrorMacro(<<"Extent" << m_VolumeInfo.m_Extent.x << ", "<< m_VolumeInfo.m_Extent.y << ", "<< m_VolumeInfo.m_Extent.z)

	vtkErVolumeProperty* pProp = dynamic_cast<vtkErVolumeProperty*>(Volume->GetProperty());

	for(int x = 0; x < m_VolumeInfo.m_Extent.x; ++x)
	{
		for(int y = 0; y < m_VolumeInfo.m_Extent.y; ++y)
		{
			for(int z = 0; z < m_VolumeInfo.m_Extent.z; ++z)
			{
				short Opacity = pProp->GetOpacity()->GetValue((short)m_pIntensity->GetScalarPointer(x, y, z)) * 255;

				if ((short)Vol->GetScalarPointer(floorf((float)x / 8.0f), floorf((float)y / 8.0f), floorf((float)z / 8.0f)) < Opacity)
				{
					Vol->SetScalarComponentFromDouble(floorf((float)x / 8.0f), floorf((float)y / 8.0f), floorf((float)z / 8.0f), 0, Opacity);
				}
			}
		}
	}

	BindExtinction((short*)Vol->GetScalarPointer(), extinctionSize);
	*/
}