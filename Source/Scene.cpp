/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "Scene.h"

CScene::CScene(void) :
	m_Camera(),
	m_Lighting(),
	m_Resolution(),
	m_DirtyFlags(),
	m_Spacing(),
	m_Scale(),
	m_BoundingBox(),
	m_TransferFunctions(),
	m_IntensityRange(),
	m_GradientMagnitudeRange(),
	m_DensityScale(5000),
	m_DenoiseParams(),
	m_NoIterations(0),
	m_ShadingType(2),
	m_StepSizeFactor(3.0f),
	m_StepSizeFactorShadow(3.0f),
	m_GradientDelta(4.0f),
	m_GradientFactor(4.0f),
	m_GradMagMean(1.0f)
{
	Vec4<int> Color;

	m_Slicing.m_Slices[0].m_P = Vec3f(0.5f, 0.5f, 0.5f);
	m_Slicing.m_Slices[0].m_N = Vec3f(0.5f, 0.5f, 0.5f);
	m_Slicing.m_NoSlices = 1;
}

CScene::CScene(const CScene& Other)
{
	*this = Other;
}

HOD CScene& CScene::operator=(const CScene& Other)
{
	m_Camera					= Other.m_Camera;
	m_Lighting					= Other.m_Lighting;
	m_Resolution				= Other.m_Resolution;
	m_DirtyFlags				= Other.m_DirtyFlags;
	m_Spacing					= Other.m_Spacing;
	m_Scale						= Other.m_Scale;
	m_BoundingBox				= Other.m_BoundingBox;
	m_TransferFunctions			= Other.m_TransferFunctions;
	m_IntensityRange			= Other.m_IntensityRange;
	m_DensityScale				= Other.m_DensityScale;
	m_DenoiseParams				= Other.m_DenoiseParams;
	m_NoIterations				= Other.m_NoIterations;
	m_ShadingType				= Other.m_ShadingType;
	m_StepSizeFactor			= Other.m_StepSizeFactor;
	m_StepSizeFactorShadow		= Other.m_StepSizeFactorShadow;
	m_GradientDelta				= Other.m_GradientDelta;
	m_GradientMagnitudeRange	= Other.m_GradientMagnitudeRange;
	m_GradientFactor			= Other.m_GradientFactor;
	m_GradMagMean				= Other.m_GradMagMean;

	return *this;
}

HO CDenoiseParams::CDenoiseParams(void)
{
	/*
	m_Enabled			= true;
	m_Noise				= 1.0f / (0.32f * 0.32f);
	m_LerpC				= 0.2f;
	m_WindowRadius		= 2.0f;
	m_WindowArea		= (2.0f * m_WindowRadius + 1.0f) * (2.0f * m_WindowRadius + 1.0f);
	m_InvWindowArea		= 1.0f / m_WindowArea;
	m_WeightThreshold	= 0.02f;
	m_LerpThreshold		= 0.79f;
	*/


	m_Enabled			= true;
	m_Noise				= 0.05f;//0.32f * 0.32f;// / (0.1f * 0.1f);
	m_LerpC				= 0.01f;
	m_WindowRadius		= 6.0f;
	m_WindowArea		= (2.0f * m_WindowRadius + 1.0f) * (2.0f * m_WindowRadius + 1.0f);
	m_InvWindowArea		= 1.0f / m_WindowArea;
	m_WeightThreshold	= 0.1f;
	m_LerpThreshold		= 0.0f;
	/**/
}

CScene gScene;