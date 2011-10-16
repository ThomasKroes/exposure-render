/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "Geometry.h"
#include "Flags.h"
#include "Camera.cuh"
#include "Lighting.cuh"

class EXPOSURE_RENDER_DLL CDenoiseParams
{
public:
	bool		m_Enabled;
	float		m_Noise;
	float		m_LerpC;
	float		m_WindowRadius;
	float		m_WindowArea;
	float		m_InvWindowArea;
	float		m_WeightThreshold;
	float		m_LerpThreshold;

public:
	HO CDenoiseParams(void);

	HOD CDenoiseParams& CDenoiseParams::operator=(const CDenoiseParams& Other)
	{
		m_Enabled			= Other.m_Enabled;
		m_Noise				= Other.m_Noise;
		m_LerpC				= Other.m_LerpC;
		m_WindowRadius		= Other.m_WindowRadius;
		m_WindowArea		= Other.m_WindowArea;
		m_InvWindowArea		= Other.m_InvWindowArea;
		m_WeightThreshold	= Other.m_WeightThreshold;
		m_LerpThreshold		= Other.m_LerpThreshold;

		return *this;
	}

	HOD void SetWindowRadius(const float& WindowRadius)
	{
		m_WindowRadius		= WindowRadius;
		m_WindowArea		= (2.0f * m_WindowRadius + 1.0f) * (2.0f * m_WindowRadius + 1.0f);
		m_InvWindowArea		= 1.0f / m_WindowArea;
	}
};

class EXPOSURE_RENDER_DLL CScene
{
public:
	CScene(void);
	CScene(const CScene& Other);
	CScene& operator = (const CScene& Other);

	CCamera				m_Camera;
	CLighting			m_Lighting;
	CResolution3D		m_Resolution;
	CFlags				m_DirtyFlags;
	Vec3f				m_Spacing;
	Vec3f				m_Scale;
	CBoundingBox		m_BoundingBox;
	CTransferFunctions	m_TransferFunctions;
	CRange				m_IntensityRange;
	CRange				m_GradientMagnitudeRange;
	float				m_DensityScale;
	CDenoiseParams		m_DenoiseParams;
	float				m_Variance;
	int					m_ShadingType;
	float				m_StepSizeFactor;
	float				m_StepSizeFactorShadow;
	float				m_GradientDelta;
	float				m_GradientFactor;
	float				m_GradMagMean;

	HOD int GetNoIterations(void) const					{ return m_NoIterations;			}
	HOD void SetNoIterations(const int& NoIterations)	{ m_NoIterations = NoIterations;	}

private:
	int					m_NoIterations;
};

extern CScene gScene;
