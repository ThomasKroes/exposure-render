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

#include <cstdio>

namespace ExposureRender
{

#ifdef _EXPORTING
	#define EXPOSURE_RENDER_DLL    __declspec(dllexport)
#else
	#define EXPOSURE_RENDER_DLL    __declspec(dllimport)
#endif

#define	MAX_NO_TIMINGS	64
#define	MAX_CHAR_SIZE	256

namespace Enums
{
	enum ProceduralType
	{
		Uniform = 0,
		Checker,
		Gradient
	};

	enum TextureType
	{
		Procedural = 0,
		Image
	};

	enum ShapeType
	{
		Plane = 0,
		Disk,
		Ring,
		Box,
		Sphere,
		Cylinder,
		Cone
	};
}

struct EXPOSURE_RENDER_DLL Exception
{
	char	Type[MAX_CHAR_SIZE];
	char	Error[MAX_CHAR_SIZE];
	char	Description[MAX_CHAR_SIZE];
	
	Exception(const char* pType = "", const char* pError = "", const char* pDescription = "")
	{
		sprintf_s(Type, MAX_CHAR_SIZE, "%s", pType);
		sprintf_s(Error, MAX_CHAR_SIZE, "%s", pError);
		sprintf_s(Description, MAX_CHAR_SIZE, "%s", pDescription);
	}

	Exception& operator = (const Exception& Other)
	{
		sprintf_s(Type, MAX_CHAR_SIZE, "%s", Other.Type);
		sprintf_s(Error, MAX_CHAR_SIZE, "%s", Other.Error);
		sprintf_s(Description, MAX_CHAR_SIZE, "%s", Other.Description);

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL KernelTiming
{
	char	Event[MAX_CHAR_SIZE];
	float	Duration;

	KernelTiming()
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, "Undefined");
		this->Duration = 0.0f;
	}

	KernelTiming(const char* pEvent, const float& Duration)
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, pEvent);
		this->Duration = Duration;
	}

	KernelTiming& operator = (const KernelTiming& Other)
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, Other.Event);
		this->Duration = Other.Duration;

		return *this;
	}

	void Init()
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, "");
		this->Duration = 0.0f;
	}
};

struct EXPOSURE_RENDER_DLL KernelTimings
{
	int				NoTimings;
	KernelTiming	Timings[MAX_NO_TIMINGS];
	
	KernelTimings& operator = (const KernelTimings& Other)
	{
		for (int i = 0; i < MAX_NO_TIMINGS; i++)
			this->Timings[i] = Other.Timings[i];

		this->NoTimings = Other.NoTimings;

		return *this;
	}

	void Add(const KernelTiming& KernelTiming)
	{
		this->Timings[this->NoTimings] = KernelTiming;
	}

	void Reset()
	{
		this->NoTimings = 0;
	}

	void Init()
	{
		this->NoTimings = 0;

		for (int i = 0; i < MAX_NO_TIMINGS; i++)
			this->Timings[i].Init();
	}
};

struct EXPOSURE_RENDER_DLL Range
{
	float	Min;
	float	Max;
	float	Extent;
	float	Inv;
	
#ifndef __CUDA_ARCH__
	Range()
	{
		this->Min 		= 0.0f;
		this->Max 		= 0.0f;
		this->Extent	= 0.0f;
		this->Inv		= 0.0f;
	}
#endif

	void Set(float Range[2])
	{
		this->Min		= Range[0];
		this->Max		= Range[1];
		this->Extent	= this->Max - this->Min;
		this->Inv		= this->Extent != 0.0f ? 1.0f / this->Extent : 0.0f;
	}

	Range& operator = (const Range& Other)
	{
		this->Min 		= Other.Min;
		this->Max 		= Other.Max;
		this->Extent	= Other.Extent;
		this->Inv		= Other.Inv;

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL Matrix44
{
	float				NN[4][4];

#ifndef __CUDA_ARCH__
	Matrix44()
	{
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				this->NN[i][j] = i == j ? 1.0f : 0.0f;
	}
#endif

	Matrix44& operator = (const Matrix44& Other)
	{
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				this->NN[i][j] = Other.NN[i][j];

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL Camera
{
	int					FilmWidth;
	int					FilmHeight;
	int					FilmNoPixels;
	float				Pos[3];
	float				Target[3];
	float				Up[3];
	float				N[3];
	float				U[3];
	float				V[3];
	float				FocalDistance;
	float				ApertureSize;
	float				ClipNear;
	float				ClipFar;
	float				Screen[2][2];
	float				InvScreen[2];
	float				Exposure;
	float				InvExposure;
	float				Gamma;
	float				InvGamma;
	float				FOV;

#ifndef __CUDA_ARCH__
	Camera()
	{
		this->FilmWidth			= 0;
		this->FilmHeight		= 0;
		this->FilmNoPixels		= 0;
		this->Pos[0]			= 0.0f;
		this->Pos[1]			= 0.0f;
		this->Pos[2]			= 0.0f;
		this->Target[0]			= 0.0f;
		this->Target[1]			= 0.0f;
		this->Target[2]			= 0.0f;
		this->Up[0]				= 0.0f;
		this->Up[1]				= 0.0f;
		this->Up[2]				= 0.0f;
		this->N[0]				= 0.0f;
		this->N[1]				= 0.0f;
		this->N[2]				= 0.0f;
		this->U[0]				= 0.0f;
		this->U[1]				= 0.0f;
		this->U[2]				= 0.0f;
		this->V[0]				= 0.0f;
		this->V[1]				= 0.0f;
		this->V[2]				= 0.0f;
		this->FocalDistance		= 1.0f;
		this->ApertureSize		= 0.0f;
		this->ClipNear			= 0.0f;
		this->ClipFar			= 0.0f;
		this->Screen[0][0]		= 0.0f;
		this->Screen[0][1]		= 0.0f;
		this->Screen[1][0]		= 0.0f;
		this->Screen[1][1]		= 0.0f;
		this->InvScreen[0]		= 0.0f;
		this->InvScreen[1]		= 0.0f;
		this->Exposure			= 0.0f;
		this->InvExposure		= 0.0f;
		this->Gamma				= 0.0f;
		this->InvGamma			= 0.0f;
		this->FOV				= 0.0f;
	}
#endif
};

#define MAX_NO_TF_NODES 256

struct EXPOSURE_RENDER_DLL PiecewiseLinearFunction
{
	Range	NodeRange;
	float	Position[MAX_NO_TF_NODES];
	float	Data[MAX_NO_TF_NODES];
	int		Count;

	PiecewiseLinearFunction()
	{
		for (int i = 0; i < MAX_NO_TF_NODES; i++)
		{
			this->Position[i]	= 0.0f;
			this->Data[i]		= 0.0f;
		}

		this->Count = 0;
	}

	PiecewiseLinearFunction& operator = (const PiecewiseLinearFunction& Other)
	{
		this->NodeRange = Other.NodeRange;

		for (int i = 0; i < MAX_NO_TF_NODES; i++)
		{
			this->Position[i]	= Other.Position[i];
			this->Data[i]		= Other.Data[i];
		}	
		
		this->Count = Other.Count;

		return *this;
	}
};

template<int Size>
struct EXPOSURE_RENDER_DLL TransferFunction1D
{
	PiecewiseLinearFunction		PLF[Size];
	
	TransferFunction1D()
	{
	}

	TransferFunction1D& operator = (const TransferFunction1D& Other)
	{	
		for (int i = 0; i < Size; i++)
			this->PLF[i] = Other.PLF[i];
		
		return *this;
	}
};

typedef TransferFunction1D<1>	ScalarTransferFunction1D;
typedef TransferFunction1D<3>	ColorTransferFunction1D;

struct EXPOSURE_RENDER_DLL Shape
{
	Matrix44			TM;
	Matrix44			InvTM;
	bool				OneSided;
	Enums::ShapeType	Type;
	float				Size[3];
	float				Area;
	float				InnerRadius;
	float				OuterRadius;

#ifndef __CUDA_ARCH__
	Shape()
	{
		this->OneSided		= false;
		this->Size[0]		= 0.0f;
		this->Size[1]		= 0.0f;
		this->Size[2]		= 0.0f;
		this->Area			= 0.0f;
		this->InnerRadius	= 0.0f;
		this->OuterRadius	= 0.0f;
	}
#endif
	
	Shape& operator = (const Shape& Other)
	{
		this->TM			= Other.TM;
		this->InvTM			= Other.InvTM;		
		this->OneSided		= Other.OneSided;
		this->Type			= Other.Type;		
		this->Size[0]		= Other.Size[0];	
		this->Size[1]		= Other.Size[1];	
		this->Size[2]		= Other.Size[2];	
		this->Area			= Other.Area;		
		this->InnerRadius	= Other.InnerRadius;
		this->OuterRadius	= Other.OuterRadius;

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL Light
{
	int		ID;
	bool	Enabled;
	bool	Visible;
	Shape	Shape;
	int		TextureID;
	float	Multiplier;
	int		Unit;
	
#ifndef __CUDA_ARCH__
	Light()
	{
		this->ID			= 0;
		this->Enabled		= true;
		this->Visible		= true;
		this->TextureID		= 0;
		this->Multiplier	= 100.0f;
		this->Unit			= 0;
	}
#endif
	
	Light& operator = (const Light& Other)
	{
		this->ID			= Other.ID;
		this->Enabled		= Other.Enabled;
		this->Visible		= Other.Visible;
		this->Shape			= Other.Shape;
		this->TextureID		= Other.TextureID;
		this->Multiplier	= Other.Multiplier;
		this->Unit			= Other.Unit;

		return *this;
	}
};

#define MAX_NO_LIGHTS 32

struct EXPOSURE_RENDER_DLL Lights
{
	int		Count;
	Light	List[MAX_NO_LIGHTS];

#ifndef __CUDA_ARCH__
	Lights()
	{
		this->Count	= 0;
	}
#endif

	Lights& operator = (const Lights& Other)
	{
		this->Count = Other.Count;
		
		for (int i = 0; i < MAX_NO_LIGHTS; i++)
			this->List[i] = Other.List[i];

		return *this;
	}
};

#define NO_COLOR_COMPONENTS 4

struct EXPOSURE_RENDER_DLL Procedural
{
	Enums::ProceduralType	Type;
	float					UniformColor[3];
	float					CheckerColor1[3];
	float					CheckerColor2[3];
	float					GradientColor1[3];
	float					GradientColor2[3];
	float					GradientColor3[3];

#ifndef __CUDA_ARCH__
	Procedural()
	{
	}
#endif

	Procedural& operator = (const Procedural& Other)
	{
		this->Type				= Other.Type;
		this->UniformColor[0]	= Other.UniformColor[0];
		this->UniformColor[1]	= Other.UniformColor[1];
		this->UniformColor[2]	= Other.UniformColor[2];
		this->CheckerColor1[0]	= Other.CheckerColor1[0];
		this->CheckerColor1[1]	= Other.CheckerColor1[1];
		this->CheckerColor1[2]	= Other.CheckerColor1[2];
		this->CheckerColor2[0]	= Other.CheckerColor2[0];
		this->CheckerColor2[1]	= Other.CheckerColor2[1];
		this->CheckerColor2[2]	= Other.CheckerColor2[2];
		this->GradientColor1[0]	= Other.GradientColor1[0];
		this->GradientColor1[1]	= Other.GradientColor1[1];
		this->GradientColor1[2]	= Other.GradientColor1[2];
		this->GradientColor2[0]	= Other.GradientColor2[0];
		this->GradientColor2[1]	= Other.GradientColor2[1];
		this->GradientColor2[2]	= Other.GradientColor2[2];
		this->GradientColor3[0]	= Other.GradientColor3[0];
		this->GradientColor3[1]	= Other.GradientColor3[1];
		this->GradientColor3[2]	= Other.GradientColor3[2];

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL RGBA
{
	unsigned char Data[NO_COLOR_COMPONENTS];

	RGBA& operator = (const RGBA& Other)
	{
		for (int i = 0; i < NO_COLOR_COMPONENTS; i++)
			this->Data[i] = Other.Data[i];

		return *this;
	}

#ifndef __CUDA_ARCH__
	RGBA()
	{
		for (int i = 0; i < NO_COLOR_COMPONENTS; i++)
			this->Data[i] = 0;
	}
#endif
};

struct EXPOSURE_RENDER_DLL Image
{
	RGBA*		pData;
	int			Size[2];
	bool		Dirty;

	Image& operator = (const Image& Other)
	{
//		this->pData			= Other.pData;
		this->Size[0]		= Other.Size[0];
		this->Size[1]		= Other.Size[1];
		this->Dirty			= Other.Dirty;

		return *this;
	}

#ifndef __CUDA_ARCH__
	Image()
	{
		this->pData				= NULL;
		this->Size[0]			= 0;
		this->Size[1]			= 0;
		this->Dirty				= false;
	}
#endif
};

struct EXPOSURE_RENDER_DLL Texture
{
	int						ID;
	Enums::TextureType		Type;
	float					OutputLevel;
	Image					Image;
	Procedural				Procedural;
	float					Offset[2];
	float					Repeat[2];
	bool					Flip[2];

#ifndef __CUDA_ARCH__
	Texture()
	{
	}
#endif

	Texture& operator = (const Texture& Other)
	{
		this->ID			= Other.ID;
		this->Type			= Other.Type;
		this->OutputLevel	= Other.OutputLevel;
		this->Image			= Other.Image;
		this->Procedural	= Other.Procedural;
		this->Offset[0]		= Other.Offset[0];
		this->Offset[1]		= Other.Offset[1];
		this->Repeat[0]		= Other.Repeat[0];
		this->Repeat[1]		= Other.Repeat[1];
		this->Flip[0]		= Other.Flip[0];
		this->Flip[1]		= Other.Flip[1];

		return *this;
	}
};

#define MAX_NO_TEXTURES 64

struct EXPOSURE_RENDER_DLL Textures
{
	int			Count;
	Texture		List[MAX_NO_TEXTURES];
	
#ifndef __CUDA_ARCH__
	Textures()
	{
		this->Count = 0;
	}
#endif

	Textures& operator = (const Textures& Other)
	{
		this->Count = Other.Count;
		
		for (int i = 0; i < MAX_NO_TEXTURES; i++)
			this->List[i] = Other.List[i];

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL Object
{
	int			ID;
	bool		Enabled;
	Shape		Shape;
	int			DiffuseTextureID;
	int			SpecularTextureID;
	int			GlossinessTextureID;
	float		Ior;

#ifndef __CUDA_ARCH__
	Object()
	{
		this->ID					= 0;
		this->Enabled				= true;
		this->DiffuseTextureID		= -1;
		this->SpecularTextureID		= -1;
		this->GlossinessTextureID	= -1;
		this->Ior					= 0.0f;
	}
#endif

	Object& operator = (const Object& Other)
	{
		this->ID					= Other.ID;
		this->Enabled				= Other.Enabled;
		this->Shape					= Other.Shape;
		this->DiffuseTextureID		= Other.DiffuseTextureID;
		this->SpecularTextureID		= Other.SpecularTextureID;
		this->GlossinessTextureID	= Other.GlossinessTextureID;
		this->Ior					= Other.Ior;

		return *this;
	}
};

#define MAX_NO_OBJECTS 32

struct EXPOSURE_RENDER_DLL Objects
{
	int			Count;
	Object		List[MAX_NO_OBJECTS];

#ifndef __CUDA_ARCH__
	Objects()
	{
		this->Count = 0;
	}
#endif

	Objects& operator = (const Objects& Other)
	{
		this->Count = Other.Count;
		
		for (int i = 0; i < MAX_NO_OBJECTS; i++)
			this->List[i] = Other.List[i];

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL ClippingObject
{
	int		ID;
	bool	Enabled;
	Shape	Shape;
	bool	Invert;

#ifndef __CUDA_ARCH__
	ClippingObject()
	{
		this->ID		= 0;
		this->Enabled	= true;
		this->Invert	= false;
	}
#endif

	ClippingObject& operator = (const ClippingObject& Other)
	{
		this->ID		= Other.ID;
		this->Enabled	= Other.Enabled;
		this->Shape		= Other.Shape;
		this->Invert	= Other.Invert;

		return *this;
	}
};

#define MAX_NO_CLIPPING_OBJECTS 32

struct EXPOSURE_RENDER_DLL ClippingObjects
{
	int				Count;
	ClippingObject	List[MAX_NO_CLIPPING_OBJECTS];

#ifndef __CUDA_ARCH__
	ClippingObjects()
	{
		this->Count = 0;
	}
#endif

	ClippingObjects& operator = (const ClippingObjects& Other)
	{
		this->Count = Other.Count;
		
		for (int i = 0; i < MAX_NO_CLIPPING_OBJECTS; i++)
			this->List[i] = Other.List[i];

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL RenderSettings
{
	struct EXPOSURE_RENDER_DLL TraversalSettings
	{
		float				StepSize;
		float				StepSizeShadow;
		bool				Shadows;
		float				MaxShadowDistance;

#ifndef __CUDA_ARCH__
		TraversalSettings()
		{
			this->StepSize			= 0.1f;
			this->StepSizeShadow	= 0.1f;
			this->Shadows			= true;
			this->MaxShadowDistance	= 1.0f;
		}
#endif
	};

	struct EXPOSURE_RENDER_DLL ShadingSettings
	{
		int					Type;
		float				DensityScale;
		float				IndexOfReflection;
		bool				OpacityModulated;
		int					GradientComputation;
		float				GradientThreshold;
		float				GradientFactor;

#ifndef __CUDA_ARCH__
		ShadingSettings()
		{
			this->Type					= 0;
			this->DensityScale			= 100.0f;
			this->IndexOfReflection		= 5.0f;
			this->OpacityModulated		= true;
			this->GradientComputation	= 1;
			this->GradientThreshold		= 0.5f;
			this->GradientFactor		= 0.5f;
		}
#endif
	};

	TraversalSettings	Traversal;
	ShadingSettings		Shading;

#ifndef __CUDA_ARCH__
	RenderSettings()
	{
	}
#endif
};

struct EXPOSURE_RENDER_DLL Filtering
{
	struct EXPOSURE_RENDER_DLL GaussianFilterParameters
	{
		int		KernelRadius;
		float	Sigma;

#ifndef __CUDA_ARCH__
		GaussianFilterParameters()
		{
			this->KernelRadius	= 2;
			this->Sigma			= 1.25f;
		}
#endif
	};

	struct EXPOSURE_RENDER_DLL BilateralFilterParameters
	{
		float	SigmaD;
		float	SigmaR;

#ifndef __CUDA_ARCH__
		BilateralFilterParameters()
		{
			this->SigmaD	= 5.0f;
			this->SigmaR	= 5.0f;
		}
#endif
	};

	GaussianFilterParameters	FrameEstimateFilter;
	BilateralFilterParameters	PostProcessingFilter;

#ifndef __CUDA_ARCH__
	Filtering()
	{
	}
#endif
};

}