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

#define	MAX_CHAR_SIZE				256
#define	MAX_NO_TIMINGS				64
#define MAX_NO_TF_NODES 			256
#define MAX_NO_TRACERS				64
#define MAX_NO_VOLUMES				64
#define MAX_NO_LIGHTS				64
#define MAX_NO_OBJECTS				64
#define MAX_NO_CLIPPING_OBJECTS		64
#define MAX_NO_TEXTURES				64

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

	enum ShadingMode
	{
		BRDF = 0,
		Phase,
		Hybrid,
		Modulation,
		Threshold,
		GradientMagnitude
	};

	enum GradientMode
	{
		ForwardDifferences = 0,
		CentralDifferences,
		Filtered,
	};

	enum ExceptionLevel
	{
		Info = 0,
		Warning,
		Error,
		Fatal
	};
}

struct EXPOSURE_RENDER_DLL ErException
{
	Enums::ExceptionLevel	Level;
	char					Message[MAX_CHAR_SIZE];

	ErException(const Enums::ExceptionLevel& Level, const char* pMessage = "")
	{
		this->Level = Level;
		sprintf_s(this->Message, MAX_CHAR_SIZE, "%s", pMessage);
	}

	ErException& operator = (const ErException& Other)
	{
		this->Level = Other.Level;
		sprintf_s(this->Message, MAX_CHAR_SIZE, "%s", Other.Message);

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL ErKernelTiming
{
	char	Event[MAX_CHAR_SIZE];
	float	Duration;

	ErKernelTiming()
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, "Undefined");
		this->Duration = 0.0f;
	}

	ErKernelTiming(const char* pEvent, const float& Duration)
	{
		sprintf_s(this->Event, MAX_CHAR_SIZE, pEvent);
		this->Duration = Duration;
	}

	ErKernelTiming& operator = (const ErKernelTiming& Other)
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

struct EXPOSURE_RENDER_DLL ErKernelTimings
{
	int				NoTimings;
	ErKernelTiming	Timings[MAX_NO_TIMINGS];
	
	ErKernelTimings& operator = (const ErKernelTimings& Other)
	{
		for (int i = 0; i < MAX_NO_TIMINGS; i++)
			this->Timings[i] = Other.Timings[i];

		this->NoTimings = Other.NoTimings;

		return *this;
	}

	void Add(const ErKernelTiming& ErKernelTiming)
	{
		this->Timings[this->NoTimings] = ErKernelTiming;
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

struct EXPOSURE_RENDER_DLL ErRange
{
	float	Min;
	float	Max;
	float	Extent;
	float	Inv;
	
	ErRange()
	{
		this->Min 		= 0.0f;
		this->Max 		= 0.0f;
		this->Extent	= 0.0f;
		this->Inv		= 0.0f;
	}

	void Set(float Range[2])
	{
		this->Min		= Range[0];
		this->Max		= Range[1];
		this->Extent	= this->Max - this->Min;
		this->Inv		= this->Extent != 0.0f ? 1.0f / this->Extent : 0.0f;
	}

	ErRange& operator = (const ErRange& Other)
	{
		this->Min 		= Other.Min;
		this->Max 		= Other.Max;
		this->Extent	= Other.Extent;
		this->Inv		= Other.Inv;

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL ErMatrix44
{
	float				NN[4][4];

	ErMatrix44()
	{
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				this->NN[i][j] = i == j ? 1.0f : 0.0f;
	}

	ErMatrix44& operator = (const ErMatrix44& Other)
	{
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				this->NN[i][j] = Other.NN[i][j];

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL ErCamera
{
	int					FilmSize[2];
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
	float				Exposure;
	float				Gamma;
	float				FOV;

	ErCamera()
	{
		this->FilmSize[0]		= 0;
		this->FilmSize[1]		= 0;
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
		this->Exposure			= 0.0f;
		this->Gamma				= 0.0f;
		this->FOV				= 0.0f;
	}
};

struct EXPOSURE_RENDER_DLL ErVolume
{
	int					Resolution[3];
	float				Spacing[3];
	unsigned short*		pVoxels;
	bool				NormalizeSize;

	ErVolume()
	{
		this->Resolution[0]			= 0;
		this->Resolution[1]			= 0;
		this->Resolution[2]			= 0;
		this->Spacing[0]			= 0.0f;
		this->Spacing[1]			= 0.0f;
		this->Spacing[2]			= 0.0f;
		this->pVoxels				= NULL;
		this->NormalizeSize			= false;
	}

	ErVolume& operator = (const ErVolume& Other)
	{
		this->Resolution[0]			= Other.Resolution[0];
		this->Resolution[1]			= Other.Resolution[1];
		this->Resolution[2]			= Other.Resolution[2];
		this->Spacing[0]			= Other.Spacing[0];
		this->Spacing[1]			= Other.Spacing[1];
		this->Spacing[2]			= Other.Spacing[2];
		this->pVoxels				= Other.pVoxels;
		this->NormalizeSize			= Other.NormalizeSize;

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL ErPiecewiseLinearFunction
{
	ErRange	NodeRange;
	float	Position[MAX_NO_TF_NODES];
	float	Data[MAX_NO_TF_NODES];
	int		Count;

	ErPiecewiseLinearFunction()
	{
		for (int i = 0; i < MAX_NO_TF_NODES; i++)
		{
			this->Position[i]	= 0.0f;
			this->Data[i]		= 0.0f;
		}

		this->Count = 0;
	}

	ErPiecewiseLinearFunction& operator = (const ErPiecewiseLinearFunction& Other)
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
struct EXPOSURE_RENDER_DLL ErTransferFunction1D
{
	ErPiecewiseLinearFunction		PLF[Size];
	
	ErTransferFunction1D& operator = (const ErTransferFunction1D& Other)
	{	
		for (int i = 0; i < Size; i++)
			this->PLF[i] = Other.PLF[i];
		
		return *this;
	}
};

typedef ErTransferFunction1D<1>	ErScalarTransferFunction1D;
typedef ErTransferFunction1D<3>	ErColorTransferFunction1D;

struct EXPOSURE_RENDER_DLL ErShape
{
	ErMatrix44			TM;
	ErMatrix44			InvTM;
	bool				OneSided;
	Enums::ShapeType	Type;
	float				Size[3];
	float				Area;
	float				InnerRadius;
	float				OuterRadius;

	ErShape()
	{
		this->OneSided		= false;
		this->Size[0]		= 0.0f;
		this->Size[1]		= 0.0f;
		this->Size[2]		= 0.0f;
		this->Area			= 0.0f;
		this->InnerRadius	= 0.0f;
		this->OuterRadius	= 0.0f;
	}
	
	ErShape& operator = (const ErShape& Other)
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

struct EXPOSURE_RENDER_DLL ErLight
{
	bool	Enabled;
	bool	Visible;
	ErShape	Shape;
	int		TextureID;
	float	Multiplier;
	int		Unit;
	
	ErLight()
	{
		this->Enabled		= true;
		this->Visible		= true;
		this->TextureID		= 0;
		this->Multiplier	= 100.0f;
		this->Unit			= 0;
	}
	
	ErLight& operator = (const ErLight& Other)
	{
		this->Enabled		= Other.Enabled;
		this->Visible		= Other.Visible;
		this->Shape			= Other.Shape;
		this->TextureID		= Other.TextureID;
		this->Multiplier	= Other.Multiplier;
		this->Unit			= Other.Unit;

		return *this;
	}
};

#define NO_COLOR_COMPONENTS 4

struct EXPOSURE_RENDER_DLL ErProcedural
{
	Enums::ProceduralType	Type;
	float					UniformColor[3];
	float					CheckerColor1[3];
	float					CheckerColor2[3];
	float					GradientColor1[3];
	float					GradientColor2[3];
	float					GradientColor3[3];

	ErProcedural()
	{
	}

	ErProcedural& operator = (const ErProcedural& Other)
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

struct EXPOSURE_RENDER_DLL ErRGBA
{
	unsigned char Data[NO_COLOR_COMPONENTS];

	ErRGBA& operator = (const ErRGBA& Other)
	{
		for (int i = 0; i < NO_COLOR_COMPONENTS; i++)
			this->Data[i] = Other.Data[i];

		return *this;
	}

	ErRGBA()
	{
		for (int i = 0; i < NO_COLOR_COMPONENTS; i++)
			this->Data[i] = 0;
	}
};

struct EXPOSURE_RENDER_DLL ErImage
{
	ErRGBA*		pData;
	int			Size[2];
	bool		Dirty;

	ErImage& operator = (const ErImage& Other)
	{
//		this->pData			= Other.pData;
		this->Size[0]		= Other.Size[0];
		this->Size[1]		= Other.Size[1];
		this->Dirty			= Other.Dirty;

		return *this;
	}

	ErImage()
	{
		this->pData				= NULL;
		this->Size[0]			= 0;
		this->Size[1]			= 0;
		this->Dirty				= false;
	}
};

struct EXPOSURE_RENDER_DLL ErTexture
{
	Enums::TextureType		Type;
	float					OutputLevel;
	ErImage					Image;
	ErProcedural			Procedural;
	float					Offset[2];
	float					Repeat[2];
	bool					Flip[2];

	ErTexture()
	{
	}

	ErTexture& operator = (const ErTexture& Other)
	{
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

struct EXPOSURE_RENDER_DLL ErObject
{
	bool		Enabled;
	ErShape		Shape;
	int			DiffuseTextureID;
	int			SpecularTextureID;
	int			GlossinessTextureID;
	float		Ior;

	ErObject()
	{
		this->Enabled				= true;
		this->DiffuseTextureID		= -1;
		this->SpecularTextureID		= -1;
		this->GlossinessTextureID	= -1;
		this->Ior					= 0.0f;
	}

	ErObject& operator = (const ErObject& Other)
	{
		this->Enabled				= Other.Enabled;
		this->Shape					= Other.Shape;
		this->DiffuseTextureID		= Other.DiffuseTextureID;
		this->SpecularTextureID		= Other.SpecularTextureID;
		this->GlossinessTextureID	= Other.GlossinessTextureID;
		this->Ior					= Other.Ior;

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL ErClippingObject
{
	bool	Enabled;
	ErShape	Shape;
	bool	Invert;

	ErClippingObject()
	{
		this->Enabled	= true;
		this->Invert	= false;
	}

	ErClippingObject& operator = (const ErClippingObject& Other)
	{
		this->Enabled	= Other.Enabled;
		this->Shape		= Other.Shape;
		this->Invert	= Other.Invert;

		return *this;
	}
};

struct EXPOSURE_RENDER_DLL ErRenderSettings
{
	struct EXPOSURE_RENDER_DLL ErTraversalSettings
	{
		float				StepSize;
		float				StepSizeShadow;
		bool				Shadows;
		float				MaxShadowDistance;

		ErTraversalSettings()
		{
			this->StepSize			= 0.1f;
			this->StepSizeShadow	= 0.1f;
			this->Shadows			= true;
			this->MaxShadowDistance	= 1.0f;
		}
	};

	struct EXPOSURE_RENDER_DLL ErShadingSettings
	{
		int					Type;
		float				DensityScale;
		float				IndexOfReflection;
		bool				OpacityModulated;
		int					GradientComputation;
		float				GradientThreshold;
		float				GradientFactor;

		ErShadingSettings()
		{
			this->Type					= 0;
			this->DensityScale			= 100.0f;
			this->IndexOfReflection		= 5.0f;
			this->OpacityModulated		= true;
			this->GradientComputation	= 1;
			this->GradientThreshold		= 0.5f;
			this->GradientFactor		= 0.5f;
		}
	};

	ErTraversalSettings	Traversal;
	ErShadingSettings		Shading;

	ErRenderSettings()
	{
	}
};

struct EXPOSURE_RENDER_DLL ErFiltering
{
	struct EXPOSURE_RENDER_DLL ErGaussianFilterParameters
	{
		int		KernelRadius;
		float	Sigma;

		ErGaussianFilterParameters()
		{
			this->KernelRadius	= 2;
			this->Sigma			= 1.25f;
		}
	};

	struct EXPOSURE_RENDER_DLL ErBilateralFilterParameters
	{
		float	SigmaD;
		float	SigmaR;

		ErBilateralFilterParameters()
		{
			this->SigmaD	= 5.0f;
			this->SigmaR	= 5.0f;
		}
	};

	ErGaussianFilterParameters	FrameEstimateFilter;
	ErBilateralFilterParameters	PostProcessingFilter;

	ErFiltering()
	{
	}
};

}