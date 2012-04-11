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

#include "Defines.cuh"
#include "General.cuh"
#include "SharedResources.cuh"

namespace ExposureRender
{

#define NO_COLOR_COMPONENTS 4
#define MAX_NO_TEXTURES		64

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

struct Texture : public ErTexture
{
	Texture()
	{
		printf("Texture()\n");
	}

	~Texture()
	{
		printf("~Texture()\n");
	}

	HOST Texture& Texture::operator = (const ErTexture& Other)
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

		if (this->Image.Dirty)
		{
			if (this->Image.pData)
				CUDA::Free(this->Image.pData);

			if (this->Image.pData)
			{
				const int NoPixels = this->Image.Size[0] * this->Image.Size[1];
			
				CUDA::Allocate(this->Image.pData, NoPixels);
				CUDA::MemCopyHostToDevice(Other.Image.pData, this->Image.pData, NoPixels);
			}
		} 

		return *this;
	}

	DEVICE_NI ColorXYZf EvaluateBitmap(const int& U, const int& V)
	{
		if (Image.pData == NULL)
			return ColorXYZf(0.0f);

		ErRGBA ColorRGBA = Image.pData[V * Image.Size[0] + U];
		ColorXYZf L;
		L.FromRGB(ONE_OVER_255 * (float)ColorRGBA.Data[0], ONE_OVER_255 * (float)ColorRGBA.Data[1], ONE_OVER_255 * (float)ColorRGBA.Data[2]);

		return L;
	}

	DEVICE_NI ColorXYZf EvaluateProcedural(const Vec2f& UVW)
	{
		ColorXYZf L;

		switch (Procedural.Type)
		{
			case Enums::Uniform:
			{
				L.FromRGB(Procedural.UniformColor[0], Procedural.UniformColor[1], Procedural.UniformColor[2]);
				break;
			}

			case Enums::Checker:
			{
				const int UV[2] =
				{
					(int)(UVW[0] * 2.0f),
					(int)(UVW[1] * 2.0f)
				};

				if (UV[0] % 2 == 0)
				{
					if (UV[1] % 2 == 0)
						L.FromRGB(Procedural.CheckerColor1[0], Procedural.CheckerColor1[1], Procedural.CheckerColor1[2]);
					else
						L.FromRGB(Procedural.CheckerColor2[0], Procedural.CheckerColor2[1], Procedural.CheckerColor2[2]);
				}
				else
				{
					if (UV[1] % 2 == 0)
						L.FromRGB(Procedural.CheckerColor2[0], Procedural.CheckerColor2[1], Procedural.CheckerColor2[2]);
					else
						L.FromRGB(Procedural.CheckerColor1[0], Procedural.CheckerColor1[1], Procedural.CheckerColor1[2]);
				}

				break;
			}

			case Enums::Gradient:
			{
				break;
			}
		}

		return L;
	}

	DEVICE_NI ColorXYZf Evaluate(const Vec2f& UV)
	{
		ColorXYZf L;

		Vec2f TextureUV = UV;

		TextureUV[0] *= Repeat[0];
		TextureUV[1] *= Repeat[1];
		
		TextureUV[0] += Offset[0];
		TextureUV[1] += 1.0f - Offset[1];
		
		TextureUV[0] = TextureUV[0] - floorf(TextureUV[0]);
		TextureUV[1] = TextureUV[1] - floorf(TextureUV[1]);

		TextureUV[0] = clamp(TextureUV[0], 0.0f, 1.0f);
		TextureUV[1] = clamp(TextureUV[1], 0.0f, 1.0f);

		if (Flip[0])
			TextureUV[0] = 1.0f - TextureUV[0];

		if (Flip[1])
			TextureUV[1] = 1.0f - TextureUV[1];

		switch (Type)
		{
			case Enums::Procedural:
			{
				L = EvaluateProcedural(TextureUV);
				break;
			}

			case Enums::Image:
			{
				if (Image.pData != NULL)
				{
					const int Size[2] = { Image.Size[0], Image.Size[1] };

					int umin = int(Size[0] * TextureUV[0]);
					int vmin = int(Size[1] * TextureUV[1]);
					int umax = int(Size[0] * TextureUV[0]) + 1;
					int vmax = int(Size[1] * TextureUV[1]) + 1;
					float ucoef = fabsf(Size[0] * TextureUV[0] - umin);
					float vcoef = fabsf(Size[1] * TextureUV[1] - vmin);

					umin = min(max(umin, 0), Size[0] - 1);
					umax = min(max(umax, 0), Size[0] - 1);
					vmin = min(max(vmin, 0), Size[1] - 1);
					vmax = min(max(vmax, 0), Size[1] - 1);
			
					const ColorXYZf Color[4] = 
					{
						EvaluateBitmap(umin, vmin),
						EvaluateBitmap(umax, vmin),
						EvaluateBitmap(umin, vmax),
						EvaluateBitmap(umax, vmax)
					};

					L = (1.0f - vcoef) * ((1.0f - ucoef) * Color[0] + ucoef * Color[1]) + vcoef * ((1.0f - ucoef) * Color[2] + ucoef * Color[3]);
				}

				break;
			}
		}

		return OutputLevel * L;
	}
};

typedef ResourceList<Texture, MAX_NO_TEXTURES> Textures;

}
