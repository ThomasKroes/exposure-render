/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Defines.cuh"
#include "General.cuh"

namespace ExposureRender
{

DEVICE_NI ColorXYZf EvaluateBitmap(const int& ID, const int& U, const int& V)
{
	if (gTextures.TextureList[ID].Image.pData == NULL)
		return ColorXYZf(0.0f);

	RGBA ColorRGBA = gTextures.TextureList[ID].Image.pData[V * gTextures.TextureList[ID].Image.Size[0] + U];
	ColorXYZf L;
	L.FromRGB(ONE_OVER_255 * (float)ColorRGBA.Data[0], ONE_OVER_255 * (float)ColorRGBA.Data[1], ONE_OVER_255 * (float)ColorRGBA.Data[2]);

	return L;
}

DEVICE_NI ColorXYZf EvaluateTexture(const int& ID, Vec3f& UVW)
{
	ColorXYZf L;

	int id = 0;

	for (int i = 0; i < gTextures.NoTextures; i++)
	{
		if (gTextures.TextureList[i].ID == ID)
			id = i;
	}

	UVW[0] *= gTextures.TextureList[id].Repeat[0];
	UVW[1] *= gTextures.TextureList[id].Repeat[1];
	
	UVW[0] += gTextures.TextureList[id].Offset[0];
	UVW[1] += 1.0f - gTextures.TextureList[id].Offset[1];
	
	UVW[0] = UVW[0] - floorf(UVW[0]);
	UVW[1] = UVW[1] - floorf(UVW[1]);

	UVW[0] = clamp(UVW[0], 0.0f, 1.0f);
	UVW[1] = clamp(UVW[1], 0.0f, 1.0f);

	switch (gTextures.TextureList[id].Type)
	{
		case 0:
		{
			L.FromRGB(gTextures.TextureList[id].Procedural.UniformColor[0], gTextures.TextureList[id].Procedural.UniformColor[1], gTextures.TextureList[id].Procedural.UniformColor[2]);
			break;
		}

		case 1:
		{
			if (gTextures.TextureList[id].Image.pData != NULL)
			{
				const int Size[2] = { gTextures.TextureList[id].Image.Size[0], gTextures.TextureList[id].Image.Size[1] };

				int umin = int(Size[0] * UVW[0]);
				int vmin = int(Size[1] * UVW[1]);
				int umax = int(Size[0] * UVW[0]) + 1;
				int vmax = int(Size[1] * UVW[1]) + 1;
				float ucoef = fabsf(Size[0] * UVW[0] - umin);
				float vcoef = fabsf(Size[1] * UVW[1] - vmin);

				umin = min(max(umin, 0), Size[0] - 1);
				umax = min(max(umax, 0), Size[0] - 1);
				vmin = min(max(vmin, 0), Size[1] - 1);
				vmax = min(max(vmax, 0), Size[1] - 1);
		
				const ColorXYZf Color[4] = 
				{
					EvaluateBitmap(id, umin, vmin),
					EvaluateBitmap(id, umax, vmin),
					EvaluateBitmap(id, umin, vmax),
					EvaluateBitmap(id, umax, vmax)
				};

				L = (1.0f - vcoef) * ((1.0f - ucoef) * Color[0] + ucoef * Color[1]) + vcoef * ((1.0f - ucoef) * Color[2] + ucoef * Color[3]);

				/*
				UVW[0] = clamp(UVW[0], 0.0f, 1.0f);
				UVW[1] = clamp(UVW[1], 0.0f, 1.0f);

				const int ImageUV[2] =
				{
					(int)floorf(UVW[0] * (float)gTextures.TextureList[id].Image.Size[0]),
					(int)floorf(UVW[1] * (float)gTextures.TextureList[id].Image.Size[1])
				};
				
				const int PID = ImageUV[1] * gTextures.TextureList[id].Image.Size[0] + ImageUV[0];
				
				RGBA Color = gTextures.TextureList[id].Image.pData[PID];

				L.FromRGB(ONE_OVER_255 * (float)Color.Data[0], ONE_OVER_255 * (float)Color.Data[1], ONE_OVER_255 * (float)Color.Data[2]);
				*/
			}

			break;
		}
	}

	return L;
}

}