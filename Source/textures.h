/*
    Exposure Render: An interactive photo-realistic volume rendering framework
    Copyright (C) 2011 Thomas Kroes

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "texture.h"

namespace ExposureRender
{

HOST_DEVICE ColorXYZf EvaluateProcedural(const Procedural& Procedural, const Vec2f& UVW)
{
	switch (Procedural.Type)
	{
		case Enums::Uniform:
			return ColorXYZf(Procedural.UniformColor);

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
					return ColorXYZf(Procedural.CheckerColor1);
				else
					return ColorXYZf(Procedural.CheckerColor2);
			}
			else
			{
				if (UV[1] % 2 == 0)
					return ColorXYZf(Procedural.CheckerColor2);
				else
					return ColorXYZf(Procedural.CheckerColor1);
			}
		}

		case Enums::Gradient:
			return Procedural.Gradient.Evaluate(UVW[1]);
	}

	return ColorXYZf::Black();
}

HOST_DEVICE ColorXYZf EvaluateTexture(const int& ID, const Vec2f& UV)
{
	if (ID < 0)
		return ColorXYZf::Black();

	const Texture& T = gpTextures[ID];

	ColorXYZf L;

	Vec2f TextureUV = UV;

	TextureUV[0] *= T.Repeat[0];
	TextureUV[1] *= T.Repeat[1];
	
	TextureUV[0] += T.Offset[0];
	TextureUV[1] += 1.0f - T.Offset[1];
	
	TextureUV[0] = TextureUV[0] - floorf(TextureUV[0]);
	TextureUV[1] = TextureUV[1] - floorf(TextureUV[1]);

	TextureUV[0] = Clamp(TextureUV[0], 0.0f, 1.0f);
	TextureUV[1] = Clamp(TextureUV[1], 0.0f, 1.0f);

	if (T.Flip[0])
		TextureUV[0] = 1.0f - TextureUV[0];

	if (T.Flip[1])
		TextureUV[1] = 1.0f - TextureUV[1];

	switch (T.Type)
	{
		case Enums::Procedural:
		{
			L = EvaluateProcedural(T.Procedural, TextureUV);
			break;
		}

		case Enums::Bitmap:
		{
			if (T.BitmapID >= 0)
				L = ColorXYZf::FromRGBAuc(gpBitmaps[T.BitmapID].Pixels(TextureUV, true));

			break;
		}
	}

	return T.OutputLevel * L;
}

}
