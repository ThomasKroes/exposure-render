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

#include "ertracer.h"
#include "ervolume.h"
#include "erlight.h"
#include "erobject.h"
#include "erclippingobject.h"
#include "ertexture.h"
#include "erbitmap.h"

namespace ExposureRender
{

EXPOSURE_RENDER_DLL void BindTracer(const ErTracer& Tracer, const bool& Bind = true);
EXPOSURE_RENDER_DLL void BindVolume(const ErVolume& Volume, const bool& Bind = true);
EXPOSURE_RENDER_DLL void BindLight(const ErLight& Light, const bool& Bind = true);
EXPOSURE_RENDER_DLL void BindObject(const ErObject& Object, const bool& Bind = true);
EXPOSURE_RENDER_DLL void BindClippingObject(const ErClippingObject& ClippingObject, const bool& Bind = true);
EXPOSURE_RENDER_DLL void BindTexture(const ErTexture& Texture, const bool& Bind = true);
EXPOSURE_RENDER_DLL void BindBitmap(const ErBitmap& Bitmap, const bool& Bind = true);
EXPOSURE_RENDER_DLL void RenderEstimate(int TracerID);
EXPOSURE_RENDER_DLL void GetEstimate(int TracerID, unsigned char* pData);
EXPOSURE_RENDER_DLL void GetAutoFocusDistance(int TracerID, int FilmU, int FilmV, float& AutoFocusDistance);
EXPOSURE_RENDER_DLL void GetNoIterations(int TracerID, int& NoIterations);

}
