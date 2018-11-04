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

namespace ExposureRender
{

namespace Enums
{
	enum MemoryType
	{
		Host,
		Device
	};

	enum MemoryUnit
	{
		KiloByte,
		MegaByte,
		GigaByte
	};

	enum ProceduralType
	{
		Uniform = 0,
		Checker,
		Gradient
	};

	enum TextureType
	{
		Procedural = 0,
		Bitmap
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
		BrdfOnly = 0,
		PhaseFunctionOnly,
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

	enum EmissionUnit
	{
		Power = 0,
		Lux,
		Intensity
	};

	enum ScatterFunction
	{
		Brdf,
		PhaseFunction
	};

	enum ScatterType
	{
		Volume,
		Light,
		Object,
		SlicePlane
	};
}

}
