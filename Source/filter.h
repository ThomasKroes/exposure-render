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

#define MAX_GAUSSIAN_FILTER_KERNEL_SIZE		256
#define MAX_BILATERAL_FILTER_KERNEL_SIZE	256

class GaussianFilter
{
public:
	int		KernelRadius;
	float	KernelD[MAX_BILATERAL_FILTER_KERNEL_SIZE];
};

class BilateralFilter
{
public:
	int		KernelRadius;
	float	KernelD[MAX_BILATERAL_FILTER_KERNEL_SIZE];
	float	GaussSimilarity[256];
};

}
