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

#include <stdio.h>
#include <stdarg.h>

namespace ExposureRender
{

HOST inline void DebugLog(const char* format, ...)
{
	/*
	va_list fmtargs;
	char buffer[1024];

	va_start(fmtargs,format);
	vsnprintf(buffer,sizeof(buffer)-1,format,fmtargs);
	va_end(fmtargs);

	printf("%s\n", buffer);
	*/
}

}
