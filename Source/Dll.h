#pragma once

#ifdef _EXPORTING
	#define FIDELITY_RENDER_DLL    __declspec(dllexport)
#else
	#define FIDELITY_RENDER_DLL    __declspec(dllimport)
#endif