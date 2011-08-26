#pragma once

#include <FidelityRender\Defines.h>
#include <FidelityRender\Enumerations.h>
#include <FidelityRender\Dll.h>

class FIDELITY_RENDER_DLL CBaseException
{
public:
	CBaseException(const char* pName);

	char			m_Name[MAX_CHAR_SIZE];

	virtual void	ShowException(void) = 0;
};

class FIDELITY_RENDER_DLL CGeneralException
{
public:
	CGeneralException(const char* pName);

	virtual void	ShowException(void);
};