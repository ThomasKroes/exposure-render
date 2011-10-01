#pragma once

class CCudaFrameBuffers
{
public:
	CCudaFrameBuffers(void);
	virtual ~CCudaFrameBuffers(void);
	CCudaFrameBuffers::CCudaFrameBuffers(const CCudaFrameBuffers& Other);
	CCudaFrameBuffers& CCudaFrameBuffers::operator=(const CCudaFrameBuffers& Other);

	void Resize(const Vec2i& Resolution);
	void Reset(void);
	void Free(void);

	Vec2i			m_Resolution;
	CColorXyz*		m_pDevAccEstXyz;
	CColorXyz*		m_pDevEstXyz;
	CColorXyz*		m_pDevEstFrameXyz;
	CColorXyz*		m_pDevEstFrameBlurXyz;
	CColorRgbaLdr*	m_pDevEstRgbaLdr;
	CColorRgbLdr*	m_pDevRgbLdrDisp;
	unsigned int*	m_pDevSeeds;
};