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

	Vec2i				m_Resolution;
	cudaArray*			m_pDevEstXyz;
	cudaArray*			m_pDevEstFrameXyz;
	cudaArray*			m_pDevEstFrameBlurXyz;
	cudaArray*			m_pDevEstRgbaLdr;
	cudaArray*			m_pRunningSpecularBloom;
	CColorRgbLdr*		m_pDevRgbLdrDisp;
	int*				m_pDevSeeds;
	int*				m_pNoEstimates;
};