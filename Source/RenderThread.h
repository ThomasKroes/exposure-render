#pragma once

// Qt
#include <QtGui>

// CUDA
#include "cutil_inline.h"
#include "curand_kernel.h"

#include "Statistics.h"

class vtkImageData;
class CScene;
class CColorXyz;

extern bool gThreadAlive;

class CRenderThread : public QThread
{
	Q_OBJECT

public:
	CRenderThread(const QString& FileName, QObject* pParent = 0);
	virtual ~CRenderThread(void);

	void run();
	bool Loaded(void) const { return m_Loaded; }
	QString FileName(void) const { return m_FileName; }
	int NoIterations(void) const { return m_N; }
	unsigned char* RenderImage(void) const { return m_pRenderImage; }

	QString					m_FileName;
	bool					m_Loaded;
	QMutex					m_Mutex;
    QWaitCondition			m_Condition;
	int						m_N;
	unsigned char*			m_pRenderImage;

	vtkImageData*			m_pImageDataVolume;

	// CUDA allocations
	CScene*					m_pDevScene;
	curandStateXORWOW_t*	m_pDevRandomStates;
	CColorXyz*				m_pDevAccEstXyz;
	CColorXyz*				m_pDevEstFrameXyz;
	CColorXyz*				m_pDevEstFrameBlurXyz;
	unsigned char*			m_pDevEstRgbLdr;

	// Host image buffers
	unsigned char*			m_pImageCanvas;

	int	m_SizeVolume;
	int m_SizeHdrAccumulationBuffer;
	int m_SizeHdrFrameBuffer;
	int m_SizeHdrBlurFrameBuffer;
	int m_SizeLdrFrameBuffer;

signals:
	void RenderBegin(void);
	void RenderEnd(void);
	void MemoryAllocate(void);
	void MemoryFree(void);
	void PreFrame(void);
	void PostFrame(void);
};