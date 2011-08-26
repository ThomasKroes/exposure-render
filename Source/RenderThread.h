#pragma once

// Qt
#include <QtGui>

// CUDA
#include "cutil_inline.h"
#include "curand_kernel.h"

// VTK
#include <vtkImageData.h>

#include "Scene.h"

extern bool gThreadAlive;

class CRenderThread : public QThread
{
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
	CStatistics				m_Statistics;
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
};