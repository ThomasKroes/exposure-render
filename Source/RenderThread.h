#pragma once

// Qt
#include <QtGui>

// CUDA
#include "cutil_inline.h"
#include "curand_kernel.h"

#include "Statistics.h"
#include "Scene.h"

class vtkImageData;
class CScene;
class CColorXyz;

class CRenderThread : public QThread
{
	Q_OBJECT

public:
	CRenderThread(const QString& FileName, QObject* pParent = 0);
	virtual ~CRenderThread(void);

	void run();
	QString FileName(void) const { return m_FileName; }
	int NoIterations(void) const { return m_N; }
	unsigned char* RenderImage(void) const { return m_pRenderImage; }

	QString					m_FileName;
	int						m_N;
	unsigned char*			m_pRenderImage;

	vtkImageData*			m_pImageDataVolume;
	CScene					m_Scene;

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

	bool m_Abort;

signals:
	void RenderBegin(void);
	void RenderEnd(void);
	void MemoryAllocate(void);
	void MemoryFree(void);
	void PreFrame(void);
	void PostFrame(void);

public slots:
	void OnCloseRenderThread(void);
};

// Render thread
extern CRenderThread* gpRenderThread;

CScene* Scene(void);
void StartRenderThread(const QString& FileName);
void KillRenderThread(void);