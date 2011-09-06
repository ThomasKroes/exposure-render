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

class CRenderStatus : public QObject
{
	Q_OBJECT

public:

signals:
	void RenderBegin(void);
	void RenderEnd(void);
	void PreRenderFrame(void);
	void PostRenderFrame(void);

	friend class CRenderThread;
};

extern CRenderStatus gRenderStatus;

class CRenderThread : public QThread
{
	Q_OBJECT

public:
	CRenderThread(QObject* pParent = NULL);
	virtual ~CRenderThread(void);

	void run();

	QString			GetFileName(void) const;
	int				GetNoIterations(void) const;
	unsigned char*	GetRenderImage(void) const;
	CScene*			GetScene(void);
	bool			Load(QString& FileName);
	void			Close(void);

private:
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

public:
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
};

// Render thread
extern CRenderThread* gpRenderThread;

CScene* Scene(void);
void StartRenderThread(QString& FileName);
void KillRenderThread(void);