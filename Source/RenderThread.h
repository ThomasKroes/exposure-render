#pragma once

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

signals:
	void RenderBegin(void);
	void RenderEnd(void);
	void PreRenderFrame(void);
	void PostRenderFrame(void);
	void Resize(void);
	void LoadPreset(const QString& PresetName);
	void StatisticChanged(const QString& Group, const QString& Name, const QString& Value, const QString& Unit = "", const QString& Icon = "");
	
	friend class QRenderThread;
	friend class QFilmWidget;
	friend class QApertureWidget;
	friend class QProjectionWidget;
	friend class QFocusWidget;
};

extern CRenderStatus gRenderStatus;

class QRenderThread : public QThread
{
	Q_OBJECT

public:
	QRenderThread(const QString& FileName = "", QObject* pParent = NULL);
	virtual ~QRenderThread(void);
	QRenderThread(const QRenderThread& Other);
	QRenderThread& operator = (const QRenderThread& Other);

	void run();

	QString			GetFileName(void) const;
	void			SetFileName(const QString& FileName);
	bool			InitializeCuda(void);
	unsigned char*	GetRenderImage(void) const;
	CScene*			GetScene(void);
	bool			Load(QString& FileName);
	void			Close(void);
	void			PauseRendering(const bool& Pause);

private:
	void HandleCudaError(const cudaError_t CudaError);

private:
	QString			m_FileName;
	unsigned char*	m_pRenderImage;

	vtkImageData*	m_pVtkDensityBuffer;
	vtkImageData*	m_pVtkGradientMagnitudeBuffer;

	// CUDA allocations
	CScene*			m_pScene;
	CColorXyz*		m_pDevAccEstXyz;
	CColorXyz*		m_pDevEstFrameXyz;
	CColorXyz*		m_pDevEstFrameBlurXyz;
	unsigned char*	m_pDevEstRgbLdr;
	unsigned char*	m_pDevRgbLdrDisp;

	// Host image buffers
	unsigned char*	m_pImageCanvas;
	unsigned int*	m_pSeeds;

public:
	QMutex			m_Mutex;
	bool			m_Abort;
	CScene			m_Scene;
	bool			m_Pause;

public slots:
	void OnUpdateTransferFunction(void);
	void OnUpdateCamera(void);
	void OnUpdateLighting(void);

signals:
	void RenderBegin(void);
	void RenderEnd(void);
	void MemoryAllocate(void);
	void MemoryFree(void);
	void PreFrame(void);
	void PostFrame(void);
};

// Render thread
extern QRenderThread* gpRenderThread;

CScene* Scene(void);
void StartRenderThread(QString& FileName);
void KillRenderThread(void);