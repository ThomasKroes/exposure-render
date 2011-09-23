#pragma once

#include "Statistics.h"
#include "Scene.h"

class vtkImageData;
class CScene;
class CColorXyz;

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
	unsigned char*	GetRenderImage(void) const;
	CScene*			GetScene(void);
	bool			Load(QString& FileName);
	void			Close(void);
	void			PauseRendering(const bool& Pause);

private:
	QString			m_FileName;
	
	CScene*			m_pScene;
	CColorXyz*		m_pDevAccEstXyz;
	CColorXyz*		m_pDevEstFrameXyz;
	CColorXyz*		m_pDevEstFrameBlurXyz;
	unsigned char*	m_pDevEstRgbLdr;
	unsigned char*	m_pDevRgbLdrDisp;

	unsigned char*	m_pRenderImage;
	unsigned int*	m_pSeeds;
	vtkImageData*	m_pVtkDensityBuffer;
	vtkImageData*	m_pVtkGradientMagnitudeBuffer;

public:
	QMutex			m_Mutex;
	bool			m_Abort;
	CScene			m_Scene;
	bool			m_Pause;

public slots:
	void OnUpdateTransferFunction(void);
	void OnUpdateCamera(void);
	void OnUpdateLighting(void);
};

// Render thread
extern QRenderThread* gpRenderThread;

CScene* Scene(void);
void StartRenderThread(QString& FileName);
void KillRenderThread(void);