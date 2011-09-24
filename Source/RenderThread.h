#pragma once

#include "Statistics.h"
#include "Scene.h"

#include <cutil_inline.h>

class QRenderThread : public QThread
{
	Q_OBJECT

public:
	QRenderThread(const QString& FileName = "", QObject* pParent = NULL);
	virtual ~QRenderThread(void);
	QRenderThread(const QRenderThread& Other);
	QRenderThread& operator = (const QRenderThread& Other);

	void run();

	bool			Load(QString& FileName);

	QString			GetFileName(void) const						{	return m_FileName;		}
	void			SetFileName(const QString& FileName)		{	m_FileName = FileName;	}
	unsigned char*	GetRenderImage(void) const;
	CScene*			GetScene(void)								{	return &m_Scene;		}
	void			Close(void)									{	m_Abort = true;			}
	void			PauseRendering(const bool& Pause)			{	m_Pause = Pause;		}
	
private:
	QString			m_FileName;
	
	CScene*			m_pScene;
	CColorXyz*		m_pDevAccEstXyz;
	CColorXyz*		m_pDevEstFrameXyz;
	CColorXyz*		m_pDevEstFrameBlurXyz;
	CColorRgbaLdr*	m_pDevEstRgbaLdr;
	unsigned char*	m_pDevRgbLdrDisp;
	unsigned int*	m_pDevSeeds;

	unsigned char*	m_pRenderImage;
	float*			m_pDensityBuffer;
	float*			m_pGradientMagnitudeBuffer;
	float*			m_pExtinctionBuffer;
	cudaExtent		m_ExtinctionSize;

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