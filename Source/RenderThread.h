#pragma once

#include "Statistics.h"
#include "Scene.h"
#include "Variance.h"

class QRenderThread : public QThread
{
	Q_OBJECT

public:
	QRenderThread(const QString& FileName = "", QObject* pParent = NULL);
	QRenderThread(const QRenderThread& Other);
	virtual ~QRenderThread(void);
	QRenderThread& QRenderThread::operator=(const QRenderThread& Other);

	void run();

	bool			Load(QString& FileName);

	QString			GetFileName(void) const						{	return m_FileName;		}
	void			SetFileName(const QString& FileName)		{	m_FileName = FileName;	}
	CColorRgbaLdr*	GetRenderImage(void) const;
	CScene*			GetScene(void)								{	return &m_Scene;		}
	void			Close(void)									{	m_Abort = true;			}
	void			PauseRendering(const bool& Pause)			{	m_Pause = Pause;		}
	
private:
	QString			m_FileName;
	
	CScene*			m_pDevScene;
	CColorXyz*		m_pDevAccEstXyz;
	CColorXyz*		m_pDevEstXyz;
	CColorXyz*		m_pDevEstFrameXyz;
	CColorXyz*		m_pDevEstFrameBlurXyz;
	CColorRgbaLdr*	m_pDevEstRgbaLdr;
	CColorRgbLdr*	m_pDevRgbLdrDisp;
	unsigned int*	m_pDevSeeds;

	CVariance	m_Variance;
	CVariance*	m_pDevVariance;
	
	CColorRgbaLdr*	m_pRenderImage;
	short*			m_pDensityBuffer;
	short*			m_pGradientMagnitudeBuffer;

public:
	bool			m_Abort;
	CScene			m_Scene;
	bool			m_Pause;

public:
	QList<int>		m_SaveFrames;
	QString			m_SaveBaseName;

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