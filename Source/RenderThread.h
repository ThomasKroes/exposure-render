#pragma once

#include "Variance.h"
#include "CudaFrameBuffers.h"

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
	void			Close(void)									{	m_Abort = true;			}
	void			PauseRendering(const bool& Pause)			{	m_Pause = Pause;		}
	
private:
	QString				m_FileName;
	CCudaFrameBuffers	m_CudaFrameBuffers;
	CScene*				m_pDevScene;
	

	CVariance			m_Variance;
	CVariance*			m_pDevVariance;
	
	CColorRgbaLdr*		m_pRenderImage;
	short*				m_pDensityBuffer;
	short*				m_pGradientMagnitudeBuffer;

public:
	bool			m_Abort;
	bool			m_Pause;
	QMutex			m_Mutex;

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

void StartRenderThread(QString& FileName);
void KillRenderThread(void);