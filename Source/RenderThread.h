#pragma once

#include "Variance.h"

class QFrameBuffer
{
public:
	QFrameBuffer(void);
	QFrameBuffer(const QFrameBuffer& Other);
	QFrameBuffer& QFrameBuffer::operator=(const QFrameBuffer& Other);
	virtual ~QFrameBuffer(void);
	void Set(unsigned char* pPixels, const int& Width, const int& Height);
	unsigned char* GetPixels(void) { return m_pPixels; }
	int GetWidth(void) const { return m_Width; }
	int GetHeight(void) const { return m_Height; }
	int GetNoPixels(void) const { return m_NoPixels; }

	QMutex			m_Mutex;

private :
	unsigned char*	m_pPixels;
	int				m_Width;
	int				m_Height;
	int				m_NoPixels;
};

extern QFrameBuffer gFrameBuffer;

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
	CColorRgbLdr*	GetRenderImage(void) const;
	void			Close(void)									{	m_Abort = true;			}
	void			PauseRendering(const bool& Pause)			{	m_Pause = Pause;		}
	
private:
	QString				m_FileName;
//	CCudaFrameBuffers	m_CudaFrameBuffers;
	CColorRgbLdr*		m_pRenderImage;
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
	void OnRenderPause(const bool& Pause);
};

// Render thread
extern QRenderThread* gpRenderThread;

void StartRenderThread(QString& FileName);
void KillRenderThread(void);

extern QMutex gSceneMutex;
extern int gCurrentDeviceID;