#pragma once

#include <QtGui>

class QStatisticsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QStatisticsWidget(QWidget* pParent = NULL);

private slots:
	void OnUpdateFPS(void);
	void OnRenderBegin(void);
	void OnRenderEnd(void);

private:
	void Default(void);

	QGridLayout		m_MainLayout;
	QTreeWidget		m_Tree;

	/*
	QGroupBox*		m_pPerformanceGroupBox;
	QGridLayout*	m_pPerformanceLayout;
	QLabel*			m_pTracerFPS;
	QLabel*			m_pVtkFPS;
	QGroupBox*		m_pMemoryGroupBox;
	QGridLayout*	m_pMemoryLayout;
	QLabel*			m_pSizeRandomStates;
	QLabel*			m_pSizeAccEstXyz;
	QLabel*			m_pSizeEstFrameXyz;
	QLabel*			m_pSizeEstFrameBlurXyz;
	QLabel*			m_pSizeEstRgbLdr;
	QGroupBox*		m_pMiscellaneousGroupBox;
	QGridLayout*	m_pMiscellaneousLayout;
	QLabel*			m_pNoIterations;
	QGroupBox*		m_pVolumeGroupBox;
	QGridLayout*	m_pVolumeLayout;
	QLabel*			m_pFile;
	QLabel*			m_pBoundingBox;
	QLabel*			m_pResolutionX;
	QLabel*			m_pResolutionY;
	QLabel*			m_pResolutionZ;
	QLabel*			m_pSpacingX;
	QLabel*			m_pSpacingY;
	QLabel*			m_pSpacingZ;
	QLabel*			m_pScaleX;
	QLabel*			m_pScaleY;
	QLabel*			m_pScaleZ;
	QLabel*			m_pVolumeSize;
	QLabel*			m_pNoVoxels;
	QLabel*			m_pDensityRange;
	*/
};

class QStatisticsDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QStatisticsDockWidget(QWidget* pParent = 0);

private:
	QStatisticsWidget	m_StatisticsWidget;
};