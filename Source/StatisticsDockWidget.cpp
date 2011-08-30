
#include "StatisticsDockWidget.h"
#include "MainWindow.h"

QStatisticsWidget::QStatisticsWidget(QWidget* pParent) :
	QWidget(pParent),
	m_pMainLayout(NULL),
	m_pPerformanceGroupBox(NULL),
	m_pPerformanceLayout(NULL),
	m_pTracerFPS(NULL),
	m_pVtkFPS(NULL),
	m_pMemoryGroupBox(NULL),
	m_pMemoryLayout(NULL),
	m_pSizeRandomStates(NULL),
	m_pSizeAccEstXyz(NULL),
	m_pSizeEstFrameXyz(NULL),
	m_pSizeEstFrameBlurXyz(NULL),
	m_pSizeEstRgbLdr(NULL),
	m_pMiscellaneousGroupBox(NULL),
	m_pMiscellaneousLayout(NULL),
	m_pNoIterations(NULL),
	m_pVolumeGroupBox(NULL),
	m_pVolumeLayout(NULL),
	m_pFile(NULL),
	m_pBoundingBox(NULL),
	m_pResolutionX(NULL),
	m_pResolutionY(NULL),
	m_pResolutionZ(NULL),
	m_pSpacingX(NULL),
	m_pSpacingY(NULL),
	m_pSpacingZ(NULL),
	m_pScaleX(NULL),
	m_pScaleY(NULL),
	m_pScaleZ(NULL),
	m_pVolumeSize(NULL),
	m_pNoVoxels(NULL),
	m_pDensityRange(NULL)
{
	setEnabled(false);

	m_pMainLayout = new QVBoxLayout(this);
	m_pMainLayout->setAlignment(Qt::AlignTop);

	// Performance
	m_pPerformanceGroupBox = new QGroupBox(this);
	m_pPerformanceGroupBox->setTitle("Performance");
	m_pMainLayout->addWidget(m_pPerformanceGroupBox);

	// Align
	m_pPerformanceLayout = new QGridLayout(m_pPerformanceGroupBox);
	m_pPerformanceLayout->setColSpacing(0, 120);
	m_pPerformanceLayout->setColumnStretch(1, 2);
	m_pPerformanceLayout->setColumnStretch(2, 80);
	m_pPerformanceLayout->setColumnStretch(3, 80);
	m_pPerformanceLayout->setColumnStretch(4, 80);

	// Tracer FPS
	m_pTracerFPS = new QLabel("--", m_pPerformanceGroupBox);
	m_pPerformanceLayout->addWidget(new QLabel("Tracer FPS"), 0, 0);
	m_pPerformanceLayout->addWidget(new QLabel(":"), 0, 1);
	m_pPerformanceLayout->addWidget(m_pTracerFPS, 0, 2);

	// VTK FPS
	m_pVtkFPS = new QLabel("--", m_pPerformanceGroupBox);
	m_pPerformanceLayout->addWidget(new QLabel("VTK FPS"), 1, 0);
	m_pPerformanceLayout->addWidget(new QLabel(":"), 1, 1);
	m_pPerformanceLayout->addWidget(m_pVtkFPS, 1, 2);
	
	// Memory
	m_pMemoryGroupBox = new QGroupBox(this);
	m_pMemoryGroupBox->setTitle("Memory");
	m_pMainLayout->addWidget(m_pMemoryGroupBox);

	// Align
	m_pMemoryLayout = new QGridLayout(m_pMemoryGroupBox);
	m_pMemoryLayout->setColSpacing(0, 120);
	m_pMemoryLayout->setColumnStretch(1, 2);
	m_pMemoryLayout->setColumnStretch(2, 80);
	m_pMemoryLayout->setColumnStretch(3, 80);
	m_pMemoryLayout->setColumnStretch(4, 80);

	// Tracer FPS
	m_pSizeRandomStates = new QLabel("--", m_pMemoryGroupBox);
	m_pMemoryLayout->addWidget(new QLabel("Random States"), 0, 0);
	m_pMemoryLayout->addWidget(new QLabel(":"), 0, 1);
	m_pMemoryLayout->addWidget(m_pSizeRandomStates, 0, 2);

	// ByteSizeAccEstXyz
	m_pSizeAccEstXyz = new QLabel("--", m_pMemoryGroupBox);
	m_pMemoryLayout->addWidget(new QLabel("SizeAccEstXyz"), 1, 0);
	m_pMemoryLayout->addWidget(new QLabel(":"), 1, 1);
	m_pMemoryLayout->addWidget(m_pSizeAccEstXyz, 1, 2);

	// ByteSizeEstFrameXyz
	m_pSizeEstFrameXyz = new QLabel("--", m_pMemoryGroupBox);
	m_pMemoryLayout->addWidget(new QLabel("ByteSizeEstFrameXyz"), 2, 0);
	m_pMemoryLayout->addWidget(new QLabel(":"), 2, 1);
	m_pMemoryLayout->addWidget(m_pSizeEstFrameXyz, 2, 2);

	// EstFrameBlurXyz
	m_pSizeEstFrameBlurXyz = new QLabel("--", m_pMemoryGroupBox);
	m_pMemoryLayout->addWidget(new QLabel("EstFrameBlurXyz"), 3, 0);
	m_pMemoryLayout->addWidget(new QLabel(":"), 3, 1);
	m_pMemoryLayout->addWidget(m_pSizeEstFrameBlurXyz, 3, 2);

	// EstRgbLdr
	m_pSizeEstRgbLdr = new QLabel("--", m_pMemoryGroupBox);
	m_pMemoryLayout->addWidget(new QLabel("EstRgbLdr"), 4, 0);
	m_pMemoryLayout->addWidget(new QLabel(":"), 4, 1);
	m_pMemoryLayout->addWidget(m_pSizeEstRgbLdr, 4, 2);

	// Miscellaneous
	m_pMiscellaneousGroupBox = new QGroupBox(this);
	m_pMiscellaneousGroupBox->setTitle("Miscellaneous");
	m_pMainLayout->addWidget(m_pMiscellaneousGroupBox);

	// Align
	m_pMiscellaneousLayout = new QGridLayout(m_pMiscellaneousGroupBox);
	m_pMiscellaneousLayout->setColSpacing(0, 120);
	m_pMiscellaneousLayout->setColumnStretch(1, 2);
	m_pMiscellaneousLayout->setColumnStretch(2, 80);
	m_pMiscellaneousLayout->setColumnStretch(3, 80);
	m_pMiscellaneousLayout->setColumnStretch(4, 80);
	
	// No. iterations
	m_pNoIterations = new QLabel("--", m_pMiscellaneousGroupBox);
	m_pMiscellaneousLayout->addWidget(new QLabel("No. iterations"), 0, 0);
	m_pMiscellaneousLayout->addWidget(new QLabel(":"), 0, 1);
	m_pMiscellaneousLayout->addWidget(m_pNoIterations, 0, 2);

	// Miscellaneous
	m_pVolumeGroupBox = new QGroupBox(this);
	m_pVolumeGroupBox->setTitle("Volume");
	m_pMainLayout->addWidget(m_pVolumeGroupBox);

	// Align
	m_pVolumeLayout = new QGridLayout(m_pVolumeGroupBox);
	m_pVolumeLayout->setColSpacing(0, 120);
	m_pVolumeLayout->setColumnStretch(1, 2);
	m_pVolumeLayout->setColumnStretch(2, 80);
	m_pVolumeLayout->setColumnStretch(3, 80);
	m_pVolumeLayout->setColumnStretch(4, 80);

	// File
	m_pFile = new QLabel("--", m_pVolumeGroupBox);
	m_pVolumeLayout->addWidget(new QLabel("File"), 0, 0);
	m_pVolumeLayout->addWidget(new QLabel(":"), 0, 1);
	m_pVolumeLayout->addWidget(m_pFile, 0, 2, 1, 3);

	// Bounding box
	m_pBoundingBox = new QLabel("--", m_pVolumeGroupBox);
	m_pVolumeLayout->addWidget(new QLabel("Bounding Box"), 1, 0);
	m_pVolumeLayout->addWidget(new QLabel(":"), 1, 1);
	m_pVolumeLayout->addWidget(m_pBoundingBox, 1, 2, 1, 3);

	// Resolution
	m_pResolutionX = new QLabel("--", m_pVolumeGroupBox);
	m_pResolutionY = new QLabel("--", m_pVolumeGroupBox);
	m_pResolutionZ = new QLabel("--", m_pVolumeGroupBox);
	m_pVolumeLayout->addWidget(new QLabel("Resolution"), 2, 0);
	m_pVolumeLayout->addWidget(new QLabel(":"), 2, 1);
	m_pVolumeLayout->addWidget(m_pResolutionX, 2, 2);
	m_pVolumeLayout->addWidget(m_pResolutionY, 2, 3);
	m_pVolumeLayout->addWidget(m_pResolutionZ, 2, 4);

	// Spacing
	m_pSpacingX = new QLabel("--", m_pVolumeGroupBox);
	m_pSpacingY = new QLabel("--", m_pVolumeGroupBox);
	m_pSpacingZ = new QLabel("--", m_pVolumeGroupBox);
	m_pVolumeLayout->addWidget(new QLabel("Spacing"), 3, 0);
	m_pVolumeLayout->addWidget(new QLabel(":"), 3, 1);
	m_pVolumeLayout->addWidget(m_pSpacingX, 3, 2);
	m_pVolumeLayout->addWidget(m_pSpacingY, 3, 3);
	m_pVolumeLayout->addWidget(m_pSpacingZ, 3, 4);

	// Scale
	m_pScaleX = new QLabel("--", m_pVolumeGroupBox);
	m_pScaleY = new QLabel("--", m_pVolumeGroupBox);
	m_pScaleZ = new QLabel("--", m_pVolumeGroupBox);
	m_pVolumeLayout->addWidget(new QLabel("Scale"), 4, 0);
	m_pVolumeLayout->addWidget(new QLabel(":"), 4, 1);
	m_pVolumeLayout->addWidget(m_pScaleX, 4, 2);
	m_pVolumeLayout->addWidget(m_pScaleY, 4, 3);
	m_pVolumeLayout->addWidget(m_pScaleZ, 4, 4);

	// Volume size
	m_pVolumeSize = new QLabel("--", m_pVolumeGroupBox);
	m_pVolumeLayout->addWidget(new QLabel("Size"), 5, 0);
	m_pVolumeLayout->addWidget(new QLabel(":"), 5, 1);
	m_pVolumeLayout->addWidget(m_pVolumeSize, 5, 2, 1, 3);

	// No. voxels
	m_pNoVoxels = new QLabel("--", m_pVolumeGroupBox);
	m_pVolumeLayout->addWidget(new QLabel("No. voxels"), 6, 0);
	m_pVolumeLayout->addWidget(new QLabel(":"), 6, 1);
	m_pVolumeLayout->addWidget(m_pNoVoxels, 6, 2, 1, 3);

	// Density range
	m_pDensityRange = new QLabel("--", m_pVolumeGroupBox);
	m_pVolumeLayout->addWidget(new QLabel("Density range"), 7, 0);
	m_pVolumeLayout->addWidget(new QLabel(":"), 7, 1);
	m_pVolumeLayout->addWidget(m_pDensityRange, 7, 2, 1, 3);

	// Create, comnfigure and start timer
	connect(&m_Timer, SIGNAL(timeout()), this, SLOT(update()));
	m_Timer.start(10.0);
}

void QStatisticsWidget::Default(void)
{
}

void QStatisticsWidget::update(void)
{
	if (gpRenderThread && gpRenderThread->Loaded())
	{
		m_pTracerFPS->setText(QString::number(gpScene->m_FPS.m_FilteredDuration, 'f', 3));
		m_pFile->setText(QFileInfo(gpRenderThread->FileName()).fileName());
		m_pBoundingBox->setText("[" + QString::number(gpScene->m_BoundingBox.m_MinP.x, 'f', 2) + ", " + QString::number(gpScene->m_BoundingBox.m_MinP.y, 'f', 2) + ", " + QString::number(gpScene->m_BoundingBox.m_MinP.z, 'f', 2) + "]" + " - " + "[" + QString::number(gpScene->m_BoundingBox.m_MaxP.x, 'f', 2) + ", " + QString::number(gpScene->m_BoundingBox.m_MaxP.y, 'f', 2) + ", " + QString::number(gpScene->m_BoundingBox.m_MaxP.z, 'f', 2) + "]");
		m_pNoIterations->setText(QString::number(gpRenderThread->NoIterations()));
		m_pResolutionX->setText(QString::number(gpScene->m_Resolution.m_XYZ.x));
		m_pResolutionY->setText(QString::number(gpScene->m_Resolution.m_XYZ.y));
		m_pResolutionZ->setText(QString::number(gpScene->m_Resolution.m_XYZ.z));
		m_pSpacingX->setText(QString::number(gpScene->m_Spacing.x, 'g', 3));
		m_pSpacingY->setText(QString::number(gpScene->m_Spacing.y, 'g', 3));
		m_pSpacingZ->setText(QString::number(gpScene->m_Spacing.z, 'g', 3));
		m_pScaleX->setText(QString::number(gpScene->m_Scale.x, 'g', 3));
		m_pScaleY->setText(QString::number(gpScene->m_Scale.y, 'g', 3));
		m_pScaleZ->setText(QString::number(gpScene->m_Scale.z, 'g', 3));
		m_pVolumeSize->setText(QString::number(gpScene->m_MemorySize, 'f', 2) + " MB");
		m_pNoVoxels->setText(QString(QString::number(gpScene->m_NoVoxels)));
		m_pDensityRange->setText(QString::number(gpScene->m_IntensityRange.m_Min, 'f', 1) + " - " + QString::number(gpScene->m_IntensityRange.m_Max, 'f', 1));

		if (!isEnabled())
			setEnabled(true);
	}

	else
	{
		if (isEnabled())
			setEnabled(false);
	}

	QWidget::update();
}

QStatisticsDockWidget::QStatisticsDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_StatisticsWidget()
{
	setWindowTitle("Statistics");
	setToolTip("Rendering statistics");

	setWidget(&m_StatisticsWidget);
}