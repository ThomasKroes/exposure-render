
#include "CameraDockWidget.h"
#include "RenderThread.h"
#include "MainWindow.h"

CFilmWidget::CFilmWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pGridLayout(NULL),
	m_pFilmWidthLabel(NULL),
	m_pFilmWidthSlider(NULL),
	m_pFilmWidthSpinBox(NULL),
	m_pFilmHeightLabel(NULL),
	m_pFilmHeightSlider(NULL),
	m_pFilmHeightSpinBox(NULL),
	m_pLockFilmHeightCheckBox(NULL)
{
	setTitle("Film");
	setToolTip("Film properties");

	// Create grid layout
	m_pGridLayout = new QGridLayout();
	m_pGridLayout->setColSpacing(0, 70);
	setLayout(m_pGridLayout);

	// Film width
	m_pFilmWidthLabel = new QLabel("Film width");
	m_pGridLayout->addWidget(m_pFilmWidthLabel, 0, 0);

	m_pFilmWidthSlider = new QSlider(Qt::Orientation::Horizontal);
	m_pFilmWidthSlider->setRange(4, 1024);
//	m_pFilmWidthSlider->setValue(gpScene->m_Camera.m_Film.m_Resolution.Width());
    
	m_pGridLayout->addWidget(m_pFilmWidthSlider, 0, 1);
	
	m_pFilmWidthSpinBox = new QSpinBox;
    m_pFilmWidthSpinBox->setRange(4, 1024);
	m_pGridLayout->addWidget(m_pFilmWidthSpinBox, 0, 2);
	
	connect(m_pFilmWidthSlider, SIGNAL(valueChanged(int)), m_pFilmWidthSpinBox, SLOT(setValue(int)));
	connect(m_pFilmWidthSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFilmWidth(int)));
	connect(m_pFilmWidthSpinBox, SIGNAL(valueChanged(int)), m_pFilmWidthSlider, SLOT(setValue(int)));

	// Film height
	m_pFilmHeightLabel = new QLabel("Film height");
	m_pGridLayout->addWidget(m_pFilmHeightLabel, 2, 0);

	m_pFilmHeightSlider = new QSlider(Qt::Orientation::Horizontal);
	m_pFilmHeightSlider->setRange(4, 1024);
//	m_pFilmHeightSlider->setValue(gpScene->m_Camera.m_Film.m_Resolution.Height());
	m_pGridLayout->addWidget(m_pFilmHeightSlider, 2, 1);
	
	m_pFilmHeightSpinBox = new QSpinBox;
    m_pFilmHeightSpinBox->setRange(0, 1024);
	m_pGridLayout->addWidget(m_pFilmHeightSpinBox, 2, 2);
	
	m_pLockFilmHeightCheckBox = new QCheckBox("Lock", this);

	m_pGridLayout->addWidget(m_pLockFilmHeightCheckBox, 2, 3);

	connect(m_pFilmHeightSlider, SIGNAL(valueChanged(int)), m_pFilmHeightSpinBox, SLOT(setValue(int)));
	connect(m_pFilmHeightSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFilmHeight(int)));
	connect(m_pFilmHeightSpinBox, SIGNAL(valueChanged(int)), m_pFilmHeightSlider, SLOT(setValue(int)));
	connect(m_pLockFilmHeightCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockFilmHeight(int)));
}

void CFilmWidget::LockFilmHeight(const int& Lock)
{
	m_pFilmHeightLabel->setEnabled(!Lock);
	m_pFilmHeightSlider->setEnabled(!Lock);
	m_pFilmHeightSpinBox->setEnabled(!Lock);

	if (Lock)
	{
		connect(m_pFilmWidthSlider, SIGNAL(valueChanged(int)), m_pFilmHeightSlider, SLOT(setValue(int)));
		connect(m_pFilmWidthSpinBox, SIGNAL(valueChanged(int)), m_pFilmHeightSpinBox, SLOT(setValue(int)));

		m_pFilmHeightSlider->setValue(m_pFilmWidthSlider->value());
	}
	else
	{
		disconnect(m_pFilmWidthSlider, SIGNAL(valueChanged(int)), m_pFilmHeightSlider, SLOT(setValue(int)));
		disconnect(m_pFilmWidthSpinBox, SIGNAL(valueChanged(int)), m_pFilmHeightSpinBox, SLOT(setValue(int)));
	}
}

void CFilmWidget::SetFilmWidth(const int& FilmWidth)
{
	if (!gpScene)
		return;

	gpScene->m_Camera.m_Film.m_Resolution.m_XY.x = FilmWidth;

	// Flag the film resolution as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(FilmResolutionDirty);
}

void CFilmWidget::SetFilmHeight(const int& FilmHeight)
{
	if (!gpScene)
		return;

	gpScene->m_Camera.m_Film.m_Resolution.m_XY.y = FilmHeight;

	// Flag the film resolution as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(FilmResolutionDirty);
}

CApertureWidget::CApertureWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pGridLayout(NULL),
	m_pApertureSizeLabel(NULL),
	m_pApertureSizeSlider(NULL),
	m_pApertureSizeSpinBox(NULL)
{
	setTitle("Aperture");
	setToolTip("Aperture properties");

	// Create grid layout
	m_pGridLayout = new QGridLayout();
	m_pGridLayout->setColSpacing(0, 70);
	setLayout(m_pGridLayout);

	// Aperture size
	m_pApertureSizeLabel = new QLabel(this);
	m_pApertureSizeLabel->setText("Size");
	m_pGridLayout->addWidget(m_pApertureSizeLabel, 3, 0);

	m_pApertureSizeSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pApertureSizeSlider->setFocusPolicy(Qt::StrongFocus);
    m_pApertureSizeSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	m_pGridLayout->addWidget(m_pApertureSizeSlider, 3, 1);
	
	m_pApertureSizeSpinBox = new QSpinBox;
    m_pApertureSizeSpinBox->setRange(-100, 100);
	m_pGridLayout->addWidget(m_pApertureSizeSpinBox, 3, 2);
	
	connect(m_pApertureSizeSlider, SIGNAL(valueChanged(int)), m_pApertureSizeSpinBox, SLOT(setValue(int)));
	connect(m_pApertureSizeSlider, SIGNAL(valueChanged(int)), this, SLOT(SetAperture(int)));
	connect(m_pApertureSizeSpinBox, SIGNAL(valueChanged(int)), m_pApertureSizeSlider, SLOT(setValue(int)));
}

void CApertureWidget::SetAperture(const int& Aperture)
{
	if (!gpScene)
		return;

	gpScene->m_Camera.m_Aperture.m_Size = 0.01f * (float)Aperture;

	// Flag the camera as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(CameraDirty);
}

CProjectionWidget::CProjectionWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pGridLayout(NULL),
	m_pFieldOfViewLabel(NULL),
	m_pFieldOfViewSlider(NULL),
	m_pFieldOfViewSpinBox(NULL)
{
	setTitle("Projection");
	setToolTip("Projection properties");

	// Projection layout
	m_pGridLayout = new QGridLayout();
	m_pGridLayout->setColSpacing(0, 70);
	setLayout(m_pGridLayout);

	// Field of view
	m_pFieldOfViewLabel = new QLabel("Field of view");
	m_pGridLayout->addWidget(m_pFieldOfViewLabel, 4, 0);

	m_pFieldOfViewSlider = new QSlider(Qt::Orientation::Horizontal);
	m_pFieldOfViewSlider->setRange(10, 200);
	m_pGridLayout->addWidget(m_pFieldOfViewSlider, 4, 1);
	
	m_pFieldOfViewSpinBox = new QSpinBox;
    m_pFieldOfViewSpinBox->setRange(10, 200);
	m_pGridLayout->addWidget(m_pFieldOfViewSpinBox, 4, 2);
	
	connect(m_pFieldOfViewSlider, SIGNAL(valueChanged(int)), m_pFieldOfViewSpinBox, SLOT(setValue(int)));
	connect(m_pFieldOfViewSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFieldOfView(int)));
	connect(m_pFieldOfViewSpinBox, SIGNAL(valueChanged(int)), m_pFieldOfViewSlider, SLOT(setValue(int)));
}

void CProjectionWidget::SetFieldOfView(const int& FieldOfView)
{
	if (!gpScene)
		return;

	gpScene->m_Camera.m_FovV = FieldOfView;

	// Flag the camera as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(CameraDirty);
}

CFocusWidget::CFocusWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pGridLayout(NULL),
	m_pFocusTypeLabel(NULL),
	m_pFocusTypeComboBox(NULL),
	m_pFocalDistanceLabel(NULL),
	m_pFocalDistanceSlider(NULL),
	m_pFocalDistanceSpinBox(NULL)
{
	setTitle("Focus");
	setToolTip("Focus properties");

	// Focus layout
	m_pGridLayout = new QGridLayout();
	m_pGridLayout->setColSpacing(0, 70);
	setLayout(m_pGridLayout);

	// Focus type
	m_pFocusTypeLabel = new QLabel("Focus type");
	m_pGridLayout->addWidget(m_pFocusTypeLabel, 5, 0);

	m_pFocusTypeComboBox = new QComboBox(this);
	m_pFocusTypeComboBox->addItem("Automatic");
	m_pFocusTypeComboBox->addItem("Pick");
	m_pFocusTypeComboBox->addItem("Manual");
	m_pGridLayout->addWidget(m_pFocusTypeComboBox, 5, 1, 1, 2);
	
	connect(m_pFocusTypeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(SetFocusType(int)));

	// Focal distance
	m_pFocalDistanceLabel = new QLabel("Focal distance");
	m_pFocalDistanceLabel->setEnabled(false);
	m_pGridLayout->addWidget(m_pFocalDistanceLabel, 6, 0);

	m_pFocalDistanceSlider = new QSlider(Qt::Orientation::Horizontal);
	m_pFocalDistanceSlider->setEnabled(false);
    m_pFocalDistanceSlider->setFocusPolicy(Qt::StrongFocus);
    m_pFocalDistanceSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	m_pGridLayout->addWidget(m_pFocalDistanceSlider, 6, 1);
	
	m_pFocalDistanceSpinBox = new QSpinBox;
	m_pFocalDistanceSpinBox->setEnabled(false);
    m_pFocalDistanceSpinBox->setRange(-100, 100);
	m_pGridLayout->addWidget(m_pFocalDistanceSpinBox, 6, 2);
	
	connect(m_pFocalDistanceSlider, SIGNAL(valueChanged(int)), m_pFocalDistanceSpinBox, SLOT(setValue(int)));
	connect(m_pFocalDistanceSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFocalDistance(int)));
	connect(m_pFocalDistanceSpinBox, SIGNAL(valueChanged(int)), m_pFocalDistanceSlider, SLOT(setValue(int)));
}

void CFocusWidget::SetFocusType(const int& FocusType)
{
	if (!gpScene)
		return;

	gpScene->m_Camera.m_Focus.m_Type = (CFocus::EType)FocusType;

	// Flag the camera as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(CameraDirty);
}

void CFocusWidget::SetFocalDistance(const int& FocalDistance)
{
	if (!gpScene)
		return;

	gpScene->m_Camera.m_Focus.m_FocalDistance = FocalDistance;

	// Flag the camera as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(CameraDirty);
}

CCameraWidget::CCameraWidget(QWidget* pParent) :
	QWidget(pParent),
	m_pMainLayout(NULL),
	m_pFilmWidget(NULL),
	m_pApertureWidget(NULL),
	m_pProjectionWidget(NULL),
	m_pFocusWidget(NULL)
{
	// Create vertical layout
	m_pMainLayout = new QVBoxLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	setLayout(m_pMainLayout);

	// Film widget
	m_pFilmWidget = new CFilmWidget(this);
	m_pMainLayout->addWidget(m_pFilmWidget);
	
	// Aperture widget
	m_pApertureWidget = new CApertureWidget(this);
	m_pMainLayout->addWidget(m_pApertureWidget);

	// Projection widget
	m_pProjectionWidget = new CProjectionWidget(this);
	m_pMainLayout->addWidget(m_pProjectionWidget);

	// Focus widget
	m_pFocusWidget = new CFocusWidget(this);
	m_pMainLayout->addWidget(m_pFocusWidget);
}

QCameraDockWidget::QCameraDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_pCameraWidget(NULL)
{
	setWindowTitle("Camera");
	setToolTip("Camera settings");

	m_pCameraWidget = new CCameraWidget(this);

	setWidget(m_pCameraWidget);
}