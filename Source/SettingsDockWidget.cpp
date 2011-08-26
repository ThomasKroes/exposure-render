
#include "SettingsDockWidget.h"
#include "RenderThread.h"

CTracerSettingsWidget::CTracerSettingsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pGridLayout(NULL),
	m_pNoBouncesLabel(NULL),
	m_pNoBouncesSlider(NULL),
	m_pNoBouncesSpinBox(NULL),
	m_pPhaseLabel(NULL),
	m_pPhaseSlider(NULL),
	m_pPhaseSpinBox(NULL)
{
	setTitle("Tracer");
	setToolTip("Tracer settings");

	// Create grid layout
	m_pGridLayout = new QGridLayout();
	m_pGridLayout->setColSpacing(0, 70);
	setLayout(m_pGridLayout);

	// Render type
	m_pGridLayout->addWidget(new QLabel("Render type"), 0, 0);

	m_pRenderTypeComboBox = new QComboBox(this);
	m_pRenderTypeComboBox->addItem("Single scattering");
	m_pRenderTypeComboBox->addItem("Multiple scattering");
	m_pRenderTypeComboBox->addItem("MIP");
	m_pGridLayout->addWidget(m_pRenderTypeComboBox, 0, 1, 1, 2);

	// No. bounces
	m_pNoBouncesLabel = new QLabel("No. bounces"); 
	m_pGridLayout->addWidget(m_pNoBouncesLabel, 1, 0);

	m_pNoBouncesSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pNoBouncesSlider->setFocusPolicy(Qt::StrongFocus);
    m_pNoBouncesSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	m_pNoBouncesSlider->setRange(0, 10);
	m_pGridLayout->addWidget(m_pNoBouncesSlider, 1, 1);
	
	m_pNoBouncesSpinBox = new QSpinBox;
    m_pNoBouncesSpinBox->setRange(0, 10);
	m_pGridLayout->addWidget(m_pNoBouncesSpinBox, 1, 2);
	
	connect(m_pNoBouncesSlider, SIGNAL(valueChanged(int)), m_pNoBouncesSpinBox, SLOT(setValue(int)));
	connect(m_pNoBouncesSpinBox, SIGNAL(valueChanged(int)), m_pNoBouncesSlider, SLOT(setValue(int)));
	connect(m_pNoBouncesSlider, SIGNAL(valueChanged(int)), this, SLOT(SetNoBounces(int)));

	// Phase
	m_pPhaseLabel = new QLabel("Phase");
	m_pGridLayout->addWidget(m_pPhaseLabel, 2, 0);

	m_pPhaseSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pPhaseSlider->setFocusPolicy(Qt::StrongFocus);
    m_pPhaseSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	m_pPhaseSlider->setRange(-100, 100);
	m_pGridLayout->addWidget(m_pPhaseSlider, 2, 1);
	
	m_pPhaseSpinBox = new QSpinBox;
    m_pPhaseSpinBox->setRange(-100, 100);
	m_pGridLayout->addWidget(m_pPhaseSpinBox, 2, 2);
	
	connect(m_pPhaseSlider, SIGNAL(valueChanged(int)), m_pPhaseSpinBox, SLOT(setValue(int)));
	connect(m_pPhaseSpinBox, SIGNAL(valueChanged(int)), m_pPhaseSlider, SLOT(setValue(int)));
	connect(m_pPhaseSpinBox, SIGNAL(valueChanged(int)), this, SLOT(SetPhase(int)));
}

void CTracerSettingsWidget::SetNoBounces(const int& NoBounces)
{
	if (gpScene)
	{
		gpScene->m_MaxNoBounces	= NoBounces;
		
		// Flag the render params as dirty, this will restart the rendering
		gpScene->m_DirtyFlags.SetFlag(RenderParamsDirty);
	}
}

void CTracerSettingsWidget::SetPhase(const int& Phase)
{
	if (gpScene)
	{
		gpScene->m_PhaseG = 0.01f * (float)Phase;
		
		// Flag the render params as dirty, this will restart the rendering
		gpScene->m_DirtyFlags.SetFlag(RenderParamsDirty);
	}
}

CKernelSettingsWidget::CKernelSettingsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pGridLayout(NULL),
	m_pKernelWidthLabel(NULL),
	m_pKernelWidthSlider(NULL),
	m_pKernelWidthSpinBox(NULL),
	m_pKernelHeightLabel(NULL),
	m_pKernelHeightSlider(NULL),
	m_pKernelHeightSpinBox(NULL)
{
	setTitle("Kernel");
	setToolTip("Kernel settings");

	// Create grid layout
	m_pGridLayout = new QGridLayout();
	m_pGridLayout->setColSpacing(0, 70);
	setLayout(m_pGridLayout);

	// Kernel width
	m_pKernelWidthLabel = new QLabel("Kernel Width");
	m_pGridLayout->addWidget(m_pKernelWidthLabel, 3, 0);

	m_pKernelWidthSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pKernelWidthSlider->setFocusPolicy(Qt::StrongFocus);
    m_pKernelWidthSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	m_pKernelWidthSlider->setRange(2, 64);
	m_pGridLayout->addWidget(m_pKernelWidthSlider, 3, 1);
	
	m_pKernelWidthSpinBox = new QSpinBox;
    m_pKernelWidthSpinBox->setRange(2, 64);
	m_pGridLayout->addWidget(m_pKernelWidthSpinBox, 3, 2);
	
	connect(m_pKernelWidthSlider, SIGNAL(valueChanged(int)), m_pKernelWidthSpinBox, SLOT(setValue(int)));
	connect(m_pKernelWidthSpinBox, SIGNAL(valueChanged(int)), m_pKernelWidthSlider, SLOT(setValue(int)));
	connect(m_pKernelWidthSpinBox, SIGNAL(valueChanged(int)), this, SLOT(SetKernelWidth(int)));

	// Kernel height
	m_pKernelHeightLabel = new QLabel("Kernel Height");
	m_pGridLayout->addWidget(m_pKernelHeightLabel, 4, 0);

	m_pKernelHeightSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pKernelHeightSlider->setFocusPolicy(Qt::StrongFocus);
    m_pKernelHeightSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	m_pKernelHeightSlider->setRange(2, 64);
	m_pGridLayout->addWidget(m_pKernelHeightSlider, 4, 1);
	
	m_pKernelHeightSpinBox = new QSpinBox;
    m_pKernelHeightSpinBox->setRange(2, 64);
	m_pGridLayout->addWidget(m_pKernelHeightSpinBox, 4, 2);
	
	m_pLockKernelHeightCheckBox = new QCheckBox("Lock", this);

	m_pGridLayout->addWidget(m_pLockKernelHeightCheckBox, 4, 3);

	connect(m_pKernelHeightSlider, SIGNAL(valueChanged(int)), m_pKernelHeightSpinBox, SLOT(setValue(int)));
	connect(m_pKernelHeightSpinBox, SIGNAL(valueChanged(int)), m_pKernelHeightSlider, SLOT(setValue(int)));
	connect(m_pKernelHeightSpinBox, SIGNAL(valueChanged(int)), this, SLOT(SetKernelHeight(int)));
	connect(m_pLockKernelHeightCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockKernelHeight(int)));
}

void CKernelSettingsWidget::SetKernelWidth(const int& KernelWidth)
{
	if (gpScene)
	{
		gpScene->m_KernelSize.x	= KernelWidth;
		
		// Flag the render params as dirty, this will restart the rendering
		gpScene->m_DirtyFlags.SetFlag(RenderParamsDirty);
	}
}

void CKernelSettingsWidget::SetKernelHeight(const int& KernelHeight)
{
	if (gpScene)
	{
		gpScene->m_KernelSize.y	= KernelHeight;
		
		// Flag the render params as dirty, this will restart the rendering
		gpScene->m_DirtyFlags.SetFlag(RenderParamsDirty);
	}
}

void CKernelSettingsWidget::LockKernelHeight(const int& Lock)
{
	m_pKernelHeightLabel->setEnabled(!Lock);
	m_pKernelHeightSlider->setEnabled(!Lock);
	m_pKernelHeightSpinBox->setEnabled(!Lock);

	if (Lock)
	{
		connect(m_pKernelWidthSlider, SIGNAL(valueChanged(int)), m_pKernelHeightSlider, SLOT(setValue(int)));
		connect(m_pKernelWidthSpinBox, SIGNAL(valueChanged(int)), m_pKernelHeightSpinBox, SLOT(setValue(int)));

		m_pKernelHeightSlider->setValue(m_pKernelWidthSlider->value());
	}
	else
	{
		disconnect(m_pKernelWidthSlider, SIGNAL(valueChanged(int)), m_pKernelHeightSlider, SLOT(setValue(int)));
		disconnect(m_pKernelWidthSpinBox, SIGNAL(valueChanged(int)), m_pKernelHeightSpinBox, SLOT(setValue(int)));
	}
}

CSettingsWidget::CSettingsWidget(QWidget* pParent) :
	QWidget(pParent),
	m_pMainLayout(NULL),
	m_pTracerSettingsWidget(NULL),
	m_pKernelSettingsWidget(NULL)
{
	// Create vertical layout
	m_pMainLayout = new QVBoxLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	setLayout(m_pMainLayout);

	// Tracer settings widget
	m_pTracerSettingsWidget = new CTracerSettingsWidget(this);
	m_pMainLayout->addWidget(m_pTracerSettingsWidget);
	
	// Kernel settings widget
	m_pKernelSettingsWidget = new CKernelSettingsWidget(this);
	m_pMainLayout->addWidget(m_pKernelSettingsWidget);
}

CSettingsDockWidget::CSettingsDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_pSettingsWidget(NULL)
{
	setWindowTitle("Settings");
	setToolTip("Volume tracer settings");

	m_pSettingsWidget = new CSettingsWidget(this);
	
	setWidget(m_pSettingsWidget);
};