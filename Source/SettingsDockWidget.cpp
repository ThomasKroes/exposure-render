
#include "SettingsDockWidget.h"
#include "Scene.h"

CTracerSettingsWidget::CTracerSettingsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_NoBouncesSlider(),
	m_NoBouncesSpinBox(),
	m_ScatteringLayout(),
	m_PhaseSlider()
{
	setTitle("Tracer");
	setTitle("Tracer Properties");
	
	// Create grid layout
	m_MainLayout.setColSpacing(0, 70);
	setLayout(&m_MainLayout);

	// Render type
	m_MainLayout.addWidget(new QLabel("Render type"), 0, 0);

	m_RenderTypeComboBox.addItem("Single scattering");
	m_RenderTypeComboBox.addItem("Multiple scattering");
	m_RenderTypeComboBox.addItem("MIP");
	m_MainLayout.addWidget(&m_RenderTypeComboBox, 0, 1, 1, 2);

	// No. bounces
	m_MainLayout.addWidget(new QLabel("No. bounces"), 1, 0);

	m_NoBouncesSlider.setOrientation(Qt::Orientation::Horizontal);
    m_NoBouncesSlider.setFocusPolicy(Qt::StrongFocus);
    m_NoBouncesSlider.setTickPosition(QSlider::TickPosition::NoTicks);
	m_NoBouncesSlider.setRange(0, 10);
	m_MainLayout.addWidget(&m_NoBouncesSlider, 1, 1);
	
    m_NoBouncesSpinBox.setRange(0, 10);
	m_MainLayout.addWidget(&m_NoBouncesSpinBox, 1, 2);
	
	connect(&m_NoBouncesSlider, SIGNAL(valueChanged(int)), &m_NoBouncesSpinBox, SLOT(setValue(int)));
	connect(&m_NoBouncesSpinBox, SIGNAL(valueChanged(int)), &m_NoBouncesSlider, SLOT(setValue(int)));
	connect(&m_NoBouncesSlider, SIGNAL(valueChanged(int)), this, SLOT(SetNoBounces(int)));

	// Phase
	m_MainLayout.addWidget(new QLabel("Scattering"), 2, 0);

	// Create scattering layout
	m_MainLayout.addLayout(&m_ScatteringLayout, 2, 1, 1, 2);

	m_ScatteringLayout.addWidget(new QLabel("Backward"), 0, 0);

	m_PhaseSlider.setOrientation(Qt::Orientation::Horizontal);
	m_PhaseSlider.setRange(-100, 100);
	m_PhaseSlider.setToolTip("Move slider to the left to increase <i>backward</i> scattering and right to increase <i>forward</i> scattering");
	m_ScatteringLayout.addWidget(&m_PhaseSlider, 0, 1);
	
	m_ScatteringLayout.addWidget(new QLabel("Forward"), 0, 2);
	
	connect(&m_PhaseSlider, SIGNAL(valueChanged(int)), this, SLOT(SetPhase(int)));
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
	m_MainLayout(),
	m_KernelWidthSlider(),
	m_KernelWidthSpinBox(),
	m_KernelHeightSlider(),
	m_KernelHeightSpinBox()
{
	setTitle("Kernel");
	setToolTip("Kernel settings");

	// Create grid layout
	m_MainLayout.setColSpacing(0, 70);
	setLayout(&m_MainLayout);

	// Kernel width
	m_MainLayout.addWidget(new QLabel("Kernel Width"), 3, 0);

	m_KernelWidthSlider.setOrientation(Qt::Orientation::Horizontal);
    m_KernelWidthSlider.setFocusPolicy(Qt::StrongFocus);
    m_KernelWidthSlider.setTickPosition(QSlider::TickPosition::NoTicks);
	m_KernelWidthSlider.setRange(2, 64);
	m_MainLayout.addWidget(&m_KernelWidthSlider, 3, 1);
	
    m_KernelWidthSpinBox.setRange(2, 64);
	m_MainLayout.addWidget(&m_KernelWidthSpinBox, 3, 2);
	
	connect(&m_KernelWidthSlider, SIGNAL(valueChanged(int)), &m_KernelWidthSpinBox, SLOT(setValue(int)));
	connect(&m_KernelWidthSpinBox, SIGNAL(valueChanged(int)), &m_KernelWidthSlider, SLOT(setValue(int)));
	connect(&m_KernelWidthSpinBox, SIGNAL(valueChanged(int)), this, SLOT(SetKernelWidth(int)));

	// Kernel height
	m_MainLayout.addWidget(new QLabel("Kernel Height"), 4, 0);

	m_KernelHeightSlider.setOrientation(Qt::Orientation::Horizontal);
    m_KernelHeightSlider.setFocusPolicy(Qt::StrongFocus);
    m_KernelHeightSlider.setTickPosition(QSlider::TickPosition::NoTicks);
	m_KernelHeightSlider.setRange(2, 64);
	m_MainLayout.addWidget(&m_KernelHeightSlider, 4, 1);
	
    m_KernelHeightSpinBox.setRange(2, 64);
	m_MainLayout.addWidget(&m_KernelHeightSpinBox, 4, 2);
	
	m_LockKernelHeightCheckBox.setText("Lock");

	m_MainLayout.addWidget(&m_LockKernelHeightCheckBox, 4, 3);

	connect(&m_KernelHeightSlider, SIGNAL(valueChanged(int)), &m_KernelHeightSpinBox, SLOT(setValue(int)));
	connect(&m_KernelHeightSpinBox, SIGNAL(valueChanged(int)), &m_KernelHeightSlider, SLOT(setValue(int)));
	connect(&m_KernelHeightSpinBox, SIGNAL(valueChanged(int)), this, SLOT(SetKernelHeight(int)));
	connect(&m_LockKernelHeightCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockKernelHeight(int)));
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
	m_KernelHeightSlider.setEnabled(!Lock);
	m_KernelHeightSpinBox.setEnabled(!Lock);

	if (Lock)
	{
		connect(&m_KernelWidthSlider, SIGNAL(valueChanged(int)), &m_KernelHeightSlider, SLOT(setValue(int)));
		connect(&m_KernelWidthSpinBox, SIGNAL(valueChanged(int)), &m_KernelHeightSpinBox, SLOT(setValue(int)));

		m_KernelHeightSlider.setValue(m_KernelWidthSlider.value());
	}
	else
	{
		disconnect(&m_KernelWidthSlider, SIGNAL(valueChanged(int)), &m_KernelHeightSlider, SLOT(setValue(int)));
		disconnect(&m_KernelWidthSpinBox, SIGNAL(valueChanged(int)), &m_KernelHeightSpinBox, SLOT(setValue(int)));
	}
}

CSettingsWidget::CSettingsWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_TracerSettingsWidget(),
	m_KernelSettingsWidget()
{
	// Create vertical layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_TracerSettingsWidget);
	m_MainLayout.addWidget(&m_KernelSettingsWidget);
}

QSettingsDockWidget::QSettingsDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_SettingsWidget()
{
	setWindowTitle("Tracer Settings");
	setToolTip("<img src=':/Images/gear.png'><div>Tracer Properties</div>");
	setWindowIcon(QIcon(":/Images/gear.png"));

	setWidget(&m_SettingsWidget);
};