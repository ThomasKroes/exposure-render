
#include <QtGui>

#include "LightingDockWidget.h"
#include "RenderThread.h"

CLightingWidget::CLightingWidget(QWidget* pParent) :
	QWidget(pParent),
	m_pListView(NULL),
	m_pLightName(NULL),
	m_pAddLight(NULL),
	m_pRemoveLight(NULL),
	m_pRenameLight(NULL),
	m_pLoadLights(NULL),
	m_pSaveLights(NULL),
	m_pLightNameAction(NULL),
	m_pAddLightAction(NULL),
	m_pRemoveLightAction(NULL),
	m_pRenameLightAction(NULL),
	m_pLoadLightsAction(NULL),
	m_pSaveLightsAction(NULL)
{
	// Create vertical layout
	m_pMainLayout = new QVBoxLayout();
	setLayout(m_pMainLayout);

	// Lights
	m_pLightsGroupBox = new QGroupBox(this);
	m_pLightsGroupBox->setTitle("Lights");
	m_pMainLayout->add(m_pLightsGroupBox);

	// Create presets layout
	QGridLayout* pPresetsLayout = new QGridLayout(m_pLightsGroupBox);

	// Create light list
	m_pListView = new QListView;
	
	pPresetsLayout->addWidget(m_pListView, 0, 0);

	// Layout for buttons
	QVBoxLayout* pPresetButtonsLayout = new QVBoxLayout();
	pPresetButtonsLayout->setAlignment(Qt::AlignTop);

	pPresetsLayout->addLayout(pPresetButtonsLayout, 0, 1);

	// Light name
	m_pLightName = new QLineEdit(this);
	pPresetButtonsLayout->addWidget(m_pLightName);
	
	// Add light
	m_pAddLight = new QPushButton(this);
	m_pAddLight->setText("Add");
	pPresetButtonsLayout->addWidget(m_pAddLight);

	// Remove light
	m_pRemoveLight = new QPushButton(this);
	m_pRemoveLight->setText("Remove");
	pPresetButtonsLayout->addWidget(m_pRemoveLight);

	// Rename light
	m_pRenameLight = new QPushButton(this);
	m_pRenameLight->setText("Rename");
	pPresetButtonsLayout->addWidget(m_pRenameLight);

	// Load lights
	m_pLoadLights = new QPushButton(this);
	m_pLoadLights->setText("Load");
	pPresetButtonsLayout->addWidget(m_pLoadLights);

	// Save lights
	m_pSaveLights = new QPushButton(this);
	m_pSaveLights->setText("Save");
	pPresetButtonsLayout->addWidget(m_pSaveLights);
	
	// Light settings
	m_pLightSettingsGroupBox = new QGroupBox();
	m_pLightSettingsGroupBox->setTitle("Light settings");
	m_pMainLayout->add(m_pLightSettingsGroupBox);

	// Create vertical layout
	QGridLayout* pLightSettingsLayout = new QGridLayout(m_pLightSettingsGroupBox);
	m_pLightSettingsGroupBox->setLayout(pLightSettingsLayout);

	// Theta
	m_pThetaLabel = new QLabel("Theta");
	pLightSettingsLayout->addWidget(m_pThetaLabel, 0, 0);

	m_pThetaSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pThetaSlider->setFocusPolicy(Qt::StrongFocus);
    m_pThetaSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	m_pThetaSlider->setRange(-360, 360);
	pLightSettingsLayout->addWidget(m_pThetaSlider, 0, 1);
	
	m_pThetaSpinBox = new QSpinBox;
    m_pThetaSpinBox->setRange(-360, 360);
	pLightSettingsLayout->addWidget(m_pThetaSpinBox, 0, 2);
	
	connect(m_pThetaSlider, SIGNAL(valueChanged(int)), m_pThetaSpinBox, SLOT(setValue(int)));
	connect(m_pThetaSpinBox, SIGNAL(valueChanged(int)), m_pThetaSlider, SLOT(setValue(int)));
	connect(m_pThetaSlider, SIGNAL(valueChanged(int)), this, SLOT(SetTheta(int)));

	// Phi
	m_pPhiLabel = new QLabel("Phi");
	pLightSettingsLayout->addWidget(m_pPhiLabel, 1, 0);

	m_pPhiSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pPhiSlider->setFocusPolicy(Qt::StrongFocus);
    m_pPhiSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	m_pPhiSlider->setRange(-90, 90);
	pLightSettingsLayout->addWidget(m_pPhiSlider, 1, 1);
	
	m_pPhiSpinBox = new QSpinBox;
    m_pPhiSpinBox->setRange(-90, 90);
	pLightSettingsLayout->addWidget(m_pPhiSpinBox, 1, 2);
	
	connect(m_pPhiSlider, SIGNAL(valueChanged(int)), m_pPhiSpinBox, SLOT(setValue(int)));
	connect(m_pPhiSpinBox, SIGNAL(valueChanged(int)), m_pPhiSlider, SLOT(setValue(int)));
	connect(m_pPhiSlider, SIGNAL(valueChanged(int)), this, SLOT(SetPhi(int)));

	// Distance
	m_pDistanceLabel = new QLabel("Distance");
	pLightSettingsLayout->addWidget(m_pDistanceLabel, 2, 0);

	m_pDistanceSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pDistanceSlider->setFocusPolicy(Qt::StrongFocus);
    m_pDistanceSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	pLightSettingsLayout->addWidget(m_pDistanceSlider, 2, 1);
	
	m_pDistanceSpinBox = new QSpinBox;
    m_pDistanceSpinBox->setRange(-90, 90);
	pLightSettingsLayout->addWidget(m_pDistanceSpinBox, 2, 2);
	
	connect(m_pDistanceSlider, SIGNAL(valueChanged(int)), m_pDistanceSpinBox, SLOT(setValue(int)));
	connect(m_pDistanceSpinBox, SIGNAL(valueChanged(int)), m_pDistanceSlider, SLOT(setValue(int)));
	connect(m_pDistanceSlider, SIGNAL(valueChanged(int)), this, SLOT(SetDistance(int)));

	// Width
	m_pWidthLabel = new QLabel("Width");
	pLightSettingsLayout->addWidget(m_pWidthLabel, 3, 0);

	m_pWidthSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pWidthSlider->setFocusPolicy(Qt::StrongFocus);
    m_pWidthSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	pLightSettingsLayout->addWidget(m_pWidthSlider, 3, 1);
	
	m_pWidthSpinBox = new QSpinBox;
    m_pWidthSpinBox->setRange(0, 100);
	pLightSettingsLayout->addWidget(m_pWidthSpinBox, 3, 2);
	
	connect(m_pWidthSlider, SIGNAL(valueChanged(int)), m_pWidthSpinBox, SLOT(setValue(int)));
	connect(m_pWidthSpinBox, SIGNAL(valueChanged(int)), m_pWidthSlider, SLOT(setValue(int)));

	// Height
	m_pHeightLabel = new QLabel("Height");
	pLightSettingsLayout->addWidget(m_pHeightLabel, 5, 0);

	m_pHeightSlider = new QSlider(Qt::Orientation::Horizontal);
    m_pHeightSlider->setFocusPolicy(Qt::StrongFocus);
    m_pHeightSlider->setTickPosition(QSlider::TickPosition::NoTicks);
	pLightSettingsLayout->addWidget(m_pHeightSlider, 5, 1);
	
	m_pHeightSpinBox = new QSpinBox;
    m_pHeightSpinBox->setRange(0, 100);
	pLightSettingsLayout->addWidget(m_pHeightSpinBox, 5, 2);
	
	m_pLockHeightCheckBox = new QCheckBox("Lock", this);
	pLightSettingsLayout->addWidget(m_pLockHeightCheckBox, 5, 3);

	connect(m_pHeightSlider, SIGNAL(valueChanged(int)), m_pHeightSpinBox, SLOT(setValue(int)));
	connect(m_pHeightSpinBox, SIGNAL(valueChanged(int)), m_pHeightSlider, SLOT(setValue(int)));
	connect(m_pLockHeightCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockHeight(int)));

	// Intensity
	m_pIntensityLabel = new QLabel("Intensity");
	pLightSettingsLayout->addWidget(m_pIntensityLabel, 6, 0);

	m_pIntensitySlider = new QSlider(Qt::Orientation::Horizontal);
    m_pIntensitySlider->setFocusPolicy(Qt::StrongFocus);
    m_pIntensitySlider->setTickPosition(QSlider::TickPosition::NoTicks);
	pLightSettingsLayout->addWidget(m_pIntensitySlider, 6, 1);
	
	m_pIntensitySpinBox = new QSpinBox;
    m_pIntensitySpinBox->setRange(0, 100);
	pLightSettingsLayout->addWidget(m_pIntensitySpinBox, 6, 2);
	
	connect(m_pIntensitySlider, SIGNAL(valueChanged(int)), m_pIntensitySpinBox, SLOT(setValue(int)));
	connect(m_pIntensitySpinBox, SIGNAL(valueChanged(int)), m_pIntensitySlider, SLOT(setValue(int)));

	/*
	// Environment
	QGroupBox* pEnvironmentGroupBox = new QGroupBox();
	pEnvironmentGroupBox->setTitle("Environment");
	pEnvironmentGroupBox->setCheckable(true);
	pVerticalLayout->add(pEnvironmentGroupBox);

	// Create grid layout
	QGridLayout* pGridLayout = new QGridLayout(pEnvironmentGroupBox);

	pGridLayout->addWidget(new QLabel("Texture"));

	pEnvironmentGroupBox->setLayout(pGridLayout);
	*/

	CreateActions();
}

void CLightingWidget::CreateActions(void)
{
	// Light name
	m_pLightNameAction = new QAction(tr("Edit light name"), this);
    m_pLightNameAction->setShortcuts(QKeySequence::Quit);
    m_pLightNameAction->setStatusTip(tr("Change the name of the light"));
    connect(m_pLightNameAction, SIGNAL(triggered()), this, SLOT(AddLight()));

	// Add light
	m_pAddLightAction = new QAction(tr("Add light"), this);
    m_pAddLightAction->setShortcuts(QKeySequence::Quit);
    m_pAddLightAction->setStatusTip(tr("Add a new light with the specified name"));
    connect(m_pAddLightAction, SIGNAL(triggered()), this, SLOT(AddLight()));
}

void CLightingWidget::AddLight(void)
{
}

void CLightingWidget::LockHeight(const int& State)
{
	m_pHeightLabel->setEnabled(!State);
	m_pHeightSlider->setEnabled(!State);
	m_pHeightSpinBox->setEnabled(!State);

	if (State)
	{
		connect(m_pWidthSlider, SIGNAL(valueChanged(int)), m_pHeightSlider, SLOT(setValue(int)));
		connect(m_pWidthSpinBox, SIGNAL(valueChanged(int)), m_pHeightSpinBox, SLOT(setValue(int)));

		m_pHeightSlider->setValue(m_pWidthSlider->value());
	}
	else
	{
		disconnect(m_pWidthSlider, SIGNAL(valueChanged(int)), m_pHeightSlider, SLOT(setValue(int)));
		disconnect(m_pWidthSpinBox, SIGNAL(valueChanged(int)), m_pHeightSpinBox, SLOT(setValue(int)));
	}
}

void CLightingWidget::SetTheta(const int& Theta)
{
	if (!gpScene)
		return;

	gpScene->m_Light.m_Theta = (float)Theta / RAD_F;

	// Flag the lights as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(LightsDirty);
}

void CLightingWidget::SetPhi(const int& Phi)
{
	if (!gpScene)
		return;

	gpScene->m_Light.m_Phi = (float)Phi / RAD_F;
	
	// Flag the lights as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(LightsDirty);
}

void CLightingWidget::SetDistance(const int& Distance)
{
	if (!gpScene)
		return;

	gpScene->m_Light.m_Distance	= 0.1f * (float)Distance;

	// Flag the lights as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(LightsDirty);
}

void CLightingWidget::SetWidth(const int& Width)
{
	if (!gpScene)
		return;

//	gpScene->m_Light.m_Width = 0.1f * (float)Width;

	// Flag the lights as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(LightsDirty);
}

void CLightingWidget::SetHeight(const int& Height)
{
	if (!gpScene)
		return;

//	gpScene->m_Light.m_Height = 0.1f * (float)Height;

	// Flag the lights as dirty, this will restart the rendering
	gpScene->m_DirtyFlags.SetFlag(LightsDirty);
}

CLightingDockWidget::CLightingDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_pLightingWidget(NULL)
{
	setWindowTitle("Lighting");
	setToolTip("Lighting configuration");

	m_pLightingWidget = new CLightingWidget(this);

	setWidget(m_pLightingWidget);
}