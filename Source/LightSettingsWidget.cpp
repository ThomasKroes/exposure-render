
#include "LightSettingsWidget.h"
#include "LightsWidget.h"
#include "RenderThread.h"

QLightSettingsWidget::QLightSettingsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_ThetaLabel(),
	m_ThetaSlider(),
	m_ThetaSpinBox(),
	m_PhiLabel(),
	m_PhiSlider(),
	m_PhiSpinBox(),
	m_DistanceLabel(),
	m_DistanceSlider(),
	m_DistanceSpinBox(),
	m_WidthLabel(),
	m_WidthSlider(),
	m_WidthSpinBox(),
	m_HeightLabel(),
	m_HeightSlider(),
	m_HeightSpinBox(),
	m_LockSizeCheckBox(),
	m_ColorLabel(),
	m_ColorButton(),
	m_IntensityLabel(),
	m_IntensitySlider(),
	m_IntensitySpinBox()
{
	// Title, status and tooltip
	setTitle("Light Settings");
	setToolTip("Light Settings");
	setStatusTip("Light Settings");

	// Apply main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	// Theta
	m_ThetaLabel.setText("Longitude");
	m_MainLayout.addWidget(&m_ThetaLabel, 0, 0);

	m_ThetaSlider.setOrientation(Qt::Horizontal);
    m_ThetaSlider.setFocusPolicy(Qt::StrongFocus);
    m_ThetaSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_ThetaSlider.setRange(-360.0, 360.0);
	m_MainLayout.addWidget(&m_ThetaSlider, 0, 1);
	
    m_ThetaSpinBox.setRange(-360.0, 360.0);
	m_ThetaSpinBox.setSuffix(" deg");
	m_MainLayout.addWidget(&m_ThetaSpinBox, 0, 2);
	
	connect(&m_ThetaSlider, SIGNAL(valueChanged(double)), &m_ThetaSpinBox, SLOT(setValue(double)));
	connect(&m_ThetaSpinBox, SIGNAL(valueChanged(double)), &m_ThetaSlider, SLOT(setValue(double)));
	connect(&m_ThetaSlider, SIGNAL(valueChanged(double)), this, SLOT(OnThetaChanged(double)));

	// Phi
	m_PhiLabel.setText("Latitude");
	m_MainLayout.addWidget(&m_PhiLabel, 1, 0);

	m_PhiSlider.setOrientation(Qt::Horizontal);
    m_PhiSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_PhiSlider.setRange(-90.0, 90.0);
	m_MainLayout.addWidget(&m_PhiSlider, 1, 1);
	
    m_PhiSpinBox.setRange(-90.0, 90.0);
	m_PhiSpinBox.setSuffix(" deg");
	m_MainLayout.addWidget(&m_PhiSpinBox, 1, 2);
	
	connect(&m_PhiSlider, SIGNAL(valueChanged(double)), &m_PhiSpinBox, SLOT(setValue(double)));
	connect(&m_PhiSpinBox, SIGNAL(valueChanged(double)), &m_PhiSlider, SLOT(setValue(double)));
	connect(&m_PhiSlider, SIGNAL(valueChanged(double)), this, SLOT(OnPhiChanged(double)));

	// Distance
	m_DistanceLabel.setText("Distance");
	m_MainLayout.addWidget(&m_DistanceLabel, 2, 0);

	m_DistanceSlider.setOrientation(Qt::Horizontal);
    m_DistanceSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_DistanceSlider.setRange(0.0, 10.0);
	m_MainLayout.addWidget(&m_DistanceSlider, 2, 1);
	
    m_DistanceSpinBox.setRange(0.0, 10.0);
	m_MainLayout.addWidget(&m_DistanceSpinBox, 2, 2);
	
	connect(&m_DistanceSlider, SIGNAL(valueChanged(double)), &m_DistanceSpinBox, SLOT(setValue(double)));
	connect(&m_DistanceSpinBox, SIGNAL(valueChanged(double)), &m_DistanceSlider, SLOT(setValue(double)));
	connect(&m_DistanceSlider, SIGNAL(valueChanged(double)), this, SLOT(OnDistanceChanged(double)));

	// Width
	m_WidthLabel.setText("Width");
	m_MainLayout.addWidget(&m_WidthLabel, 3, 0);

	m_WidthSlider.setOrientation(Qt::Horizontal);
    m_WidthSlider.setFocusPolicy(Qt::StrongFocus);
    m_WidthSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_WidthSlider.setRange(0.0, 10.0);
	m_MainLayout.addWidget(&m_WidthSlider, 3, 1);
	
	m_WidthSpinBox.setRange(0.0, 10.0);
	m_MainLayout.addWidget(&m_WidthSpinBox, 3, 2);
	
	connect(&m_WidthSlider, SIGNAL(valueChanged(double)), &m_WidthSpinBox, SLOT(setValue(double)));
	connect(&m_WidthSpinBox, SIGNAL(valueChanged(double)), &m_WidthSlider, SLOT(setValue(double)));
	connect(&m_WidthSlider, SIGNAL(valueChanged(double)), this, SLOT(OnWidthChanged(double)));

	// Height
	m_HeightLabel.setText("Height");
	m_MainLayout.addWidget(&m_HeightLabel, 5, 0);

	m_HeightSlider.setOrientation(Qt::Horizontal);
    m_HeightSlider.setFocusPolicy(Qt::StrongFocus);
    m_HeightSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_HeightSlider.setRange(0.0, 10.0);
	m_MainLayout.addWidget(&m_HeightSlider, 5, 1);
	
	m_HeightSpinBox.setRange(0.0, 10.0);
	m_MainLayout.addWidget(&m_HeightSpinBox, 5, 2);
	
	m_LockSizeCheckBox.setText("Lock Size");
	m_MainLayout.addWidget(&m_LockSizeCheckBox, 6, 1);

	connect(&m_HeightSlider, SIGNAL(valueChanged(double)), &m_HeightSpinBox, SLOT(setValue(double)));
	connect(&m_HeightSpinBox, SIGNAL(valueChanged(double)), &m_HeightSlider, SLOT(setValue(double)));
	connect(&m_HeightSlider, SIGNAL(valueChanged(double)), this, SLOT(OnHeightChanged(double)));

	connect(&m_LockSizeCheckBox, SIGNAL(stateChanged(int)), this, SLOT(OnLockSizeChanged(int)));
	
	// Color
	m_ColorLabel.setText("Color");
	m_MainLayout.addWidget(&m_ColorLabel, 7, 0);

	m_ColorButton.setText("...");
	m_ColorButton.setFixedWidth(120);
	m_ColorButton.setStatusTip("Pick a color");
	m_ColorButton.setToolTip("Pick a color");
	m_MainLayout.addWidget(&m_ColorButton, 7, 1);

	connect(&m_ColorButton, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));

	// Intensity
	m_IntensityLabel.setText("Intensity");
	m_MainLayout.addWidget(&m_IntensityLabel, 8, 0);

	m_IntensitySlider.setOrientation(Qt::Horizontal);
    m_IntensitySlider.setFocusPolicy(Qt::StrongFocus);
    m_IntensitySlider.setTickPosition(QDoubleSlider::NoTicks);
	m_IntensitySlider.setSingleStep(1);
	m_IntensitySlider.setRange(0.0, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySlider, 8, 1);
	
    m_IntensitySpinBox.setRange(0.0, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySpinBox, 8, 2);
	
	connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), &m_IntensitySpinBox, SLOT(setValue(double)));
	connect(&m_IntensitySpinBox, SIGNAL(valueChanged(double)), &m_IntensitySlider, SLOT(setValue(double)));
	connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), this, SLOT(OnIntensityChanged(double)));
	connect(&gLighting, SIGNAL(LightSelectionChanged(QLight*)), this, SLOT(OnLightSelectionChanged(QLight*)));
	connect(&gLighting.Background(), SIGNAL(BackgroundChanged()), &gLighting, SLOT(Update()));

	OnLightSelectionChanged(NULL);
}

void QLightSettingsWidget::OnLightSelectionChanged(QLight* pLight)
{
	QObject::disconnect(this, SLOT(OnLightThetaChanged(QLight*)));
	QObject::disconnect(this, SLOT(OnLightPhiChanged(QLight*)));
	QObject::disconnect(this, SLOT(OnLightDistanceChanged(QLight*)));
	QObject::disconnect(this, SLOT(OnLightWidthChanged(QLight*)));
	QObject::disconnect(this, SLOT(OnLightLockSizeChanged(QLight*)));
	QObject::disconnect(this, SLOT(OnLightHeightChanged(QLight*)));
	QObject::disconnect(this, SLOT(OnLightColorChanged(QLight*)));
	QObject::disconnect(this, SLOT(OnLightIntensityChanged(QLight*)));
		
	if (pLight)
	{
		connect(pLight, SIGNAL(ThetaChanged(QLight*)), this, SLOT(OnLightThetaChanged(QLight*)));
		connect(pLight, SIGNAL(PhiChanged(QLight*)), this, SLOT(OnLightPhiChanged(QLight*)));
		connect(pLight, SIGNAL(DistanceChanged(QLight*)), this, SLOT(OnLightDistanceChanged(QLight*)));
		connect(pLight, SIGNAL(WidthChanged(QLight*)), this, SLOT(OnLightWidthChanged(QLight*)));
		connect(pLight, SIGNAL(LockSizeChanged(QLight*)), this, SLOT(OnLightLockSizeChanged(QLight*)));
		connect(pLight, SIGNAL(HeightChanged(QLight*)), this, SLOT(OnLightHeightChanged(QLight*)));
		connect(pLight, SIGNAL(ColorChanged(QLight*)), this, SLOT(OnLightColorChanged(QLight*)));
		connect(pLight, SIGNAL(IntensityChanged(QLight*)), this, SLOT(OnLightIntensityChanged(QLight*)));
	}

	setEnabled(pLight != NULL);

	OnLightThetaChanged(pLight);
	OnLightPhiChanged(pLight);
	OnLightDistanceChanged(pLight);
	OnLightWidthChanged(pLight);
	OnLightLockSizeChanged(pLight);
	OnLightHeightChanged(pLight);
	OnLightColorChanged(pLight);
	OnLightIntensityChanged(pLight);
}

void QLightSettingsWidget::OnThetaChanged(const double& Theta)
{
	if (!gLighting.GetSelectedLight())
		return;
	
	gLighting.GetSelectedLight()->SetTheta((float)Theta);
}

void QLightSettingsWidget::OnPhiChanged(const double& Phi)
{
	if (!gLighting.GetSelectedLight())
		return;

	gLighting.GetSelectedLight()->SetPhi((float)Phi);
}

void QLightSettingsWidget::OnDistanceChanged(const double& Distance)
{
	if (!gLighting.GetSelectedLight())
		return;

	gLighting.GetSelectedLight()->SetDistance((float)Distance);
}

void QLightSettingsWidget::OnWidthChanged(const double& Width)
{
	if (!gLighting.GetSelectedLight())
		return;

	gLighting.GetSelectedLight()->SetWidth(Width);
}

void QLightSettingsWidget::OnLockSizeChanged(int LockSize)
{
	if (!gLighting.GetSelectedLight())
		return;

	gLighting.GetSelectedLight()->SetLockSize((bool)LockSize);
}

void QLightSettingsWidget::OnHeightChanged(const double& Height)
{
	if (!gLighting.GetSelectedLight())
		return;

	gLighting.GetSelectedLight()->SetHeight(Height);
}

void QLightSettingsWidget::OnCurrentColorChanged(const QColor& Color)
{
	if (!gLighting.GetSelectedLight())
		return;

	gLighting.GetSelectedLight()->SetColor(Color);

	m_ColorButton.SetColor(Color);
}

void QLightSettingsWidget::OnIntensityChanged(const double& Intensity)
{
	if (!gLighting.GetSelectedLight())
		return;

	gLighting.GetSelectedLight()->SetIntensity((float)Intensity);
}

void QLightSettingsWidget::OnLightThetaChanged(QLight* pLight)
{
	const bool Enable = pLight != NULL;

	if (pLight)
	{
		m_ThetaSlider.setValue(pLight->GetTheta(), true);
		m_ThetaSpinBox.setValue(pLight->GetTheta(), true);
	}

	m_ThetaLabel.setEnabled(Enable);
	m_ThetaSlider.setEnabled(Enable);
	m_ThetaSpinBox.setEnabled(Enable);
}

void QLightSettingsWidget::OnLightPhiChanged(QLight* pLight)
{
	const bool Enable = pLight != NULL;

	if (pLight)
	{
		m_PhiSlider.setValue(pLight->GetPhi(), true);
		m_PhiSpinBox.setValue(pLight->GetPhi(), true);
	}

	m_PhiLabel.setEnabled(Enable);
	m_PhiSlider.setEnabled(Enable);
	m_PhiSpinBox.setEnabled(Enable);
}

void QLightSettingsWidget::OnLightDistanceChanged(QLight* pLight)
{
	const bool Enable = pLight != NULL;

	if (pLight)
	{
		m_DistanceSlider.setValue(pLight->GetDistance(), true);
		m_DistanceSpinBox.setValue(pLight->GetDistance(), true);
	}

	m_DistanceLabel.setEnabled(Enable);
	m_DistanceSlider.setEnabled(Enable);
	m_DistanceSpinBox.setEnabled(Enable);
}

void QLightSettingsWidget::OnLightWidthChanged(QLight* pLight)
{
	const bool Enable = pLight != NULL;

	if (pLight)
	{
		m_WidthSlider.setValue(pLight->GetWidth(), true);
		m_WidthSpinBox.setValue(pLight->GetWidth(), true);
	}

	m_WidthLabel.setEnabled(Enable);
	m_WidthSlider.setEnabled(Enable);
	m_WidthSpinBox.setEnabled(Enable);
}

void QLightSettingsWidget::OnLightLockSizeChanged(QLight* pLight)
{
	bool Enable = true;

	if (pLight)
	{
		m_LockSizeCheckBox.blockSignals(true);
		m_LockSizeCheckBox.setChecked(pLight->GetLockSize());
		m_LockSizeCheckBox.blockSignals(false);

		const bool LockSize = pLight->GetLockSize();

		if (LockSize)
		{
			connect(&m_WidthSlider, SIGNAL(valueChanged(double)), &m_HeightSlider, SLOT(setValue(double)));
			connect(&m_WidthSpinBox, SIGNAL(valueChanged(double)), &m_HeightSpinBox, SLOT(setValue(double)));

			m_HeightSlider.setValue((double)m_WidthSlider.value());

			Enable = false;
		}
		else
		{
			disconnect(&m_WidthSlider, SIGNAL(valueChanged(double)), &m_HeightSlider, SLOT(setValue(double)));
			disconnect(&m_WidthSpinBox, SIGNAL(valueChanged(double)), &m_HeightSpinBox, SLOT(setValue(double)));
		}
	}

	m_HeightLabel.setEnabled(Enable);
	m_HeightSlider.setEnabled(Enable);
	m_HeightSpinBox.setEnabled(Enable);
}

void QLightSettingsWidget::OnLightHeightChanged(QLight* pLight)
{
	bool Enable = pLight ? (pLight->GetLockSize() ? false : true) : false;

	if (pLight)
	{
		m_HeightSlider.setValue(pLight->GetHeight(), true);
		m_HeightSpinBox.setValue(pLight->GetHeight(), true);
	}

	m_HeightLabel.setEnabled(Enable);
	m_HeightSlider.setEnabled(Enable);
	m_HeightSpinBox.setEnabled(Enable);
}

void QLightSettingsWidget::OnLightColorChanged(QLight* pLight)
{
	bool Enable = pLight!= NULL;

	if (pLight)
	{
		m_ColorButton.SetColor(pLight->GetColor(), true);
	}

	m_ColorLabel.setEnabled(Enable);
	m_ColorButton.setEnabled(Enable);
}

void QLightSettingsWidget::OnLightIntensityChanged(QLight* pLight)
{
	const bool Enable = pLight!= NULL;

	if (pLight)
	{
		m_IntensitySlider.setValue(pLight->GetIntensity(), true);
		m_IntensitySpinBox.setValue(pLight->GetIntensity(), true);
	}

	m_IntensityLabel.setEnabled(Enable);
	m_IntensitySlider.setEnabled(Enable);
	m_IntensitySpinBox.setEnabled(Enable);
}
