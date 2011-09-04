
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
	m_IntensitySpinBox(),
	m_pSelectedLight(NULL)
{
	// Title, status and tooltip
	setTitle("Light Settings");
	setToolTip("Light Settings");
	setStatusTip("Light Settings");

	// Disable
	setEnabled(false);

	// Apply main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	// Theta
	m_ThetaLabel.setText("Longitude");
	m_MainLayout.addWidget(&m_ThetaLabel, 0, 0);

	m_ThetaSlider.setOrientation(Qt::Orientation::Horizontal);
    m_ThetaSlider.setFocusPolicy(Qt::StrongFocus);
    m_ThetaSlider.setTickPosition(QDoubleSlider::TickPosition::NoTicks);
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

	m_PhiSlider.setOrientation(Qt::Orientation::Horizontal);
    m_PhiSlider.setFocusPolicy(Qt::StrongFocus);
    m_PhiSlider.setTickPosition(QDoubleSlider::TickPosition::NoTicks);
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

	m_DistanceSlider.setOrientation(Qt::Orientation::Horizontal);
    m_DistanceSlider.setFocusPolicy(Qt::StrongFocus);
    m_DistanceSlider.setTickPosition(QDoubleSlider::TickPosition::NoTicks);
	m_DistanceSlider.setRange(0.0, 100.0);
	m_MainLayout.addWidget(&m_DistanceSlider, 2, 1);
	
    m_DistanceSpinBox.setRange(0.0, 100.0);
	m_MainLayout.addWidget(&m_DistanceSpinBox, 2, 2);
	
	connect(&m_DistanceSlider, SIGNAL(valueChanged(double)), &m_DistanceSpinBox, SLOT(setValue(double)));
	connect(&m_DistanceSpinBox, SIGNAL(valueChanged(double)), &m_DistanceSlider, SLOT(setValue(double)));
	connect(&m_DistanceSlider, SIGNAL(valueChanged(double)), this, SLOT(OnDistanceChanged(double)));

	// Width
	m_WidthLabel.setText("Width");
	m_MainLayout.addWidget(&m_WidthLabel, 3, 0);

	m_WidthSlider.setOrientation(Qt::Orientation::Horizontal);
    m_WidthSlider.setFocusPolicy(Qt::StrongFocus);
    m_WidthSlider.setTickPosition(QDoubleSlider::TickPosition::NoTicks);
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

	m_HeightSlider.setOrientation(Qt::Orientation::Horizontal);
    m_HeightSlider.setFocusPolicy(Qt::StrongFocus);
    m_HeightSlider.setTickPosition(QDoubleSlider::TickPosition::NoTicks);
	m_HeightSlider.setRange(0.0, 10.0);
	m_MainLayout.addWidget(&m_HeightSlider, 5, 1);
	
	m_HeightSpinBox.setRange(0.0, 10.0);
	m_MainLayout.addWidget(&m_HeightSpinBox, 5, 2);
	
	m_LockSizeCheckBox.setText("Lock Size");
	m_MainLayout.addWidget(&m_LockSizeCheckBox, 6, 1);

	connect(&m_HeightSlider, SIGNAL(valueChanged(double)), &m_HeightSpinBox, SLOT(setValue(double)));
	connect(&m_HeightSpinBox, SIGNAL(valueChanged(double)), &m_HeightSlider, SLOT(setValue(double)));
	connect(&m_HeightSlider, SIGNAL(valueChanged(double)), this, SLOT(OnHeightChanged(double)));

	connect(&m_LockSizeCheckBox, SIGNAL(stateChanged(int)), this, SLOT(OnLockSize(int)));
	
	// Color
	m_ColorLabel.setText("Color");
	m_MainLayout.addWidget(&m_ColorLabel, 7, 0);

	m_ColorButton.setText("...");
	m_ColorButton.setStatusTip("Pick a color");
	m_ColorButton.setToolTip("Pick a color");
	m_MainLayout.addWidget(&m_ColorButton, 7, 1);

	connect(&m_ColorButton, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));

	// Intensity
	m_IntensityLabel.setText("Intensity");
	m_MainLayout.addWidget(&m_IntensityLabel, 8, 0);

	m_IntensitySlider.setOrientation(Qt::Orientation::Horizontal);
    m_IntensitySlider.setFocusPolicy(Qt::StrongFocus);
    m_IntensitySlider.setTickPosition(QDoubleSlider::TickPosition::NoTicks);
	m_IntensitySlider.setSingleStep(1);
	m_IntensitySlider.setRange(0.0, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySlider, 8, 1);
	
    m_IntensitySpinBox.setRange(0.0, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySpinBox, 8, 2);
	
	connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), &m_IntensitySpinBox, SLOT(setValue(double)));
	connect(&m_IntensitySpinBox, SIGNAL(valueChanged(double)), &m_IntensitySlider, SLOT(setValue(double)));
	connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), this, SLOT(OnIntensityChanged(double)));

	OnLightSelectionChanged(NULL);
}

void QLightSettingsWidget::OnLightSelectionChanged(QLight* pLight)
{
	m_pSelectedLight = pLight;

	if (m_pSelectedLight)
	{
		setEnabled(true);
		setTitle("Light Settings");
		setStatusTip("Light settings for " + m_pSelectedLight->GetName());
		setToolTip("Light settings for " + m_pSelectedLight->GetName());

		m_ThetaSlider.setValue(m_pSelectedLight->GetTheta() * RAD_F);
		m_PhiSlider.setValue(m_pSelectedLight->GetPhi() * RAD_F);
		
		// Distance
		m_DistanceSlider.setValue(m_pSelectedLight->GetDistance());
		
		// Width
		m_WidthSlider.setValue(m_pSelectedLight->GetWidth());
		m_HeightSlider.setValue(m_pSelectedLight->GetHeight());

		QPropertyAnimation animation(&m_WidthSlider, "value");
		animation.setDuration(1000);
		animation.setStartValue(m_WidthSlider.value());
		animation.setEndValue(100.0);

		animation.start();

		// Lock size
		m_LockSizeCheckBox.setChecked(m_pSelectedLight->GetLockSize());
		
		// Color
		m_ColorButton.SetColor(m_pSelectedLight->GetColor());
		
		// Intensity
		m_IntensitySlider.setValue(m_pSelectedLight->GetIntensity());
	}
	else
	{
		setEnabled(false);
	}
}

void QLightSettingsWidget::OnLockSize(const int& LockSize)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetLockSize((bool)LockSize);

	m_HeightLabel.setEnabled(!LockSize);
	m_HeightSlider.setEnabled(!LockSize);
	m_HeightSpinBox.setEnabled(!LockSize);

	if (LockSize)
	{
		connect(&m_WidthSlider, SIGNAL(valueChanged(double)), &m_HeightSlider, SLOT(setValue(double)));
		connect(&m_WidthSpinBox, SIGNAL(valueChanged(double)), &m_HeightSpinBox, SLOT(setValue(double)));

		m_HeightSlider.setValue((double)m_WidthSlider.value());
	}
	else
	{
 		disconnect(&m_WidthSlider, SIGNAL(valueChanged(double)), &m_HeightSlider, SLOT(setValue(double)));
 		disconnect(&m_WidthSpinBox, SIGNAL(valueChanged(double)), &m_HeightSpinBox, SLOT(setValue(double)));
	}
}

void QLightSettingsWidget::OnThetaChanged(const double& Theta)
{
	if (!m_pSelectedLight)
		return;
	
	m_pSelectedLight->SetTheta((float)Theta / RAD_F);
}

void QLightSettingsWidget::OnPhiChanged(const double& Phi)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetPhi((float)Phi / RAD_F);
}

void QLightSettingsWidget::OnDistanceChanged(const double& Distance)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetDistance((float)Distance);
}

void QLightSettingsWidget::OnWidthChanged(const double& Width)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetWidth(Width);
}

void QLightSettingsWidget::OnHeightChanged(const double& Height)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetHeight(Height);
}

void QLightSettingsWidget::OnCurrentColorChanged(const QColor& Color)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetColor(Color);

	m_ColorButton.SetColor(Color);
}

void QLightSettingsWidget::OnIntensityChanged(const double& Intensity)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetIntensity((float)Intensity);
}