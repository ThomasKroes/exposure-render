
#include "LightingSettingsWidget.h"
#include "LightsWidget.h"
#include "RenderThread.h"

QColorPushButton::QColorPushButton(QWidget* pParent) :
	QPushButton(pParent),
	m_Margin(5),
	m_Radius(4),
	m_Color(Qt::blue)
{
}

void QColorPushButton::paintEvent(QPaintEvent* pPaintEvent)
{
	QPushButton::paintEvent(pPaintEvent);

	QPainter Painter(this);

	// Get button rectangle
	QRect ColorRectangle = pPaintEvent->rect();

	// Deflate it
	ColorRectangle.adjust(m_Margin, m_Margin, -m_Margin, -m_Margin);

	// Use anti aliasing
	Painter.setRenderHints(QPainter::Antialiasing);

	// Rectangle styling
	Painter.setBrush(QBrush(m_Color));
	Painter.setPen(QPen(Qt::darkGray));

	// Draw
	Painter.drawRoundedRect(ColorRectangle, m_Radius, Qt::SizeMode::AbsoluteSize);

//	int Grey = (float)(m_Color.red() + m_Color.green() + m_Color.blue()) / 3.0f;

	// Draw text
//	Painter.setFont(QFont("Arial", 15));
//	Painter.setPen(QPen(QColor(255 - Grey, 255 - Grey, 255 - Grey)));
//	Painter.drawText(ColorRectangle, Qt::AlignCenter, "...");
}

void QColorPushButton::mousePressEvent(QMouseEvent* pEvent)
{
	QColorDialog ColorDialog;

	connect(&ColorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));

	ColorDialog.exec();

	disconnect(&ColorDialog, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));
}

int QColorPushButton::GetMargin(void) const
{
	return m_Margin;
}

void QColorPushButton::SetMargin(const int& Margin)
{
	m_Margin = m_Margin;
	update();
}

int QColorPushButton::GetRadius(void) const
{
	return m_Radius;
}

void QColorPushButton::SetRadius(const int& Radius)
{
	m_Radius = m_Radius;
	update();
}

QColor QColorPushButton::GetColor(void) const
{
	return m_Color;
}

void QColorPushButton::SetColor(const QColor& Color)
{
	m_Color = Color;
	update();
}

void QColorPushButton::OnCurrentColorChanged(const QColor& Color)
{
	SetColor(Color);

	emit currentColorChanged(m_Color);
}

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
	m_LockHeightCheckBox(),
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
	setEnabled(true);

	// Apply main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	// Theta
	m_ThetaLabel.setText("Theta");
	m_MainLayout.addWidget(&m_ThetaLabel, 0, 0);

	m_ThetaSlider.setOrientation(Qt::Orientation::Horizontal);
    m_ThetaSlider.setFocusPolicy(Qt::StrongFocus);
    m_ThetaSlider.setTickPosition(QSlider::TickPosition::NoTicks);
	m_ThetaSlider.setRange(-360, 360);
	m_MainLayout.addWidget(&m_ThetaSlider, 0, 1);
	
    m_ThetaSpinBox.setRange(-360, 360);
	m_MainLayout.addWidget(&m_ThetaSpinBox, 0, 2);
	
	connect(&m_ThetaSlider, SIGNAL(valueChanged(int)), &m_ThetaSpinBox, SLOT(setValue(int)));
	connect(&m_ThetaSpinBox, SIGNAL(valueChanged(int)), &m_ThetaSlider, SLOT(setValue(int)));
	connect(&m_ThetaSlider, SIGNAL(valueChanged(int)), this, SLOT(OnThetaChanged(int)));

	// Phi
	m_PhiLabel.setText("Phi");
	m_MainLayout.addWidget(&m_PhiLabel, 1, 0);

	m_PhiSlider.setOrientation(Qt::Orientation::Horizontal);
    m_PhiSlider.setFocusPolicy(Qt::StrongFocus);
    m_PhiSlider.setTickPosition(QSlider::TickPosition::NoTicks);
	m_PhiSlider.setRange(-90, 90);
	m_MainLayout.addWidget(&m_PhiSlider, 1, 1);
	
    m_PhiSpinBox.setRange(-90, 90);
	m_MainLayout.addWidget(&m_PhiSpinBox, 1, 2);
	
	connect(&m_PhiSlider, SIGNAL(valueChanged(int)), &m_PhiSpinBox, SLOT(setValue(int)));
	connect(&m_PhiSpinBox, SIGNAL(valueChanged(int)), &m_PhiSlider, SLOT(setValue(int)));
	connect(&m_PhiSlider, SIGNAL(valueChanged(int)), this, SLOT(OnPhiChanged(int)));

	// Distance
	m_DistanceLabel.setText("Distance");
	m_MainLayout.addWidget(&m_DistanceLabel, 2, 0);

	m_DistanceSlider.setOrientation(Qt::Orientation::Horizontal);
    m_DistanceSlider.setFocusPolicy(Qt::StrongFocus);
    m_DistanceSlider.setTickPosition(QSlider::TickPosition::NoTicks);
	m_MainLayout.addWidget(&m_DistanceSlider, 2, 1);
	
    m_DistanceSpinBox.setRange(-90, 90);
	m_MainLayout.addWidget(&m_DistanceSpinBox, 2, 2);
	
	connect(&m_DistanceSlider, SIGNAL(valueChanged(int)), &m_DistanceSpinBox, SLOT(setValue(int)));
	connect(&m_DistanceSpinBox, SIGNAL(valueChanged(int)), &m_DistanceSlider, SLOT(setValue(int)));
	connect(&m_DistanceSlider, SIGNAL(valueChanged(int)), this, SLOT(OnDistanceChanged(int)));

	// Width
	m_WidthLabel.setText("Width");
	m_MainLayout.addWidget(&m_WidthLabel, 3, 0);

	m_WidthSlider.setOrientation(Qt::Orientation::Horizontal);
    m_WidthSlider.setFocusPolicy(Qt::StrongFocus);
    m_WidthSlider.setTickPosition(QSlider::TickPosition::NoTicks);
	m_MainLayout.addWidget(&m_WidthSlider, 3, 1);
	
    m_WidthSpinBox.setRange(0, 100);
	m_MainLayout.addWidget(&m_WidthSpinBox, 3, 2);
	
	connect(&m_WidthSlider, SIGNAL(valueChanged(int)), &m_WidthSpinBox, SLOT(setValue(int)));
	connect(&m_WidthSpinBox, SIGNAL(valueChanged(int)), &m_WidthSlider, SLOT(setValue(int)));
	connect(&m_WidthSlider, SIGNAL(valueChanged(int)), this, SLOT(OnWidthChanged(int)));

	// Height
	m_HeightLabel.setText("Height");
	m_MainLayout.addWidget(&m_HeightLabel, 5, 0);

	m_HeightSlider.setOrientation(Qt::Orientation::Horizontal);
    m_HeightSlider.setFocusPolicy(Qt::StrongFocus);
    m_HeightSlider.setTickPosition(QSlider::TickPosition::NoTicks);
	m_MainLayout.addWidget(&m_HeightSlider, 5, 1);
	
    m_HeightSpinBox.setRange(0, 100);
	m_MainLayout.addWidget(&m_HeightSpinBox, 5, 2);
	
	m_LockHeightCheckBox.setText("Lock width and height");
	m_MainLayout.addWidget(&m_LockHeightCheckBox, 6, 1);

	connect(&m_HeightSlider, SIGNAL(valueChanged(int)), &m_HeightSpinBox, SLOT(setValue(int)));
	connect(&m_HeightSpinBox, SIGNAL(valueChanged(int)), &m_HeightSlider, SLOT(setValue(int)));
	connect(&m_HeightSlider, SIGNAL(valueChanged(int)), this, SLOT(OnHeightChanged(int)));

	connect(&m_LockHeightCheckBox, SIGNAL(stateChanged(int)), this, SLOT(OnLockHeight(int)));
	
	// Color
	m_ColorLabel.setText("Color");
	m_MainLayout.addWidget(&m_ColorLabel, 7, 0);

	m_ColorButton.setText("...");
	m_ColorButton.setFixedWidth(80);
	m_ColorButton.setFixedHeight(22);
	m_ColorButton.setStatusTip("Pick a color");
	m_ColorButton.setToolTip("Pick a color");
	m_MainLayout.addWidget(&m_ColorButton, 7, 1);

	connect(&m_ColorButton, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnCurrentColorChanged(const QColor&)));

	// Intensity
	m_IntensityLabel.setText("Intensity");
	m_MainLayout.addWidget(&m_IntensityLabel, 8, 0);

	m_IntensitySlider.setOrientation(Qt::Orientation::Horizontal);
    m_IntensitySlider.setFocusPolicy(Qt::StrongFocus);
    m_IntensitySlider.setTickPosition(QSlider::TickPosition::NoTicks);
	m_MainLayout.addWidget(&m_IntensitySlider, 8, 1);
	
    m_IntensitySpinBox.setRange(0, 100);
	m_MainLayout.addWidget(&m_IntensitySpinBox, 8, 2);
	
	connect(&m_IntensitySlider, SIGNAL(valueChanged(int)), &m_IntensitySpinBox, SLOT(setValue(int)));
	connect(&m_IntensitySpinBox, SIGNAL(valueChanged(int)), &m_IntensitySlider, SLOT(setValue(int)));
	connect(&m_IntensitySlider, SIGNAL(valueChanged(int)), &m_HeightSpinBox, SLOT(OnIntensityChanged(int)));
}

void QLightSettingsWidget::OnLightSelectionChanged(QLight* pLight)
{
	m_pSelectedLight = pLight;

	if (m_pSelectedLight)
	{
		setEnabled(true);
		setTitle("Light Settings: " + m_pSelectedLight->GetName());
		setStatusTip("Light settings for " + m_pSelectedLight->GetName());
		setToolTip("Light settings for " + m_pSelectedLight->GetName());
	}
	else
	{
		setEnabled(false);
	}
}

void QLightSettingsWidget::OnLockHeight(const int& State)
{
	m_HeightLabel.setEnabled(!State);
	m_HeightSlider.setEnabled(!State);
	m_HeightSpinBox.setEnabled(!State);

	if (State)
	{
		connect(&m_WidthSlider, SIGNAL(valueChanged(int)), &m_HeightSlider, SLOT(setValue(int)));
		connect(&m_WidthSpinBox, SIGNAL(valueChanged(int)), &m_HeightSpinBox, SLOT(setValue(int)));

		m_HeightSlider.setValue(m_WidthSlider.value());
	}
	else
	{
		disconnect(&m_WidthSlider, SIGNAL(valueChanged(int)), &m_HeightSlider, SLOT(setValue(int)));
		disconnect(&m_WidthSpinBox, SIGNAL(valueChanged(int)), &m_HeightSpinBox, SLOT(setValue(int)));
	}
}

void QLightSettingsWidget::OnThetaChanged(const int& Theta)
{
	if (!m_pSelectedLight)
		return;
	
	m_pSelectedLight->SetTheta((float)Theta / RAD_F);
}

void QLightSettingsWidget::OnPhiChanged(const int& Phi)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetPhi((float)Phi / RAD_F);
}

void QLightSettingsWidget::OnDistanceChanged(const int& Distance)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetDistance(0.1f * (float)Distance);
}

void QLightSettingsWidget::OnWidthChanged(const int& Width)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetWidth(0.1f * Width);
}

void QLightSettingsWidget::OnHeightChanged(const int& Height)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetHeight(0.1f * Height);
}

void QLightSettingsWidget::OnCurrentColorChanged(const QColor& Color)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetColor(Color);

	m_ColorButton.SetColor(Color);
}

void QLightSettingsWidget::OnIntensityChanged(const int& Intensity)
{
	if (!m_pSelectedLight)
		return;

	m_pSelectedLight->SetIntensity(0.1f * (float)Intensity);
}