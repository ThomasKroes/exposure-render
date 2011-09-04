
#include "BackgroundIlluminationWidget.h"
#include "LightsWidget.h"
#include "RenderThread.h"

QBackgroundIlluminationWidget::QBackgroundIlluminationWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_Color(),
	m_IntensitySlider(),
	m_IntensitySpinBox(),
	m_UseTexture(),
	m_TextureFilePath(),
	m_LoadTexture()
{
	// Title, status and tooltip
	setTitle("Background Illumination");
	setToolTip("Background Illumination");
	setStatusTip("Background Illumination");

	// Allow user to turn background illumination on/off
	setCheckable(true);

	// Apply main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	// Color
	m_MainLayout.addWidget(new QLabel("Color"), 0, 0);
	m_MainLayout.addWidget(&m_Color, 0, 1);

	m_MainLayout.addWidget(new QLabel("Intensity"), 1, 0);

	m_IntensitySlider.setOrientation(Qt::Orientation::Horizontal);
	m_IntensitySlider.setFocusPolicy(Qt::StrongFocus);
	m_IntensitySlider.setTickPosition(QDoubleSlider::TickPosition::NoTicks);
	m_IntensitySlider.setRange(0.0, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySlider, 1, 1, 1, 2);

	m_IntensitySpinBox.setRange(0.0, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySpinBox, 1, 3);

	// Use Texture
	m_UseTexture.setText("Use Texture");
	m_MainLayout.addWidget(&m_UseTexture, 2, 1);

	// Texture
	m_MainLayout.addWidget(new QLabel("Texture"), 3, 0);

	// Path
	m_TextureFilePath.setFixedHeight(22);
	m_MainLayout.addWidget(&m_TextureFilePath, 3, 1, 1, 2);

	m_LoadTexture.setIcon(QIcon(":/Images/folder-open-image.png"));
	m_LoadTexture.setFixedWidth(22);
	m_LoadTexture.setFixedHeight(22);
	m_MainLayout.addWidget(&m_LoadTexture, 3, 3);

	connect(this, SIGNAL(toggled(bool)), this, SLOT(OnBackgroundIlluminationChanged(bool)));
	connect(&m_Color, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnColorChanged(const QColor&)));
	connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), &m_IntensitySpinBox, SLOT(setValue(double)));
	connect(&m_IntensitySpinBox, SIGNAL(valueChanged(double)), &m_IntensitySlider, SLOT(setValue(double)));
	connect(&m_IntensitySpinBox, SIGNAL(valueChanged(double)), this, SLOT(SetIntensity(double)));
	connect(&m_UseTexture, SIGNAL(stateChanged(int)), this, SLOT(OnUseTextureChanged(int)));
	connect(&m_LoadTexture, SIGNAL(clicked()), this, SLOT(OnLoadTexture()));
	connect(&gLighting.Background(), SIGNAL(BackgroundChanged()), this, SLOT(OnBackgroundChanged()));

	OnBackgroundChanged();
}

void QBackgroundIlluminationWidget::OnBackgroundIlluminationChanged(bool Checked)
{
	gLighting.Background().SetEnabled(Checked);
}

void QBackgroundIlluminationWidget::OnColorChanged(const QColor& Color)
{
	gLighting.Background().SetColor(Color);
}

void QBackgroundIlluminationWidget::SetIntensity(double Intensity)
{
	gLighting.Background().SetIntensity(Intensity);
}

void QBackgroundIlluminationWidget::OnUseTextureChanged(int UseTexture)
{
	gLighting.Background().SetUseTexture(UseTexture);
}

void QBackgroundIlluminationWidget::OnLoadTexture(void)
{

}

void QBackgroundIlluminationWidget::OnBackgroundChanged(void)
{
	setChecked(gLighting.Background().GetEnabled());

	// Color
	m_Color.setEnabled(gLighting.Background().GetEnabled() && !gLighting.Background().GetUseTexture());
	m_Color.SetColor(gLighting.Background().GetColor());
	
	// Intensity
	m_IntensitySlider.setValue((double)gLighting.Background().GetIntensity());

	// Use texture
	m_TextureFilePath.setEnabled(gLighting.Background().GetEnabled() && gLighting.Background().GetUseTexture());
	m_UseTexture.setChecked(gLighting.Background().GetUseTexture());

	// Use texture
	m_LoadTexture.setEnabled(gLighting.Background().GetEnabled() && gLighting.Background().GetUseTexture());
}
