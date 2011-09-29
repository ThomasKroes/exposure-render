
// Precompiled headers
#include "Stable.h"

#include "BackgroundIlluminationWidget.h"
#include "LightsWidget.h"
#include "RenderThread.h"

QBackgroundIlluminationWidget::QBackgroundIlluminationWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_GradientColorTopLabel(),
	m_GradientColorTop(),
	m_GradientColorMiddleLabel(),
	m_GradientColorMiddle(),
	m_GradientColorBottomLabel(),
	m_GradientColorBottom(),
	m_IntensityLabel(),
	m_IntensitySlider(),
	m_IntensitySpinner(),
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

	// Gradient color top
	m_GradientColorTopLabel.setText("Top");
	m_GradientColorTopLabel.setFixedWidth(50);
	m_MainLayout.addWidget(&m_GradientColorTopLabel, 0, 0);
	m_MainLayout.addWidget(&m_GradientColorTop, 0, 1, 1, 3);

	QObject::connect(&m_GradientColorTop, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnGradientColorTopChanged(const QColor&)));

	// Gradient color Middle
	m_GradientColorMiddleLabel.setText("Middle");
	m_MainLayout.addWidget(&m_GradientColorMiddleLabel, 1, 0);
	m_MainLayout.addWidget(&m_GradientColorMiddle, 1, 1, 1, 3);

	QObject::connect(&m_GradientColorMiddle, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnGradientColorMiddleChanged(const QColor&)));

	// Gradient color Bottom
	m_GradientColorBottomLabel.setText("Bottom");
	m_MainLayout.addWidget(&m_GradientColorBottomLabel, 2, 0);
	m_MainLayout.addWidget(&m_GradientColorBottom, 2, 1, 1, 3);

	QObject::connect(&m_GradientColorBottom, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnGradientColorBottomChanged(const QColor&)));

	// Intensity
	m_IntensityLabel.setText("Intensity");
	m_MainLayout.addWidget(&m_IntensityLabel, 3, 0);

	m_IntensitySlider.setOrientation(Qt::Horizontal);
	m_IntensitySlider.setRange(0.0, 100.0);
	m_MainLayout.addWidget(&m_IntensitySlider, 3, 1, 1, 2);

	m_IntensitySpinner.setRange(0.0, 100.0);
	m_MainLayout.addWidget(&m_IntensitySpinner, 3, 3);

	QObject::connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), &m_IntensitySpinner, SLOT(setValue(double)));
	QObject::connect(&m_IntensitySpinner, SIGNAL(valueChanged(double)), &m_IntensitySlider, SLOT(setValue(double)));
	QObject::connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), this, SLOT(OnIntensityChanged(double)));

	// Use Texture
// 	m_UseTexture.setText("Use Texture");
// 	m_MainLayout.addWidget(&m_UseTexture, 4, 1);

	// Texture
// 	m_MainLayout.addWidget(new QLabel("Texture"), 5, 0);

	// Path
// 	m_TextureFilePath.setFixedHeight(22);
// 	m_MainLayout.addWidget(&m_TextureFilePath, 5, 1, 1, 2);

// 	m_LoadTexture.setIcon(GetIcon("folder-open-image"));
// 	m_LoadTexture.setFixedWidth(22);
// 	m_LoadTexture.setFixedHeight(22);
// 	m_MainLayout.addWidget(&m_LoadTexture, 5, 3);
// 
 	QObject::connect(this, SIGNAL(toggled(bool)), this, SLOT(OnBackgroundIlluminationChanged(bool)));
// 	QObject::connect(&m_UseTexture, SIGNAL(stateChanged(int)), this, SLOT(OnUseTextureChanged(int)));
// 	QObject::connect(&m_LoadTexture, SIGNAL(clicked()), this, SLOT(OnLoadTexture()));

 	QObject::connect(&gLighting.Background(), SIGNAL(Changed()), this, SLOT(OnBackgroundChanged()));

	OnBackgroundChanged();
}

void QBackgroundIlluminationWidget::OnBackgroundIlluminationChanged(bool Checked)
{
	gLighting.Background().SetEnabled(Checked);
}

void QBackgroundIlluminationWidget::OnGradientColorTopChanged(const QColor& Color)
{
	gLighting.Background().SetTopColor(Color);
}

void QBackgroundIlluminationWidget::OnGradientColorMiddleChanged(const QColor& Color)
{
	gLighting.Background().SetMiddleColor(Color);
}

void QBackgroundIlluminationWidget::OnGradientColorBottomChanged(const QColor& Color)
{
	gLighting.Background().SetBottomColor(Color);
}

void QBackgroundIlluminationWidget::OnIntensityChanged(double Intensity)
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

	m_GradientColorTop.setEnabled(gLighting.Background().GetEnabled() && !gLighting.Background().GetUseTexture());
	m_GradientColorMiddle.setEnabled(gLighting.Background().GetEnabled() && !gLighting.Background().GetUseTexture());
	m_GradientColorBottom.setEnabled(gLighting.Background().GetEnabled() && !gLighting.Background().GetUseTexture());

	m_GradientColorTop.SetColor(gLighting.Background().GetTopColor());
	m_GradientColorMiddle.SetColor(gLighting.Background().GetMiddleColor());
	m_GradientColorBottom.SetColor(gLighting.Background().GetBottomColor());

	m_IntensitySlider.setValue((double)gLighting.Background().GetIntensity(), true);

	// Use texture
	m_TextureFilePath.setEnabled(gLighting.Background().GetEnabled() && gLighting.Background().GetUseTexture());
	m_UseTexture.setChecked(gLighting.Background().GetUseTexture());

	// Use texture
	m_LoadTexture.setEnabled(gLighting.Background().GetEnabled() && gLighting.Background().GetUseTexture());
}
