
// Precompiled headers
#include "Stable.h"

#include "BackgroundIlluminationWidget.h"
#include "LightsWidget.h"
#include "RenderThread.h"

QBackgroundIlluminationWidget::QBackgroundIlluminationWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_GradientColorTop(),
	m_IntensitySliderTop(),
	m_IntensitySpinBoxTop(),
	m_GradientColorMiddle(),
	m_IntensitySliderMiddle(),
	m_IntensitySpinBoxMiddle(),
	m_GradientColorBottom(),
	m_IntensitySliderBottom(),
	m_IntensitySpinBoxBottom(),
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
	m_MainLayout.addWidget(new QLabel("Top"), 0, 0);
	m_MainLayout.addWidget(&m_GradientColorTop, 0, 1);

	m_IntensitySliderTop.setOrientation(Qt::Horizontal);
	m_IntensitySliderTop.setRange(0.01, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySliderTop, 0, 2);

	m_IntensitySpinBoxTop.setRange(0.01, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySpinBoxTop, 0, 3);

	connect(&m_GradientColorTop, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnGradientColorTopChanged(const QColor&)));
	connect(&m_IntensitySliderTop, SIGNAL(valueChanged(double)), &m_IntensitySpinBoxTop, SLOT(setValue(double)));
	connect(&m_IntensitySpinBoxTop, SIGNAL(valueChanged(double)), &m_IntensitySliderTop, SLOT(setValue(double)));
	connect(&m_IntensitySliderTop, SIGNAL(valueChanged(double)), this, SLOT(OnTopIntensityChanged(double)));

	// Gradient color Middle
	m_MainLayout.addWidget(new QLabel("Middle"), 1, 0);
	m_MainLayout.addWidget(&m_GradientColorMiddle, 1, 1);

	m_IntensitySliderMiddle.setOrientation(Qt::Horizontal);
	m_IntensitySliderMiddle.setRange(0.01, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySliderMiddle, 1, 2);

	m_IntensitySpinBoxMiddle.setRange(0.01, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySpinBoxMiddle, 1, 3);

	connect(&m_GradientColorMiddle, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnGradientColorMiddleChanged(const QColor&)));
	connect(&m_IntensitySliderMiddle, SIGNAL(valueChanged(double)), &m_IntensitySpinBoxMiddle, SLOT(setValue(double)));
	connect(&m_IntensitySpinBoxMiddle, SIGNAL(valueChanged(double)), &m_IntensitySliderMiddle, SLOT(setValue(double)));
	connect(&m_IntensitySliderMiddle, SIGNAL(valueChanged(double)), this, SLOT(OnMiddleIntensityChanged(double)));

	// Gradient color Bottom
	m_MainLayout.addWidget(new QLabel("Bottom"), 2, 0);
	m_MainLayout.addWidget(&m_GradientColorBottom, 2, 1);

	m_IntensitySliderBottom.setOrientation(Qt::Horizontal);
	m_IntensitySliderBottom.setRange(0.01, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySliderBottom, 2, 2);

	m_IntensitySpinBoxBottom.setRange(0.01, 1000.0);
	m_MainLayout.addWidget(&m_IntensitySpinBoxBottom, 2, 3);

	connect(&m_GradientColorBottom, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnGradientColorBottomChanged(const QColor&)));
	connect(&m_IntensitySliderBottom, SIGNAL(valueChanged(double)), &m_IntensitySpinBoxBottom, SLOT(setValue(double)));
	connect(&m_IntensitySpinBoxBottom, SIGNAL(valueChanged(double)), &m_IntensitySliderBottom, SLOT(setValue(double)));
	connect(&m_IntensitySliderBottom, SIGNAL(valueChanged(double)), this, SLOT(OnBottomIntensityChanged(double)));

	// Use Texture
	m_UseTexture.setText("Use Texture");
	m_MainLayout.addWidget(&m_UseTexture, 3, 1);

	// Texture
	m_MainLayout.addWidget(new QLabel("Texture"), 4, 0);

	// Path
	m_TextureFilePath.setFixedHeight(22);
	m_MainLayout.addWidget(&m_TextureFilePath, 4, 1, 1, 2);

	m_LoadTexture.setIcon(GetIcon("folder-open-image"));
	m_LoadTexture.setFixedWidth(22);
	m_LoadTexture.setFixedHeight(22);
	m_MainLayout.addWidget(&m_LoadTexture, 4, 3);

	connect(this, SIGNAL(toggled(bool)), this, SLOT(OnBackgroundIlluminationChanged(bool)));
	connect(&m_UseTexture, SIGNAL(stateChanged(int)), this, SLOT(OnUseTextureChanged(int)));
	connect(&m_LoadTexture, SIGNAL(clicked()), this, SLOT(OnLoadTexture()));
// 	connect(&gLighting.Background(), SIGNAL(BackgroundChanged()), this, SLOT(OnBackgroundChanged()));

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

void QBackgroundIlluminationWidget::OnTopIntensityChanged(double Intensity)
{
	gLighting.Background().SetTopIntensity(Intensity);
}

void QBackgroundIlluminationWidget::OnMiddleIntensityChanged(double Intensity)
{
	gLighting.Background().SetMiddleIntensity(Intensity);
}

void QBackgroundIlluminationWidget::OnBottomIntensityChanged(double Intensity)
{
	gLighting.Background().SetBottomIntensity(Intensity);
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

	m_IntensitySliderTop.setValue((double)gLighting.Background().GetTopIntensity(), true);
	m_IntensitySliderMiddle.setValue((double)gLighting.Background().GetMiddleIntensity(), true);
	m_IntensitySliderBottom.setValue((double)gLighting.Background().GetBottomIntensity(), true);

	// Use texture
	m_TextureFilePath.setEnabled(gLighting.Background().GetEnabled() && gLighting.Background().GetUseTexture());
	m_UseTexture.setChecked(gLighting.Background().GetUseTexture());

	// Use texture
	m_LoadTexture.setEnabled(gLighting.Background().GetEnabled() && gLighting.Background().GetUseTexture());
}
