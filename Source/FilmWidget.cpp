
// Precompiled headers
#include "Stable.h"

#include "FilmWidget.h"
#include "RenderThread.h"
#include "Camera.h"

// QFilmResolutionPreset
QFilmWidget::QFilmWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_PresetType(),
	m_PresetsLayout(),
	m_WidthSpinner(),
	m_HeightSpinner(),
	m_ExposureSlider(),
	m_ExposureSpinner(),
	m_NoiseReduction()
{
	setTitle("Film");
	setStatusTip("Film properties");
	setToolTip("Film properties");

	// Create grid layout
	m_GridLayout.setColumnMinimumWidth(0, 75);
	setLayout(&m_GridLayout);

	m_PresetType.addItem("NTSC D-1 (video)");
	m_PresetType.addItem("NTSC DV (video)");
	m_PresetType.addItem("PAL (video)");
	m_PresetType.addItem("PAL D-1 (video)");
	m_PresetType.addItem("HDTV (video)");

	m_GridLayout.addWidget(new QLabel("Type"), 0, 0);
	m_GridLayout.addWidget(&m_PresetType, 0, 1);
	
	m_GridLayout.addLayout(&m_PresetsLayout, 1, 2, 2, 1);
	
	m_PresetsLayout.addWidget(&m_Preset[0], 0, 1);
	m_PresetsLayout.addWidget(&m_Preset[1], 0, 2);
	m_PresetsLayout.addWidget(&m_Preset[2], 1, 1);
	m_PresetsLayout.addWidget(&m_Preset[3], 1, 2);

	QObject::connect(&m_PresetType, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(SetPresetType(const QString&)));
	
	QObject::connect(&m_Preset[0], SIGNAL(SetPreset(QFilmResolutionPreset&)), this, SLOT(SetPreset(QFilmResolutionPreset&)));
	QObject::connect(&m_Preset[1], SIGNAL(SetPreset(QFilmResolutionPreset&)), this, SLOT(SetPreset(QFilmResolutionPreset&)));
	QObject::connect(&m_Preset[2], SIGNAL(SetPreset(QFilmResolutionPreset&)), this, SLOT(SetPreset(QFilmResolutionPreset&)));
	QObject::connect(&m_Preset[3], SIGNAL(SetPreset(QFilmResolutionPreset&)), this, SLOT(SetPreset(QFilmResolutionPreset&)));

	m_PresetType.setCurrentIndex(4);

	const int ResMin = powf(2.0f, 5);
	const int ResMax = powf(2.0f, 11);

	// Film width
	m_GridLayout.addWidget(new QLabel("Film width"), 1, 0);

    m_WidthSpinner.setRange(ResMin, ResMax);
	m_GridLayout.addWidget(&m_WidthSpinner, 1, 1);
	
 	QObject::connect(&m_WidthSpinner, SIGNAL(valueChanged(int)), this, SLOT(SetWidth(int)));

	QPushButton B;
	
	// Film height
	m_GridLayout.addWidget(new QLabel("Film height"), 2, 0);

    m_HeightSpinner.setRange(ResMin, ResMax);
	m_GridLayout.addWidget(&m_HeightSpinner, 2, 1);
	
 	QObject::connect(&m_HeightSpinner, SIGNAL(valueChanged(int)), this, SLOT(SetHeight(int)));

	// Exposure
	m_GridLayout.addWidget(new QLabel("Exposure"), 3, 0);

	m_ExposureSlider.setOrientation(Qt::Horizontal);
	m_ExposureSlider.setRange(0.0f, 1.0f);
	m_GridLayout.addWidget(&m_ExposureSlider, 3, 1);

	m_ExposureSpinner.setRange(0.0f, 1.0f);
	m_GridLayout.addWidget(&m_ExposureSpinner, 3, 2);

 	QObject::connect(&m_ExposureSlider, SIGNAL(valueChanged(double)), &m_ExposureSpinner, SLOT(setValue(double)));
 	QObject::connect(&m_ExposureSlider, SIGNAL(valueChanged(double)), this, SLOT(SetExposure(double)));
 	QObject::connect(&m_ExposureSpinner, SIGNAL(valueChanged(double)), &m_ExposureSlider, SLOT(setValue(double)));

	gStatus.SetStatisticChanged("Camera", "Film", "", "", "");

	m_NoiseReduction.setText("Noise Reduction");
	m_GridLayout.addWidget(&m_NoiseReduction, 4, 1);

	QObject::connect(&m_NoiseReduction, SIGNAL(stateChanged(const int&)), this, SLOT(OnNoiseReduction(const int&)));

	QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	QObject::connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
}

void QFilmWidget::SetPresetType(const QString& PresetType)
{
	if (PresetType == "NTSC D-1 (video)")
	{
		m_Preset[0].SetPreset(720, 486);
		m_Preset[1].SetPreset(200, 135);
		m_Preset[2].SetPreset(360, 243);
		m_Preset[3].SetPreset(512, 346);
	}

	if (PresetType == "NTSC DV (video)")
	{
		m_Preset[0].SetPreset(720, 480);
		m_Preset[1].SetPreset(300, 200);
		m_Preset[2].SetPreset(360, 240);
		m_Preset[3].SetPreset(512, 341);
	}

	if (PresetType == "PAL (video)")
	{
		m_Preset[0].SetPreset(768, 576);
		m_Preset[1].SetPreset(180, 135);
		m_Preset[2].SetPreset(240, 180);
		m_Preset[3].SetPreset(480, 360);
	}

	if (PresetType == "PAL D-1 (video)")
	{
		m_Preset[0].SetPreset(720, 576);
		m_Preset[1].SetPreset(180, 144);
		m_Preset[2].SetPreset(240, 192);
		m_Preset[3].SetPreset(480, 384);
	}

	if (PresetType == "HDTV (video)")
	{
		m_Preset[0].SetPreset(1920, 1080);
		m_Preset[1].SetPreset(490, 270);
		m_Preset[2].SetPreset(1280, 720);
		m_Preset[3].SetPreset(320, 180);
	}
}

void QFilmWidget::SetPreset(QFilmResolutionPreset& Preset)
{
	m_WidthSpinner.setValue(Preset.GetWidth());
	m_HeightSpinner.setValue(Preset.GetHeight());
}

void QFilmWidget::SetWidth(const int& Width)
{
	gCamera.GetFilm().SetWidth(Width);
}

void QFilmWidget::SetHeight(const int& Height)
{
 	gCamera.GetFilm().SetHeight(Height);
}

void QFilmWidget::SetExposure(const double& Exposure)
{
	gCamera.GetFilm().SetExposure(Exposure);
}

void QFilmWidget::OnRenderBegin(void)
{
	m_WidthSpinner.blockSignals(true);
	m_WidthSpinner.setValue(gScene.m_Camera.m_Film.GetWidth());
	m_WidthSpinner.blockSignals(false);

	m_HeightSpinner.blockSignals(true);
	m_HeightSpinner.setValue(gScene.m_Camera.m_Film.GetHeight());
	m_HeightSpinner.blockSignals(false);
}

void QFilmWidget::OnRenderEnd(void)
{
//	gCamera.GetFilm().Reset();
}

void QFilmWidget::OnFilmChanged(const QFilm& Film)
{
	// Width
	m_WidthSpinner.blockSignals(true);
	m_WidthSpinner.setValue(Film.GetWidth());
	m_WidthSpinner.blockSignals(false);

	// Height
	m_HeightSpinner.blockSignals(true);
	m_HeightSpinner.setValue(Film.GetHeight());
	m_HeightSpinner.blockSignals(false);

	// Exposure
	m_ExposureSlider.setValue(Film.GetExposure(), true);
	m_ExposureSpinner.setValue(Film.GetExposure(), true);
}

void QFilmWidget::OnNoiseReduction(const int& ReduceNoise)
{
	gCamera.GetFilm().SetNoiseReduction(m_NoiseReduction.checkState());
}