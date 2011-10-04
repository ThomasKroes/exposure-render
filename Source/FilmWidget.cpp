
// Precompiled headers
#include "Stable.h"

#include "FilmWidget.h"
#include "RenderThread.h"
#include "Camera.h"

QFilmWidget::QFilmWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_WidthSlider(),
	m_WidthSpinner(),
	m_HeightSlider(),
	m_HeightSpinner(),
	m_ExposureSlider(),
	m_ExposureSpinner(),
	m_LockSizeCheckBox()
{
	setTitle("Film");
	setStatusTip("Film properties");
	setToolTip("Film properties");

	// Create grid layout
//	m_GridLayout.setColSpacing(0, 70);
	setLayout(&m_GridLayout);

	// Film width
	m_GridLayout.addWidget(new QLabel("Film width"), 0, 0);

	m_WidthSlider.setOrientation(Qt::Horizontal);
	m_WidthSlider.setRange(4, 2048);
	m_WidthSlider.setSingleStep(1);

	m_GridLayout.addWidget(&m_WidthSlider, 0, 1);
	
    m_WidthSpinner.setRange(4, 2048);
	m_GridLayout.addWidget(&m_WidthSpinner, 0, 2);
	
 	QObject::connect(&m_WidthSlider, SIGNAL(valueChanged(int)), &m_WidthSpinner, SLOT(setValue(int)));
 	QObject::connect(&m_WidthSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFilmWidth(int)));
 	QObject::connect(&m_WidthSpinner, SIGNAL(valueChanged(int)), &m_WidthSlider, SLOT(setValue(int)));
	QObject::connect(&m_WidthSlider, SIGNAL(sliderReleased()), this, SLOT(OnSetFilmWidth()));
	
	// Film height
	m_GridLayout.addWidget(new QLabel("Film height"), 2, 0);

	m_HeightSlider.setOrientation(Qt::Horizontal);
	m_HeightSlider.setRange(4, 2048);
	m_HeightSlider.setSingleStep(1);

	m_GridLayout.addWidget(&m_HeightSlider, 2, 1);
	
    m_HeightSpinner.setRange(0, 2048);
	m_GridLayout.addWidget(&m_HeightSpinner, 2, 2);
	
 	QObject::connect(&m_HeightSlider, SIGNAL(valueChanged(int)), &m_HeightSpinner, SLOT(setValue(int)));
 	QObject::connect(&m_HeightSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFilmHeight(int)));
 	QObject::connect(&m_HeightSpinner, SIGNAL(valueChanged(int)), &m_HeightSlider, SLOT(setValue(int)));
	QObject::connect(&m_HeightSlider, SIGNAL(sliderReleased()), this, SLOT(OnSetFilmHeight()));

	m_GridLayout.addWidget(&m_LockSizeCheckBox, 2, 3);

	QObject::connect(&m_LockSizeCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockFilmHeight(int)));

	// Exposure
	m_GridLayout.addWidget(new QLabel("Exposure"), 4, 0);

	m_ExposureSlider.setOrientation(Qt::Horizontal);
	m_ExposureSlider.setRange(0.0f, 1.0f);
	m_GridLayout.addWidget(&m_ExposureSlider, 4, 1);

	m_ExposureSpinner.setRange(0.0f, 1.0f);
	m_GridLayout.addWidget(&m_ExposureSpinner, 4, 2);

 	QObject::connect(&m_ExposureSlider, SIGNAL(valueChanged(double)), &m_ExposureSpinner, SLOT(setValue(double)));
 	QObject::connect(&m_ExposureSlider, SIGNAL(valueChanged(double)), this, SLOT(SetExposure(double)));
 	QObject::connect(&m_ExposureSpinner, SIGNAL(valueChanged(double)), &m_ExposureSlider, SLOT(setValue(double)));

	QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	QObject::connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
// 	QObject::connect(&gCamera.GetFilm(), SIGNAL(Changed(const QFilm&)), this, SLOT(OnFilmChanged(const QFilm&)));

	gStatus.SetStatisticChanged("Camera", "Film", "", "", "");
}

void QFilmWidget::LockFilmHeight(const int& Lock)
{
 	m_HeightSlider.setEnabled(!Lock);
 	m_HeightSpinner.setEnabled(!Lock);
 
 	if (Lock)
 	{
 		connect(&m_WidthSlider, SIGNAL(valueChanged(int)), &m_HeightSlider, SLOT(setValue(int)));
 		connect(&m_WidthSpinner, SIGNAL(valueChanged(int)), &m_HeightSpinner, SLOT(setValue(int)));
 
 		m_HeightSlider.setValue(m_WidthSlider.value());
 	}
 	else
 	{
 		disconnect(&m_WidthSlider, SIGNAL(valueChanged(int)), &m_HeightSlider, SLOT(setValue(int)));
 		disconnect(&m_WidthSpinner, SIGNAL(valueChanged(int)), &m_HeightSpinner, SLOT(setValue(int)));
 	}
}

void QFilmWidget::SetFilmWidth(const int& FilmWidth)
{
//	gScene.m_Camera.m_Film.m_Resolution.SetResX(FilmWidth);
//	gScene.m_DirtyFlags.SetFlag(FilmResolutionDirty);
	gCamera.GetFilm().SetWidth(FilmWidth);
}

void QFilmWidget::SetFilmHeight(const int& FilmHeight)
{
//	gScene.m_Camera.m_Film.m_Resolution.SetResY(FilmHeight);
//	gScene.m_DirtyFlags.SetFlag(FilmResolutionDirty);
 	gCamera.GetFilm().SetHeight(FilmHeight);
}

void QFilmWidget::SetExposure(const double& Exposure)
{
//	gCamera.GetFilm().SetExposure(Exposure);
}

void QFilmWidget::OnRenderBegin(void)
{
	gCamera.GetFilm().blockSignals(true);
	gCamera.GetFilm().SetWidth(gScene.m_Camera.m_Film.GetWidth());
	gCamera.GetFilm().SetHeight(gScene.m_Camera.m_Film.GetHeight());
	gCamera.GetFilm().blockSignals(false);
}

void QFilmWidget::OnRenderEnd(void)
{
//	gCamera.GetFilm().Reset();
}

void QFilmWidget::OnFilmChanged(const QFilm& Film)
{
	// Width
	m_WidthSlider.blockSignals(true);
	m_WidthSpinner.blockSignals(true);
	m_WidthSlider.setValue(Film.GetWidth());
	m_WidthSpinner.setValue(Film.GetWidth());
	m_WidthSlider.blockSignals(false);
	m_WidthSpinner.blockSignals(false);

	// Height
	m_HeightSlider.blockSignals(true);
	m_HeightSpinner.blockSignals(true);
	m_HeightSlider.setValue(Film.GetHeight());
	m_HeightSpinner.setValue(Film.GetHeight());
	m_HeightSlider.blockSignals(false);
	m_HeightSpinner.blockSignals(false);

	// Exposure
	m_ExposureSlider.setValue(Film.GetExposure(), true);
	m_ExposureSpinner.setValue(Film.GetExposure(), true);
}

void QFilmWidget::OnSetFilmWidth(void)
{
//	gCamera.GetFilm().SetWidth(m_WidthSlider.value());
}

void QFilmWidget::OnSetFilmHeight(void)
{
//	gCamera.GetFilm().SetHeight(m_HeightSlider.value());
}