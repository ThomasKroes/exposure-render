
#include "FilmWidget.h"
#include "RenderThread.h"
#include "Camera.h"

CFilmWidget::CFilmWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_WidthSlider(),
	m_WidthSpinBox(),
	m_HeightSlider(),
	m_HeightSpinBox(),
	m_LockSizeCheckBox()
{
	setTitle("Film");
	setStatusTip("Film properties");
	setToolTip("Film properties");

	setEnabled(false);

	// Create grid layout
	m_GridLayout.setColSpacing(0, 70);
	setLayout(&m_GridLayout);

	// Film width
	m_GridLayout.addWidget(new QLabel("Film width"), 0, 0);

	m_WidthSlider.setOrientation(Qt::Horizontal);
	m_WidthSlider.setRange(4, 1024);
	m_GridLayout.addWidget(&m_WidthSlider, 0, 1);
	
    m_WidthSpinBox.setRange(4, 1024);
	m_GridLayout.addWidget(&m_WidthSpinBox, 0, 2);
	
	connect(&m_WidthSlider, SIGNAL(valueChanged(int)), &m_WidthSpinBox, SLOT(setValue(int)));
	connect(&m_WidthSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFilmWidth(int)));
	connect(&m_WidthSpinBox, SIGNAL(valueChanged(int)), &m_WidthSlider, SLOT(setValue(int)));

	// Film height
	m_GridLayout.addWidget(new QLabel("Film height"), 2, 0);

	m_HeightSlider.setOrientation(Qt::Horizontal);
	m_HeightSlider.setRange(4, 1024);
	m_GridLayout.addWidget(&m_HeightSlider, 2, 1);
	
    m_HeightSpinBox.setRange(0, 1024);
	m_GridLayout.addWidget(&m_HeightSpinBox, 2, 2);
	
	m_GridLayout.addWidget(new QCheckBox("Lock Height"), 2, 3);

	connect(&m_HeightSlider, SIGNAL(valueChanged(int)), &m_HeightSpinBox, SLOT(setValue(int)));
	connect(&m_HeightSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFilmHeight(int)));
	connect(&m_HeightSpinBox, SIGNAL(valueChanged(int)), &m_HeightSlider, SLOT(setValue(int)));
	connect(&m_LockSizeCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockFilmHeight(int)));
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
	connect(&gCamera.GetFilm(), SIGNAL(Changed(const QFilm&)), this, SLOT(OnFilmChanged(const QFilm&)));
}

void CFilmWidget::LockFilmHeight(const int& Lock)
{
	m_HeightSlider.setEnabled(!Lock);
	m_HeightSpinBox.setEnabled(!Lock);

	if (Lock)
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

void CFilmWidget::SetFilmWidth(const int& FilmWidth)
{
	gCamera.GetFilm().SetWidth(FilmWidth);
}

void CFilmWidget::SetFilmHeight(const int& FilmHeight)
{
	gCamera.GetFilm().SetHeight(FilmHeight);
}

void CFilmWidget::OnRenderBegin(void)
{
	OnFilmChanged(gCamera.GetFilm());
}

void CFilmWidget::OnRenderEnd(void)
{
	gCamera.GetFilm().Reset();
}

void CFilmWidget::OnFilmChanged(const QFilm& Film)
{
	// Width
	m_WidthSlider.blockSignals(true);
	m_WidthSpinBox.blockSignals(true);
	m_WidthSlider.setValue(Film.GetWidth());
	m_WidthSpinBox.setValue(Film.GetWidth());
	m_WidthSlider.blockSignals(false);
	m_WidthSpinBox.blockSignals(false);

	// Height
	m_HeightSlider.blockSignals(true);
	m_HeightSpinBox.blockSignals(true);
	m_HeightSlider.setValue(Film.GetHeight());
	m_HeightSpinBox.setValue(Film.GetHeight());
	m_HeightSlider.blockSignals(false);
	m_HeightSpinBox.blockSignals(false);
}