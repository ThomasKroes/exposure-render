
#include "FilmWidget.h"
#include "MainWindow.h"
#include "Scene.h"

CFilmWidget::CFilmWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_FilmWidthSlider(),
	m_FilmWidthSpinBox(),
	m_FilmHeightSlider(),
	m_FilmHeightSpinBox(),
	m_LockFilmHeightCheckBox()
{
	setTitle("Film");
	setStatusTip("Film properties");
	setToolTip("Film properties");

	// Create grid layout
	m_GridLayout.setColSpacing(0, 70);
	setLayout(&m_GridLayout);

	// Film width
	m_GridLayout.addWidget(new QLabel("Film width"), 0, 0);

	m_FilmWidthSlider.setOrientation(Qt::Orientation::Horizontal);
	m_FilmWidthSlider.setRange(4, 1024);
	m_GridLayout.addWidget(&m_FilmWidthSlider, 0, 1);
	
    m_FilmWidthSpinBox.setRange(4, 1024);
	m_GridLayout.addWidget(&m_FilmWidthSpinBox, 0, 2);
	
	connect(&m_FilmWidthSlider, SIGNAL(valueChanged(int)), &m_FilmWidthSpinBox, SLOT(setValue(int)));
	connect(&m_FilmWidthSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFilmWidth(int)));
	connect(&m_FilmWidthSpinBox, SIGNAL(valueChanged(int)), &m_FilmWidthSlider, SLOT(setValue(int)));

	// Film height
	m_GridLayout.addWidget(new QLabel("Film height"), 2, 0);

	m_FilmHeightSlider.setOrientation(Qt::Orientation::Horizontal);
	m_FilmHeightSlider.setRange(4, 1024);
	m_GridLayout.addWidget(&m_FilmHeightSlider, 2, 1);
	
    m_FilmHeightSpinBox.setRange(0, 1024);
	m_GridLayout.addWidget(&m_FilmHeightSpinBox, 2, 2);
	
	m_GridLayout.addWidget(new QCheckBox("Lock Height"), 2, 3);

	connect(&m_FilmHeightSlider, SIGNAL(valueChanged(int)), &m_FilmHeightSpinBox, SLOT(setValue(int)));
	connect(&m_FilmHeightSlider, SIGNAL(valueChanged(int)), this, SLOT(SetFilmHeight(int)));
	connect(&m_FilmHeightSpinBox, SIGNAL(valueChanged(int)), &m_FilmHeightSlider, SLOT(setValue(int)));
	connect(&m_LockFilmHeightCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockFilmHeight(int)));
}

void CFilmWidget::LockFilmHeight(const int& Lock)
{
//	m_pFilmHeightLabel->setEnabled(!Lock);
	m_FilmHeightSlider.setEnabled(!Lock);
	m_FilmHeightSpinBox.setEnabled(!Lock);

	if (Lock)
	{
		connect(&m_FilmWidthSlider, SIGNAL(valueChanged(int)), &m_FilmHeightSlider, SLOT(setValue(int)));
		connect(&m_FilmWidthSpinBox, SIGNAL(valueChanged(int)), &m_FilmHeightSpinBox, SLOT(setValue(int)));

		m_FilmHeightSlider.setValue(m_FilmWidthSlider.value());
	}
	else
	{
		disconnect(&m_FilmWidthSlider, SIGNAL(valueChanged(int)), &m_FilmHeightSlider, SLOT(setValue(int)));
		disconnect(&m_FilmWidthSpinBox, SIGNAL(valueChanged(int)), &m_FilmHeightSpinBox, SLOT(setValue(int)));
	}
}

void CFilmWidget::SetFilmWidth(const int& FilmWidth)
{
	if (!Scene())
		return;

	Scene()->m_Camera.m_Film.m_Resolution.m_XY.x = FilmWidth;

	// Flag the film resolution as dirty, this will restart the rendering
	Scene()->m_DirtyFlags.SetFlag(FilmResolutionDirty);
}

void CFilmWidget::SetFilmHeight(const int& FilmHeight)
{
	if (!Scene())
		return;

	Scene()->m_Camera.m_Film.m_Resolution.m_XY.y = FilmHeight;

	// Flag the film resolution as dirty, this will restart the rendering
	Scene()->m_DirtyFlags.SetFlag(FilmResolutionDirty);
}