#pragma once

#include "Controls.h"
#include "Film.h"

class QFilmWidget : public QGroupBox
{
    Q_OBJECT

public:
    QFilmWidget(QWidget* pParent = NULL);

public slots:
	void LockFilmHeight(const int& State);
	void SetFilmWidth(const int& FilmWidth);
	void SetFilmHeight(const int& FilmHeight);
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnFilmChanged(const QFilm& Film);

private:
	QGridLayout	m_GridLayout;
	QSlider		m_WidthSlider;
	QSpinBox	m_WidthSpinBox;
	QSlider		m_HeightSlider;
	QSpinBox	m_HeightSpinBox;
	QCheckBox	m_LockSizeCheckBox;
};