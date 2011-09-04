#pragma once

#include <QtGui>

#include "Controls.h"

class CFilmWidget : public QGroupBox
{
    Q_OBJECT

public:
    CFilmWidget(QWidget* pParent = NULL);

private slots:
	void LockFilmHeight(const int& State);
	void SetFilmWidth(const int& FilmWidth);
	void SetFilmHeight(const int& FilmHeight);

private:
	QGridLayout		m_GridLayout;
	QDoubleSlider	m_FilmWidthSlider;
	QSpinBox		m_FilmWidthSpinBox;
	QDoubleSlider	m_FilmHeightSlider;
	QSpinBox		m_FilmHeightSpinBox;
	QCheckBox		m_LockFilmHeightCheckBox;
};