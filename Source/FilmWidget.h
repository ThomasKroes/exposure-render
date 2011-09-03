#pragma once

#include <QtGui>

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
	QGridLayout	m_GridLayout;
	QSlider		m_FilmWidthSlider;
	QSpinBox	m_FilmWidthSpinBox;
	QSlider		m_FilmHeightSlider;
	QSpinBox	m_FilmHeightSpinBox;
	QCheckBox	m_LockFilmHeightCheckBox;
};