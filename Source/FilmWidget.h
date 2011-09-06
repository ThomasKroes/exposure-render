#pragma once

#include <QtGui>

#include "Controls.h"

class CFilmWidget : public QGroupBox
{
    Q_OBJECT

public:
    CFilmWidget(QWidget* pParent = NULL);

public slots:
	void LockFilmHeight(const int& State);
	void SetFilmWidth(const int& FilmWidth);
	void SetFilmHeight(const int& FilmHeight);
	void OnRenderBegin(void);
	void OnRenderEnd(void);

private:
	QGridLayout		m_GridLayout;
	QSlider			m_FilmWidthSlider;
	QSpinBox		m_FilmWidthSpinBox;
	QSlider			m_FilmHeightSlider;
	QSpinBox		m_FilmHeightSpinBox;
	QCheckBox		m_LockFilmHeightCheckBox;
};