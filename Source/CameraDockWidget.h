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
	QGridLayout*	m_pGridLayout;
	QLabel*			m_pFilmWidthLabel;
	QSlider*		m_pFilmWidthSlider;
	QSpinBox*		m_pFilmWidthSpinBox;
	QLabel*			m_pFilmHeightLabel;
	QSlider*		m_pFilmHeightSlider;
	QSpinBox*		m_pFilmHeightSpinBox;
	QCheckBox*		m_pLockFilmHeightCheckBox;
};

class CApertureWidget : public QGroupBox
{
    Q_OBJECT

public:
    CApertureWidget(QWidget* pParent = NULL);

private slots:
	void SetAperture(const int& Aperture);

private:
	QGridLayout*	m_pGridLayout;
	QLabel*			m_pApertureSizeLabel;
	QSlider*		m_pApertureSizeSlider;
	QSpinBox*		m_pApertureSizeSpinBox;
};

class CProjectionWidget : public QGroupBox
{
    Q_OBJECT

public:
    CProjectionWidget(QWidget* pParent = NULL);

private slots:
	void SetFieldOfView(const int& FieldOfView);

private:
	QGridLayout*	m_pGridLayout;
	QLabel*			m_pFieldOfViewLabel;
	QSlider*		m_pFieldOfViewSlider;
	QSpinBox*		m_pFieldOfViewSpinBox;
};

class CFocusWidget : public QGroupBox
{
    Q_OBJECT

public:
    CFocusWidget(QWidget* pParent = NULL);

private slots:
	void SetFocusType(const int& FocusType);
	void SetFocalDistance(const int& FocalDistance);

private:
	QGridLayout*	m_pGridLayout;
	QLabel*			m_pFocusTypeLabel;
	QComboBox*		m_pFocusTypeComboBox;
	QLabel*			m_pFocalDistanceLabel;
	QSlider*		m_pFocalDistanceSlider;
	QSpinBox*		m_pFocalDistanceSpinBox;
};

class CCameraWidget : public QWidget
{
    Q_OBJECT

public:
    CCameraWidget(QWidget* pParent = NULL);

private:
	QVBoxLayout*		m_pMainLayout;
	CFilmWidget*		m_pFilmWidget;
	CApertureWidget*	m_pApertureWidget;
	CProjectionWidget*	m_pProjectionWidget;
	CFocusWidget*		m_pFocusWidget;
};

class CCameraDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    CCameraDockWidget(QWidget* pParent = NULL);

private:
	CCameraWidget*	m_pCameraWidget;
};