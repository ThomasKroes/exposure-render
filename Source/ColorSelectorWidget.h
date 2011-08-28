#pragma once

#include <QtGui>

class QColorFavoritesWidget : public QWidget
{
    Q_OBJECT

public:
    QColorFavoritesWidget(QWidget* pParent = NULL);

private:
	QGridLayout*		m_pMainLayout;
	QTableWidget*		m_pTableWidget;
};

class QColorPresetsWidget : public QWidget
{
    Q_OBJECT

public:
    QColorPresetsWidget(QWidget* pParent = NULL);

private:
	QGridLayout*		m_pMainLayout;
	QTableWidget*		m_pTableWidget;
};

class QColorSelectorWidget : public QGroupBox
{
    Q_OBJECT

public:
    QColorSelectorWidget(QWidget* pParent = NULL);

private:
	QGridLayout*		m_pMainLayout;
	QTabWidget*			m_pColorsTab;
};