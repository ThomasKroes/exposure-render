#pragma once

#include <QtGui>

#include "PresetsWidget.h"

class QLightingPresetsWidget : public QPresetsWidget
{
    Q_OBJECT

public:
    QLightingPresetsWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(200, 200); }

	virtual void LoadPresetsFromFile(const bool& ChoosePath = false);
	virtual void SavePresetsToFile(const bool& ChoosePath = false);

private slots:

protected:
};