#pragma once

#include <QtGui>

#include "PresetsWidget.h"

class QLightingPresetsWidget : public QPresetsWidget
{
    Q_OBJECT

public:
    QLightingPresetsWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(200, 200); }

private slots:

protected:
};