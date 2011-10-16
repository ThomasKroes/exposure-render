/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "StatisticsDockWidget.h"
#include "MainWindow.h"

class QGraphicsWidget2 : public QGraphicsWidget
{

};

class QTfCanvas : public QWidget
{
public:
	void paintEvent(QPaintEvent * pe)
	{

		QPainter Painter(this);

		if (isEnabled())
			Painter.fillRect(rect(), QBrush(QColor(230, 230, 230)));
		else
			Painter.fillRect(rect(), QBrush(QColor(200, 200, 200)));
	}
};

QStatisticsDockWidget::QStatisticsDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_MainLayout(),
	m_StatisticsWidget()
{
	setWindowTitle("Statistics");
	setToolTip("<img src=':/Images/application-list.png'><div>Rendering statistics</div>");
	setWindowIcon(GetIcon("application-list"));

	setWidget(&m_StatisticsWidget);
}