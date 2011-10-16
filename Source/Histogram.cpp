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

#include "Histogram.h"

QHistogram gHistogram;

QHistogram::QHistogram(QObject* pParent /*= NULL*/) :
	QObject(pParent),
	m_Enabled(false),
	m_Bins(),
	m_Max(0),
	m_pPixMap(NULL)
{
}

QHistogram::QHistogram(const QHistogram& Other)
{
	*this = Other;
};

QHistogram& QHistogram::operator=(const QHistogram& Other)
{
	m_Enabled	= Other.m_Enabled;
	m_Bins		= Other.m_Bins;
	m_Max		= Other.m_Max;

	return *this;
}

bool QHistogram::GetEnabled(void) const
{
	return m_Enabled;
}

void QHistogram::SetEnabled(const bool& Enabled)
{
	m_Enabled = Enabled;

	// Inform others that the histogram has changed
	emit HistogramChanged();
}

QList<int>& QHistogram::GetBins(void)
{
	return m_Bins;
}

void QHistogram::SetBins(const QList<int>& Bins)
{
	m_Bins = Bins;

	// Inform others that the histogram has changed
	emit HistogramChanged();
}

void QHistogram::SetBins(const int* pBins, const int& NoBins)
{
	// Clear the bin list
	m_Bins.clear();

	m_Max = 0;

	for (int i = 0; i < NoBins; i++)
	{
		if (pBins[i] > GetMax())
			m_Max = pBins[i];
	}

	for (int i = 0; i < NoBins; i++)
		m_Bins.append(pBins[i]);

	m_Enabled = true;

	// Inform others that the histogram has changed
	emit HistogramChanged();
}

void QHistogram::CreatePixMap(void)
{
	
	QRect Rect(0, 0, 500, 500);

	QPainter Painter;

	m_pPixMap = new QPixmap();

	Painter.begin(m_pPixMap);

	Painter.drawRect(Rect);

	Painter.end();

		/*
	QPolygonF Polygon;

	QLinearGradient LinearGradient;

	LinearGradient.setStart(0, rect().bottom());
	LinearGradient.setFinalStop(0, rect().top());

	QGradientStops GradientStops;

	GradientStops.append(QGradientStop(0, QColor::fromHsl(0, 100, 150, 0)));
	GradientStops.append(QGradientStop(1, QColor::fromHsl(0, 100, 150, 255)));

	LinearGradient.setStops(GradientStops);
	
	// Set the gradient stops
	for (int i = 0; i < GetBins().size(); i++)
	{
		// Compute polygon point in scene coordinates
		QPointF ScenePoint = TransferFunctionToScene(QPointF(i, logf((float)gHistogram.GetBins()[i]) / logf(1.5f * (float)gHistogram.GetMax())));

		if (i == 0)
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}

		Polygon.append(ScenePoint);

		if (i == (gHistogram.GetBins().size() - 1))
		{
			QPointF CenterCopy = ScenePoint;

			CenterCopy.setY(rect().height());

			Polygon.append(CenterCopy);
		}
	}

	Painter.end();
	*/
}

QPixmap* QHistogram::GetPixMap(void)
{
	return m_pPixMap;
}

int QHistogram::GetMax(void) const
{
	return m_Max;
}

void QHistogram::SetMax(const int& Max)
{
	m_Max = Max;

	// Inform others that the histogram has changed
	emit HistogramChanged();
}

void QHistogram::Reset(void)
{
	m_Enabled = false;
	m_Bins.clear();
	m_Max = 0;

	// Inform others that the histogram has changed
	emit HistogramChanged();
}