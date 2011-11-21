/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "GridItem.h"

QGridItem::QGridItem(QGraphicsItem* pParent) :
	QGraphicsRectItem(pParent),
	m_BrushEnabled(QBrush(QColor::fromHsl(0, 0, 170))),
	m_BrushDisabled(QBrush(QColor::fromHsl(0, 0, 210))),
	m_PenEnabled(QPen(QColor::fromHsl(0, 0, 170), 0.1)),
	m_PenDisabled(QPen(QColor::fromHsl(0, 0, 190))),
	m_NumY(10),
	m_Font()
{
	m_Font.setFamily("Arial");
	m_Font.setPointSize(6);

	setAcceptHoverEvents(true);
}

QGridItem::QGridItem(const QGridItem& Other)
{
	*this = Other;
}

QGridItem& QGridItem::operator=(const QGridItem& Other)
{
	m_BrushEnabled	= Other.m_BrushEnabled;
	m_BrushDisabled	= Other.m_BrushDisabled;
	m_PenEnabled	= Other.m_PenEnabled;
	m_PenDisabled	= Other.m_PenDisabled;

	return *this;
}

void QGridItem::paint(QPainter* pPainter, const QStyleOptionGraphicsItem* pOption, QWidget* pWidget)
{
	/*
	pPainter->setRenderHint(QPainter::Antialiasing, false);

	if (isEnabled())
	{
		pPainter->setBrush(m_BrushEnabled);
		pPainter->setPen(m_PenEnabled);
	}
	else
	{
		pPainter->setBrush(m_BrushDisabled);
		pPainter->setPen(m_PenDisabled);
	}

	pPainter->setFont(m_Font);

	const float Width = 25.0f;
	const float Height = 18.0f;

	const float DY = rect().height() / (float)m_NumY;

	for (int i = 0; i < m_NumY + 1; i++)
	{
		if (i > 0 && i  < m_NumY)
			pPainter->drawLine(QPointF(0, i * DY), QPointF(rect().width(), i * DY));

		pPainter->drawLine(QPointF(-2, i * DY), QPointF(0, i * DY));
		pPainter->drawText(QRectF(-Width - 5, (-0.5f * Height) + i * DY, Width, Height), Qt::AlignVCenter | Qt::AlignRight, QString::number((10 - i) * 10.0f) + "%");
	}

	const float DX = 500.0f;

	const int NumNegX = fabs(gScene.m_IntensityRange.GetMin()) / DX;

	for (int i = 1; i < NumNegX; i++)
	{
		const float Intensity = i * -DX;
		const float NormalizedIntensity = ((Intensity) - gScene.m_IntensityRange.GetMin()) / gScene.m_IntensityRange.GetRange();

		float X = NormalizedIntensity * rect().width();

		pPainter->drawLine(QPointF(X, 0), QPointF(X, rect().height()));
		pPainter->drawText(QRectF(X - (0.5f * Width), rect().height(), Width, Height), Qt::AlignCenter, QString::number(Intensity));
	}

	const int NumPosX = fabs(gScene.m_IntensityRange.GetMax()) / DX;

	for (int i = 0; i < NumPosX; i++)
	{
		const float Intensity = i * DX;
		const float NormalizedIntensity = ((Intensity) - gScene.m_IntensityRange.GetMin()) / gScene.m_IntensityRange.GetRange();

		float X = NormalizedIntensity * rect().width();

		pPainter->drawLine(QPointF(X, 0), QPointF(X, rect().height() + 2.0f));
		pPainter->drawText(QRectF(X - (0.5f * Width), rect().height(), Width, Height), Qt::AlignCenter, QString::number(Intensity));
	}

	pPainter->drawLine(QPointF(0, rect().height()), QPointF(0, rect().height() + 2.0f));
	pPainter->drawLine(QPointF(rect().width(), rect().height()), QPointF(QPointF(rect().width(), rect().height() + 2.0f)));
	pPainter->drawText(QRectF(0 - (0.5f * Width), rect().height(), Width, Height), Qt::AlignCenter, QString::number(gScene.m_IntensityRange.GetMin()));
	pPainter->drawText(QRectF(rect().width() - (0.5f * Width), rect().height(), Width, Height), Qt::AlignCenter, QString::number(gScene.m_IntensityRange.GetMax()));
	*/
}