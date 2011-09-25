#pragma once

#include "Geometry.h"
#include "Logger.h"

class QMargin
{
public:
	QMargin(void) :
		m_Left(0),
		m_Right(0),
		m_Top(0),
		m_Bottom(0)
	{
	}

	QMargin(const int& Left, const int& Right, const int& Top, const int& Bottom) :
		m_Left(Left),
		m_Right(Right),
		m_Top(Top),
		m_Bottom(Bottom)
	{
	}

	int		GetLeft(void)					{ return m_Left;		}
	void	SetLeft(const int& Left)		{ m_Left = Left;		}
	int		GetRight(void)					{ return m_Right;		}
	void	SetRight(const int& Right)		{ m_Right = Right;		}
	int		GetTop(void)					{ return m_Top;			}
	void	SetTop(const int& Top)			{ m_Top = Top;			}
	int		GetBottom(void)					{ return m_Bottom;		}
	void	SetBottom(const int& Bottom)	{ m_Bottom = Bottom;	}

private:
	int		m_Left;
	int		m_Right;
	int		m_Top;
	int		m_Bottom;
};

inline QString FormatVector(const Vec3f& Vector, const int& Precision = 2)
{
	return "[" + QString::number(Vector.x, 'f', Precision) + ", " + QString::number(Vector.y, 'f', Precision) + ", " + QString::number(Vector.z, 'f', Precision) + "]";
}

inline QString FormatVector(const Vec3i& Vector)
{
	return "[" + QString::number(Vector.x) + ", " + QString::number(Vector.y) + ", " + QString::number(Vector.z) + "]";
}

inline QString FormatSize(const Vec3f& Size, const int& Precision = 2)
{
	return QString::number(Size.x, 'f', Precision) + " x " + QString::number(Size.y, 'f', Precision) + " x " + QString::number(Size.z, 'f', Precision);
}

inline QString FormatSize(const Vec3i& Size)
{
	return QString::number(Size.x) + " x " + QString::number(Size.y) + " x " + QString::number(Size.z);
}