#!/usr/bin/env python
# Filename: bhm_color.py
# Created by Brian Meanley (brian <DOT> meanley <AT> gmail <DOT> com

import struct
import math

class ColorConvert:
	"""Useful functions for converting colors to and from various color spaces"""

	def cie_to_XYZ(self,x,y,Y):
		"""Converts CIE xyY coordinates to equivalent XYZ tristimulus values"""
		Y = float(Y)
		X = ((x*Y)/y)
		Z = (((1-x-y)*Y)/y)
		XYZ = [X, Y, Z]
		return XYZ

	def XYZ_to_RGB(self,XYZ):
		"""
		Converts colors from a XYZ tristimulus triplet to the linear RGB equivalent
		Taken from http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
		Uses this matrix: [R]   [ 3.2404542 -1.5371385 -0.4985314][X]
		                  [G] = [-0.9692660  1.8760108  0.0415560][Y]
		                  [B]   [ 0.0556434 -0.2040259  1.0572252][Z]
		"""
		X = XYZ[0]
		Y = XYZ[1]
		Z = XYZ[2]
		r = (3.2404542 * X) + (-1.5371385 * Y) + (-0.4985314 * Z)
		g = (-0.9692660 * X) + (1.8760108 * Y) + (0.0415560 * Z)
		b = (0.0556434 * X) + (-0.2040259 * Y) + (1.0572252 * Z)

		rgb = [r, g, b]
		return rgb

	def linToSRGB(self,cLin):
		"""Converts linear RGB values to sRGB"""
		cSRGB = []
		for c in cLin:
			if c <= 0.0031308:
				c *= 12.92
			else:
				c = (1+0.055) * (c**(1/2.4) - 0.055)
			cSRGB.append(c)	
		return cSRGB

	def sRGBtoLin(self,cSRGB):
		"""Converts sRGB values to linear"""
		cLin = []
		for c in cSRGB:
			if c <= 0.04045:
				c /=12.92
			else:
				c = ((c + 0.055)/(1+0.055))**2.4
			cLin.append(c)
		return cLin

	def rgbNorm(self,rgb):
		"""Normalizes RGB values into the range of 0-1"""
		maxRGB = max(rgb[0], rgb[1], rgb[2])
		if maxRGB > 1:
			rgbNorm = [i/maxRGB for i in rgb]
		else:
			rgbNorm = rgb
		return rgbNorm

	def rgbScaled(self,rgb):
		"""Scales normalized float RGB values to a range of 0-255"""
		if rgb[0] < 0:
			Rx = 0
		else: Rx = int(rgb[0]*255)
		if rgb[1] < 0:
			Gx = 0
		else: 	Gx = int(rgb[1]*255)
		if rgb[2] < 0:
			Bx = 0
		else: 	Bx = int(rgb[2]*255)
		rgbScaled = [Rx,Gx,Bx]
		return rgbScaled

	def rgbToHex(self,rgb):
		"""returns a hex string based off rgb values. RGB values must be scaled to 0-255"""
		hex = '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])
		return hex
	
	def rgbToFloat(self,triple):
		self.triple = triple
		r = self.triple[0]/float(256)
		g = self.triple[1]/float(256)
		b = self.triple[2]/float(256)
		rgb = (r,g,b)
		return rgb


class Blackbody(ColorConvert):
	"""
	Various functions used to calculate the corresponding CIE xy coordinates from a given Kelvin temperature.
	This is a subclass of the ColorConvert class, as this class is much more specific and also relies on
	many of the methods of that class to return useful information (color values) to the user.
	The steps taken to make these conversion are:

	#************************************************
	# Calculate Kelvin -> cie xy coordinates.
	* Let the user define the Y (intensity value).
	# Convert cie xyY -> XYZ tristimulus values.
	# Convert XYZ -> colorspaces (RGB in this case).
	#************************************************

	I am using an aproximation of the Planckian Locus discussed by Kim et al in:
	US patent 7024034, Kim et al., "Color Temperature Conversion System and Method Using the Same", issued 2006-04-04
	The formula can be found here: http://en.wikipedia.org/wiki/Planckian_locus#Approximation
	This is an approximation of the curve, and cannot account for values below 1667k.
	"""

	def validateTemp(self,temp,dict=None):
		"""validates the given temperature to make sure it is either in the range of 1000-25000, among
		the given presets.  Really, the range should be limited to 1667-25000, as our equations only work in that range,
		but we are allowing for a little bit of flexibility, and clamping the values later."""
		self.temp = temp
		self.dict = dict
		if self.temp == '':
			print('Please enter valid temperature in the range of 1000-25000.')
			return
		else:
			try:
				int(self.temp) + 1
				if int(self.temp) in range(1000, 25001):
					return self.temp
				else:
					print('Please enter valid temperature in the range of 1000-25000.')
					return False
			except:
				try:
					if self.dict != None:
						self.temp = int(self.dict[self.temp])
						int(self.temp) + 1
					else: return False 
				except:
					print('Please enter a valid preset or temperature.')
					return False
		return self.temp

	def tempOffset(self,wbk,*args):
		"""This function takes a specified white balance and light temperature, and offsets the light temperature
		   by the difference of the white balance and the default white point (6504k).  This is a hack, and should
		   be replaced by a true Chromatic Adaptation transform at some point.  This current method means that returned
		   values are really only truely accurate when set to the default value of 6504k. An optional offset (os) can
		   also be supplied to manually push the relative light temperature one direction or the other."""
		self.wbkOS = abs(self.wbk - 6504)
		self.wbk = wbk
		if self.wbk < 6504:
			self.ltkOS = self.ltk + self.wbkOS
			# offset temp.  See note above about his hack...
			self.ltkOS = self.ltkOS - int(self.os)
		elif self.wbk > 6504:
			self.ltkOS = self.ltk - self.wbkOS
			# offset temp.  See note above about his hack...
			self.ltkOS = self.ltkOS - int(self.os)
		else:
			self.ltkOS = self.ltk
			# offset temp.  See note above about his hack...
			self.ltkOS = self.ltkOS - int(self.os)
		return self.ltkOS
	
	def tempClamp(self,ltkOS,*args):
		"""This clamps any given values (in degrees Kelvin) to a range that works with our
		   equations.  This range is limited between 1667k-25000k. Any values outside of this
		   range will either be clamped or should return errors."""
		self.ltkOS = ltkOS

		if self.ltkOS < 1667:
			self.ltkOS = 1667
			return self.ltkOS
		elif self.ltkOS > 25000:
			self.ltkOS = 25000
			return 20000
		else:
			return self.ltkOS

	def cie_x_Calc(self,T,*args):
		"""This function will calculate the 'x' component of the xyY color coordinates"""
		self.T = T
		if self.T in range(1667, 4000):
			a = 0.2661239
			b = 0.2343580
			c = 0.8776956
			d = 0.179910
			x = (((-a*(10.0**9/self.T**3))-b*(10.0**6/self.T**2))+c*(10.0**3/self.T))+d
		elif self.T in range(4000, 25001):
			a = 3.0258469
			b = 2.1070379
			c = 0.2226347
			d = 0.240390
			x = (((-a*(10.0**9/self.T**3))+b*(10.0**6/self.T**2))+c*(10.0**3/self.T))+d
		else:
			raise ValueError, 'That number is out of range.'
		return x

	def cie_y_Calc(self,T, x):
		"""This function will calculate the 'y' component of the xyY color coordinates.
		   This function requires that the 'x' component be computed first"""
		self.T = T
		self.x = x
		if self.T in range(1667, 2222):
			a = 1.10638140
			b = 1.34811020
			c = 2.18555832
			d = 0.20219683
			y = (((-a*(self.x**3))-b*(self.x**2))+c*(self.x))-d
		elif self.T in range(2222, 4000):
			a = 0.9549476
			b = 1.37418593
			c = 2.09137015
			d = 0.16748867
			y = (((-a*(self.x**3))-b*(self.x**2))+c*(self.x))-d
		elif self.T in range(4000, 25001):
			a = 3.0817580
			b = 5.87338670
			c = 3.75112997
			d = 0.37001483
			y = (((a*(self.x**3))-b*(self.x**2))+c*(self.x))-d
		else:
			raise ValueError, 'That number is out of range.'
		return y

	def calculate(self,ltk,wbk=6504,Y=1,os=0,wb_dict=None,light_dict=None):
		print ('\n************************************')
		# cast values to integers
		self.wbk = wbk
		self.ltk = ltk
		self.Y = Y
		self.os = os
		self.wb_dict = wb_dict
		self.lt_dict = light_dict

		self.wbk = self.validateTemp(self.wbk,dict=self.wb_dict)
		if not self.wbk:
			print('Something is up with that white balance... please check it and try again...')
			return False
		self.ltk = self.validateTemp(self.ltk,dict=self.lt_dict)
		if not self.ltk:
			print('Something is up with that light temperature... please check it and try again...')
			return False

		self.wbk = int(self.wbk) # cast to integer values
		self.ltk = int(self.ltk)

		self.ltkOS = self.tempOffset(self.wbk, self.ltk, self.os)
		self.ltkOS = self.tempClamp(self.ltkOS)
		self.ltkOS = int(self.ltkOS)

		print('Your white balance has been set to: %s' % self.wbk)
		print('Your light temperature is: %s' % self.ltk)
		print('Your offset light temp. is: %s' % self.ltkOS)

		# CIE xyY aproximation
		self.x = self.cie_x_Calc(self.ltkOS)
		self.y = self.cie_y_Calc(self.ltkOS, self.x)
		print ('Your CIE xy coordinates are: %.4f %.4f') % (self.x, self.y)

		# xyY to XYZ convert
		self.XYZ = self.cie_to_XYZ(self.x,self.y,self.Y)
		print ('Your XYZ values are: %f %f %f') % (self.XYZ[0] * 100, self.XYZ[1] * 100, self.XYZ[2] * 100)

		# XYZ to RGB convert
		self.rgb = self.XYZ_to_RGB(self.XYZ)

		# Normalize values if any channel is greater than 1.0
		rgbN_long = self.rgbNorm(self.rgb)
		rgbN = [round(color,4) for color in rgbN_long]
		print ('Your normalized RGB values are: %s') % str(rgbN).replace(',','')
		print ('************************************\n')
		return rgbN
