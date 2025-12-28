# -*- coding: utf-8 -*-
"""
Raster Calculator Plugin
"""


def classFactory(iface):
    """Load RasterCalculator class from file raster_calculator.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .raster_calculator import RasterCalculator
    return RasterCalculator(iface)
