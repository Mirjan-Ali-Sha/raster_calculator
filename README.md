# Raster Calculator Plugin for QGIS

**Version:** 1.0.0  
**Author:** Mirjan Ali Sha  
**Email:** mastools.help@gmail.com  
**Category:** Raster  
**Tags:** raster, calculator, map algebra, raster calculator, math, conditional, processing, analysis

## Description

A powerful map algebra raster calculator similar to ArcGIS Raster Calculator. Part of **MAS Raster Processing** tools.

This plugin provides a robust environment for complex raster analysis using mathematical and logical operators. It is designed to offer a familiar experience for users accustomed to ArcGIS-style Map Algebra.

## Features

- **Interactive Expression Builder:** User-friendly interface with a numeric keypad and operator buttons.
- **Multi-layer Support:** Easily add multiple raster layers to your calculations.
- **Conditional Functions:** Support for `Con` (Condition), `Pick`, and `SetNull` operations.
- **Math Functions:** Comprehensive set of mathematical functions including `Abs`, `Exp`, `Log`, `Sqrt`, `Sin`, `Cos`, `Tan`, and more.
- **Logical Operators:** Full suite of comparison and logical operators for boolean raster creation.
- **Processing Integration:** Available as both a standalone GUI tool and a QGIS Processing algorithm, making it chainable in models.

## Installation

1.  Open QGIS.
2.  Go to **Plugins** > **Manage and Install Plugins...**
3.  Search for "Raster Calculator" (or install from ZIP if distributed manually).
4.  Click **Install Plugin**.

## Usage

1.  Open the tool from **Raster** > **MAS Raster Processing** > **Raster Calculator**.
2.  Available raster layers will be listed in the "Raster Bands" panel. Double-click a layer to add it to the expression box.
3.  Use the calculator buttons (Operators, Math, Conditional) to build your Map Algebra expression.
    *   Example: `Con("dem@1" > 1000, 1, 0)`
4.  Specify the **Output layer** path.
5.  Click **Calculate**.

## Support

- **Bug Tracker:** [GitHub Issues](https://github.com/Mirjan-Ali-Sha/raster_calculator/issues)
- **Repository:** [GitHub Repository](https://github.com/Mirjan-Ali-Sha/raster_calculator)
- **Wiki/Homepage:** [Documentation](https://github.com/Mirjan-Ali-Sha/raster_calculator/wiki)

## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
