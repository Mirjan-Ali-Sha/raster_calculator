# -*- coding: utf-8 -*-
"""
Raster Calculator Dialog - Compact UI with Function Lists
"""

import os
import re
import numpy as np
from osgeo import gdal
from qgis.PyQt import QtCore, QtWidgets, uic
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.core import QgsProject, QgsRasterLayer, QgsMessageLog, Qgis

FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'raster_calculator_dialog.ui'))


# Function definitions organized by category
FUNCTION_CATEGORIES = {
    'Conditional': [
        ('Con', 'Con(condition, true_val, false_val)'),
        ('Pick', 'Pick(value, opt1, opt2, ...)'),
        ('SetNull', 'SetNull(condition, false_val)'),
    ],
    'Math': [
        ('Abs', 'Abs(x)'),
        ('Exp', 'Exp(x) - e^x'),
        ('Exp10', 'Exp10(x) - 10^x'),
        ('Exp2', 'Exp2(x) - 2^x'),
        ('Float', 'Float(x) - Convert to float'),
        ('Int', 'Int(x) - Truncate to integer'),
        ('Ln', 'Ln(x) - Natural log'),
        ('Log10', 'Log10(x) - Base-10 log'),
        ('Log2', 'Log2(x) - Base-2 log'),
        ('Mod', 'Mod(x, y) - Remainder'),
        ('Power', 'Power(base, exp)'),
        ('Round', 'Round(x)'),
        ('RoundDown', 'RoundDown(x) - Floor'),
        ('RoundUp', 'RoundUp(x) - Ceiling'),
        ('Square', 'Square(x) - x²'),
        ('SquareRoot', 'SquareRoot(x) - √x'),
    ],
    'Trigonometric': [
        ('Sin', 'Sin(x)'),
        ('Cos', 'Cos(x)'),
        ('Tan', 'Tan(x)'),
        ('ASin', 'ASin(x) - Arc sine'),
        ('ACos', 'ACos(x) - Arc cosine'),
        ('ATan', 'ATan(x) - Arc tangent'),
        ('ATan2', 'ATan2(y, x) - Two-arg arctan'),
        ('SinH', 'SinH(x) - Hyperbolic sine'),
        ('CosH', 'CosH(x) - Hyperbolic cosine'),
        ('TanH', 'TanH(x) - Hyperbolic tangent'),
        ('ASinH', 'ASinH(x) - Inverse hyp. sine'),
        ('ACosH', 'ACosH(x) - Inverse hyp. cosine'),
        ('ATanH', 'ATanH(x) - Inverse hyp. tangent'),
    ],
    'Logical/Other': [
        ('Diff', 'Diff(x, y) - Absolute difference'),
        ('InList', 'InList(value, v1, v2, ...)'),
        ('IsNull', 'IsNull(x) - Check NoData'),
        ('Over', 'Over(x, y) - x if not null, else y'),
        ('Test', 'Test(condition, value)'),
    ],
}

# Custom function implementations
def _Con(condition, true_val, false_val):
    return np.where(condition, true_val, false_val)

def _SetNull(condition, false_val):
    return np.where(condition, np.nan, false_val)

def _Pick(value, *options):
    result = np.zeros_like(value, dtype=np.float64)
    for i, opt in enumerate(options, 1):
        mask = (value == i)
        if isinstance(opt, np.ndarray):
            result[mask] = opt[mask]
        else:
            result[mask] = opt
    return result

def _Exp10(x): return np.power(10.0, x)
def _Exp2(x): return np.power(2.0, x)
def _Float(x): return x.astype(np.float64) if isinstance(x, np.ndarray) else float(x)
def _Int(x): return np.trunc(x).astype(np.int32) if isinstance(x, np.ndarray) else int(x)
def _Ln(x): return np.log(x)
def _Log2(x): return np.log2(x)
def _Mod(x, y): return np.mod(x, y)
def _Power(base, exp): return np.power(base, exp)
def _Square(x): return np.square(x)
def _SquareRoot(x): return np.sqrt(x)
def _RoundDown(x): return np.floor(x)
def _RoundUp(x): return np.ceil(x)
def _Round(x): return np.round(x)
def _Diff(x, y): return np.abs(x - y)
def _InList(value, *options):
    result = np.zeros_like(value, dtype=np.float64)
    for opt in options:
        result = np.where(value == opt, 1.0, result)
    return result
def _IsNull(x): return np.where(np.isnan(x), 1.0, 0.0)
def _Over(x, y): return np.where(~np.isnan(x), x, y)
def _Test(condition, true_val): return np.where(condition, true_val, 0.0)
def _ATan2(y, x): return np.arctan2(y, x)

# Function replacements mapping
FUNCTION_REPLACEMENTS = {
    'Con': '_Con', 'SetNull': '_SetNull', 'Pick': '_Pick',
    'Abs': 'np.abs', 'Exp': 'np.exp', 'Exp10': '_Exp10', 'Exp2': '_Exp2',
    'Float': '_Float', 'Int': '_Int', 'Ln': '_Ln', 'Log': 'np.log',
    'Log10': 'np.log10', 'Log2': '_Log2', 'Mod': '_Mod', 'Power': '_Power',
    'Square': '_Square', 'SquareRoot': '_SquareRoot', 'Sqrt': 'np.sqrt',
    'Round': '_Round', 'RoundDown': '_RoundDown', 'RoundUp': '_RoundUp',
    'Floor': 'np.floor', 'Ceil': 'np.ceil',
    'Sin': 'np.sin', 'Cos': 'np.cos', 'Tan': 'np.tan',
    'ASin': 'np.arcsin', 'ACos': 'np.arccos', 'ATan': 'np.arctan', 'ATan2': '_ATan2',
    'SinH': 'np.sinh', 'CosH': 'np.cosh', 'TanH': 'np.tanh',
    'ASinH': 'np.arcsinh', 'ACosH': 'np.arccosh', 'ATanH': 'np.arctanh',
    'Diff': '_Diff', 'InList': '_InList', 'IsNull': '_IsNull', 'Over': '_Over', 'Test': '_Test',
}

EVAL_CONTEXT = {
    'np': np,
    '_Con': _Con, '_SetNull': _SetNull, '_Pick': _Pick,
    '_Exp10': _Exp10, '_Exp2': _Exp2, '_Float': _Float, '_Int': _Int,
    '_Ln': _Ln, '_Log2': _Log2, '_Mod': _Mod, '_Power': _Power,
    '_Square': _Square, '_SquareRoot': _SquareRoot,
    '_RoundDown': _RoundDown, '_RoundUp': _RoundUp, '_Round': _Round,
    '_Diff': _Diff, '_InList': _InList, '_IsNull': _IsNull, '_Over': _Over, '_Test': _Test,
    '_ATan2': _ATan2,
}


class RasterCalculatorWorker(QThread):
    """Worker thread for raster calculation."""
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, expression, layer_refs, output_path):
        super().__init__()
        self.expression = expression
        self.layer_refs = layer_refs
        self.output_path = output_path
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        try:
            self.calculate_raster()
        except Exception as e:
            self.finished.emit(False, str(e))
    
    def calculate_raster(self):
        self.message.emit("Parsing expression...")
        self.progress.emit(10)
        
        # Pattern: "layername" or "layername@band" - exclude @ from layer name
        layer_pattern = r'"([^"@]+)(?:@(\d+))?"'
        matches = re.findall(layer_pattern, self.expression)
        
        if not matches and not self.layer_refs:
            self.finished.emit(False, "No raster layers found. Use \"layername@1\" format.")
            return
        
        datasets = {}
        arrays = {}
        reference_ds = None
        
        self.message.emit("Loading raster layers...")
        self.progress.emit(20)
        
        for layer_name, band_str in matches:
            if self._cancelled:
                self.finished.emit(False, "Operation cancelled")
                return
            
            if layer_name not in self.layer_refs:
                self.finished.emit(False, f"Layer '{layer_name}' not found")
                return
            
            file_path = self.layer_refs[layer_name]
            band = int(band_str) if band_str else 1
            key = f'"{layer_name}@{band}"' if band_str else f'"{layer_name}"'
            
            if file_path not in datasets:
                ds = gdal.Open(file_path, gdal.GA_ReadOnly)
                if ds is None:
                    self.finished.emit(False, f"Cannot open: {file_path}")
                    return
                datasets[file_path] = ds
                if reference_ds is None:
                    reference_ds = ds
            
            ds = datasets[file_path]
            if band > ds.RasterCount:
                self.finished.emit(False, f"Band {band} not in '{layer_name}'")
                return
            
            rb = ds.GetRasterBand(band)
            arr = rb.ReadAsArray().astype(np.float64)
            nodata = rb.GetNoDataValue()
            if nodata is not None:
                arr[arr == nodata] = np.nan
            
            arrays[key] = arr
            simple_key = f'"{layer_name}"'
            if simple_key not in arrays:
                arrays[simple_key] = arr
        
        if reference_ds is None:
            self.finished.emit(False, "No valid rasters found")
            return
        
        self.message.emit("Evaluating expression...")
        self.progress.emit(50)
        
        eval_expr = self.expression
        for key in arrays.keys():
            eval_expr = re.sub(re.escape(key), f'_arrays[{repr(key)}]', eval_expr)
        
        eval_expr = eval_expr.replace('^', '**').replace('&', ' & ').replace('|', ' | ').replace('~', ' ~')
        
        for func, repl in FUNCTION_REPLACEMENTS.items():
            eval_expr = re.sub(r'\b' + func + r'\s*\(', repl + '(', eval_expr, flags=re.IGNORECASE)
        
        context = EVAL_CONTEXT.copy()
        context['_arrays'] = arrays
        
        try:
            result = eval(eval_expr, {"__builtins__": {}}, context)
        except Exception as e:
            self.finished.emit(False, f"Expression error: {str(e)}")
            return
        
        if self._cancelled:
            self.finished.emit(False, "Operation cancelled")
            return
        
        self.message.emit("Writing output...")
        self.progress.emit(80)
        
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            self.output_path,
            reference_ds.RasterXSize, reference_ds.RasterYSize,
            1, gdal.GDT_Float32, options=['COMPRESS=LZW', 'TILED=YES']
        )
        
        if out_ds is None:
            self.finished.emit(False, f"Cannot create: {self.output_path}")
            return
        
        out_ds.SetGeoTransform(reference_ds.GetGeoTransform())
        out_ds.SetProjection(reference_ds.GetProjection())
        
        out_band = out_ds.GetRasterBand(1)
        nodata_out = -9999.0
        if isinstance(result, np.ndarray):
            result = result.astype(np.float32)
            result[np.isnan(result)] = nodata_out
        out_band.SetNoDataValue(nodata_out)
        out_band.WriteArray(result)
        out_ds.BuildOverviews('NEAREST', [2, 4, 8, 16])
        out_band.FlushCache()
        out_ds = None
        
        for ds in datasets.values():
            ds = None
        
        self.progress.emit(100)
        self.finished.emit(True, f"Complete: {self.output_path}")


class RasterCalculatorDialog(QtWidgets.QDialog, FORM_CLASS):
    """Raster Calculator Dialog with compact list-based UI."""
    
    def __init__(self, iface, parent=None):
        super(RasterCalculatorDialog, self).__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.worker = None
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.layer_refs = {}
        
        self.setup_connections()
        self.load_project_rasters()
        self.update_functions_list()
    
    def setup_connections(self):
        # Layer and function list double-click
        self.layersList.itemDoubleClicked.connect(self.insert_layer)
        self.functionsList.itemDoubleClicked.connect(self.insert_function_from_list)
        self.functionCategoryCombo.currentTextChanged.connect(self.update_functions_list)
        
        # Numeric buttons
        for i in range(10):
            getattr(self, f'btn{i}').clicked.connect(lambda c, n=str(i): self.insert_text(n))
        
        # Operators
        self.btnAdd.clicked.connect(lambda: self.insert_text(' + '))
        self.btnSub.clicked.connect(lambda: self.insert_text(' - '))
        self.btnMul.clicked.connect(lambda: self.insert_text(' * '))
        self.btnDiv.clicked.connect(lambda: self.insert_text(' / '))
        self.btnPow.clicked.connect(lambda: self.insert_text(' ^ '))
        self.btnDot.clicked.connect(lambda: self.insert_text('.'))
        self.btnLParen.clicked.connect(lambda: self.insert_text('('))
        self.btnRParen.clicked.connect(lambda: self.insert_text(')'))
        self.btnComma.clicked.connect(lambda: self.insert_text(', '))
        self.btnEq.clicked.connect(lambda: self.insert_text(' == '))
        self.btnNeq.clicked.connect(lambda: self.insert_text(' != '))
        self.btnGt.clicked.connect(lambda: self.insert_text(' > '))
        self.btnGte.clicked.connect(lambda: self.insert_text(' >= '))
        self.btnLt.clicked.connect(lambda: self.insert_text(' < '))
        self.btnLte.clicked.connect(lambda: self.insert_text(' <= '))
        self.btnAnd.clicked.connect(lambda: self.insert_text(' & '))
        self.btnOr.clicked.connect(lambda: self.insert_text(' | '))
        self.btnNot.clicked.connect(lambda: self.insert_text('~'))
        
        # Controls
        self.btnClear.clicked.connect(lambda: self.expressionEdit.clear())
        self.btnBackspace.clicked.connect(lambda: self.expressionEdit.textCursor().deletePreviousChar())
        self.browseBtn.clicked.connect(self.browse_output)
        self.calculateBtn.clicked.connect(self.run_calculation)
        self.cancelBtn.clicked.connect(self.reject)
        self.helpBtn.clicked.connect(self.show_help)
    
    def update_functions_list(self):
        """Update functions list based on selected category."""
        self.functionsList.clear()
        category = self.functionCategoryCombo.currentText()
        if category in FUNCTION_CATEGORIES:
            for func_name, description in FUNCTION_CATEGORIES[category]:
                item = QtWidgets.QListWidgetItem(description)
                item.setData(Qt.UserRole, func_name)
                self.functionsList.addItem(item)
    
    def load_project_rasters(self):
        self.layersList.clear()
        self.layer_refs.clear()
        project = QgsProject.instance()
        for layer_id, layer in project.mapLayers().items():
            if isinstance(layer, QgsRasterLayer):
                name = layer.name()
                item = QtWidgets.QListWidgetItem(f"◆ {name}")
                item.setData(Qt.UserRole, name)
                self.layersList.addItem(item)
                self.layer_refs[name] = layer.source()
    
    def insert_layer(self, item):
        layer_name = item.data(Qt.UserRole)
        self.insert_text(f'"{layer_name}@1"')
    
    def insert_function_from_list(self, item):
        func_name = item.data(Qt.UserRole)
        self.insert_text(f'{func_name}(')
    
    def insert_text(self, text):
        self.expressionEdit.textCursor().insertText(text)
        self.expressionEdit.setFocus()
    
    def browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Output", "", "GeoTIFF (*.tif)")
        if path:
            if not path.lower().endswith(('.tif', '.tiff')):
                path += '.tif'
            self.outputPathEdit.setText(path)
    
    def show_help(self):
        QMessageBox.information(self, "Help", """
<h3>Raster Calculator</h3>
<p><b>Layers:</b> Double-click to add "layername@1"</p>
<p><b>Functions:</b> Select category and double-click function</p>
<p><b>Operators:</b> + - * / ^ == != > >= < <= & | ~</p>
<p><b>Example:</b> Con("dem@1" > 500, 1, 0)</p>
        """)
    
    def run_calculation(self):
        expr = self.expressionEdit.toPlainText().strip()
        out = self.outputPathEdit.text().strip()
        
        if not expr:
            QMessageBox.warning(self, "Warning", "Enter an expression")
            return
        if not out:
            QMessageBox.warning(self, "Warning", "Select output path")
            return
        
        progress = QProgressDialog("Calculating...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        self.worker = RasterCalculatorWorker(expr, self.layer_refs, out)
        self.worker.progress.connect(progress.setValue)
        self.worker.message.connect(progress.setLabelText)
        self.worker.finished.connect(lambda s, m: self.on_finished(s, m, progress))
        progress.canceled.connect(self.worker.cancel)
        self.worker.start()
    
    def on_finished(self, success, message, progress):
        progress.close()
        if success:
            QMessageBox.information(self, "Success", message)
            if self.openOutputCheck.isChecked():
                layer = QgsRasterLayer(self.outputPathEdit.text(), 
                    os.path.splitext(os.path.basename(self.outputPathEdit.text()))[0])
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
            self.accept()
        else:
            QMessageBox.critical(self, "Error", message)
    
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        event.accept()
