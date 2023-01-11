import ctypes
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPixmapItem, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QImage, QPixmap, QPixmap, QPen, QColor, QFont
import numpy as np

from UI import Ui_MainWindow
from NImage import ImageClass
from NImgProcess import ImgProcessClass
from NObject import NObjectClass
from NGauge import NGaugeClass
from MLP import MLPClass

from ctypes import *

import os

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.NImg = ImageClass()
        self.NImg2 = ImageClass()
        self.NSplittedImg = ImageClass()
        self.NOCRImg = ImageClass()
        self.NOCRImg.ReSize(10,15)
        self.NImgProcess = ImgProcessClass()
        self.NObject = NObjectClass()
        self.NGauge = NGaugeClass()
        self.NMLP = MLPClass()
        self.ui = Ui_MainWindow()
        self.OCR_Jog_index = 0
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.img_path = ''
        self.ui.OpenBMP.clicked.connect(self.open_bmp)
        self.ui.Inverse.clicked.connect(self.inverse)
        self.ui.SingleThres.clicked.connect(self.single_thres)
        self.ui.OtsuThres.clicked.connect(self.otsu_thres)
        self.ui.BlobLabelling.clicked.connect(self.blobLabelling)
        self.ui.CharSegment.clicked.connect(self.char_segment)
        self.ui.OCRJog.clicked.connect(self.ocr_jog)
        self.ui.OCRALL.clicked.connect(self.ocr_all)
        self.ui.LoadNetwork.clicked.connect(self.load_net_wrok)
        self.ui.MLPTraining.clicked.connect(self.mlp_training)
        self.ui.SaveNetwork.clicked.connect(self.save_network)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.img_path = fileName

            if os.path.isfile(self.img_path):
                print("Found " + self.img_path)
                print(os.path.relpath(self.img_path,'.'))

    def display_img(self):
        self.img = self.NImg.LoadBMP(os.path.relpath(self.img_path,'.'))
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
        self.bytesPerline = self.width
        self.bmp_load = False
        self.ui.BlobCount.setText("")
        self.ReloadImg(self.NImg)
        self.NImg2.ReSize(self.NImg.GetWidth(), self.NImg.GetHeight())
        self.bmp_load = True
        self.OCR_Jog_index = 0
                
    def open_bmp(self):
        self.openFileNameDialog()
        self.display_img()

    def load_net_wrok(self):
        # Open a file dialog to select an MLP file
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(None, "Select MLP File", "", "MLP (*.mlp);;All Files (*)", options=options)
        if file_name:
            self.NMLP.LoadNetwork(file_name)

    def save_network(self):
        filters = "MultiLayer Perceptron Files (*.mlp)"
        save_file_dialog = QFileDialog()
        save_file_dialog.setNameFilter(filters)
        save_file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        if save_file_dialog.exec_():
            file_name = save_file_dialog.selectedFiles()[0]
            self.NMLP.SaveNetwork(file_name)

    def inverse(self):
        if (self.NImgProcess.Inverse(self.NImg.GetHandle())):
            self.ReloadImg(self.NImg)
        
    def single_thres(self):
        if (self.NImgProcess.SingleThresholding(self.NImg.GetHandle(), 100)):
            self.ReloadImg(self.NImg)

    def otsu_thres(self):
        if (self.NImgProcess.OtsuThresholding(self.NImg.GetHandle(), self.NImg2.GetHandle())):
            self.ReloadImg(self.NImg2)

    def blobLabelling(self):
        self.ui.BlobCount.setText("")
        self.NImgProcess.OtsuThresholding(self.NImg.GetHandle(), self.NImg2.GetHandle())
        array = self.NImg2.ReturnArray()
        blob_count = self.NObject.Blob_Labelling(array)
        draw_font = QFont("Time New Roman", 12)
        ct_x = (ctypes.c_int * 10000)()
        ct_y = (ctypes.c_int * 10000)()

        chain_count = 0
        center_pen = QPen(QColor(255, 0, 0), 1)
        for i in range(blob_count):
            chain_count = self.NObject.Contour_Tracing(array, i, ct_x, ct_y)

            blob_area = self.NObject.Area(i)

            area_item = QtWidgets.QGraphicsTextItem(str(blob_area))
            area_item.setFont(draw_font)
            area_item.setDefaultTextColor(QColor("Blue"))
            area_item.setPos(ct_x[0], ct_y[0])
            self.scene.addItem(area_item)
            for j in range(chain_count - 1):
                line = QGraphicsLineItem(ct_x[j], ct_y[j], ct_x[j+1], ct_y[j+1])
                line.setPen(center_pen)
                self.scene.addItem(line)

        self.ui.BlobCount.setText("Blob count = " + str(blob_count))

    def dilaition(self):
        if (self.NImgProcess.Dilation3x3(self.NImg.GetHandle(), self.NImg2.GetHandle())):
            self.ReloadImg(self.NImg2)

    def erosion(self):
        if (self.NImgProcess.Erosion3x3(self.NImg.GetHandle(), self.NImg2.GetHandle())):
            self.ReloadImg(self.NImg2)

    def char_segment(self):
        if not self.bmp_load:
            return
        #Otsu二值化
        self.NImgProcess.OtsuThresholding(self.NImg.GetHandle(), self.NImg2.GetHandle())
        #反向
        self.NImgProcess.Inverse(self.NImg2.GetHandle())
        array = self.NImg2.ReturnArray()
        #標出Blob
        blob_count = self.NObject.Blob_Labelling(array)

        start_x_ptr = ctypes.c_int(0)
        start_y_ptr = ctypes.c_int(0)
        rect_w_ptr = ctypes.c_int(0)
        rect_h_ptr = ctypes.c_int(0)
        rec_pen = QPen(QColor(255, 0, 0), 1)
        #針對每個blob畫出外接矩形
        for i in range(blob_count):
            if self.NObject.Rect(i, start_x_ptr, start_y_ptr, rect_w_ptr, rect_h_ptr):
                start_x = start_x_ptr.value
                start_y = start_y_ptr.value
                rect_w = rect_w_ptr.value
                rect_h = rect_h_ptr.value
                line = QGraphicsLineItem(start_x, int(start_y), int(start_x + rect_w), int(start_y))
                line.setPen(rec_pen)
                line2 = QGraphicsLineItem(int(start_x + rect_w), int(start_y), int(start_x + rect_w), int(start_y+ rect_h))
                line2.setPen(rec_pen)
                line3 = QGraphicsLineItem(int(start_x + rect_w), int(start_y + rect_h), int(start_x), int(start_y + rect_h))
                line3.setPen(rec_pen)
                line4 = QGraphicsLineItem(int(start_x), int(start_y + rect_h), int(start_x), int(start_y))
                line4.setPen(rec_pen)
                self.scene.addItem(line)
                self.scene.addItem(line2)
                self.scene.addItem(line3)
                self.scene.addItem(line4)
	
    def mlp_training(self):
        if not self.bmp_load:
            return
        #清空UI上blob_count數字
        self.ui.BlobCount.setText("")
        #Otsu二值化
        self.NImgProcess.OtsuThresholding(self.NImg.GetHandle(), self.NImg2.GetHandle())
        #反向
        self.NImgProcess.Inverse(self.NImg2.GetHandle())
        array = self.NImg2.ReturnArray()
        #標出Blob
        blob_count = self.NObject.Blob_Labelling(array)

        # Char Segmentation
        start_x_ptr = ctypes.c_int(0)
        start_y_ptr = ctypes.c_int(0)
        rect_w_ptr = ctypes.c_int(0)
        rect_h_ptr = ctypes.c_int(0)

        samples = np.full((150, blob_count), 0, dtype=np.uint8)
        vector = np.empty(150, dtype=np.uint8)
        for i in range(blob_count):
            #影像分割：取得字元外接矩形  Exercise_7
            if self.NObject.Rect(i, start_x_ptr, start_y_ptr, rect_w_ptr, rect_h_ptr):
                #影像分割：寬高比過大，為避免大寫 I 的誤判問題，延伸外框
                start_x = start_x_ptr.value
                start_y = start_y_ptr.value
                rect_w = rect_w_ptr.value
                rect_h = rect_h_ptr.value
                if rect_h / rect_w > 3.0:
                    start_x -= 5
                    rect_w += 10

                    self.NSplittedImg.ReSize(rect_w, rect_h)
                    #影像分割：切割字元影像 Exercise_7
                    if self.NImgProcess.Split_Image(self.NImg2.GetHandle(), self.NSplittedImg.GetHandle(), start_x, start_y, rect_w, rect_h):
                        #特徵擷取：Resize image to 10*15  Exercise_7
                        if self.NImgProcess.Small_Transform(self.NSplittedImg.GetHandle(), self.NOCRImg.GetHandle()):
                            #特徵擷取：Transfer to 1-D feature vector Exercise_7
                            self.NImgProcess.FromImageToVector(self.NOCRImg.GetHandle(), vector, 150)
                            for j in range(150):
                                samples[j][i] = vector[j]
        
        trainer_string = ""
        number_of_input_sets = 0
        options = QFileDialog.Options()
        # Create file dialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Character Trainer Set (*.cts)|*.cts", options=options)
        if fileName:
            character_trainer_set_file = open(fileName, 'r')

        # Read file line by line
        for line in character_trainer_set_file:
            trainer_string += line

        # Close file
        character_trainer_set_file.close()

        # Get number of input sets
        number_of_input_sets = len(trainer_string)

        # Set label text
        self.ui.TrainingString.setText(trainer_string)

        # Perform MLP training if number of input sets and blob count match
        if number_of_input_sets == blob_count:
            self.NMLP.Training(samples, trainer_string, blob_count)
        
    def ocr_jog(self):
        if not self.bmp_load:
            return
        self.NImgProcess.OtsuThresholding(self.NImg.GetHandle(), self.NImg2.GetHandle())
        self.NImgProcess.Inverse(self.NImg2.GetHandle())
        array = self.NImg2.ReturnArray()
        if self.NObject.Blob_Count() == 0:
            self.NObject.Blob_Labelling(array)
        elif self.OCR_Jog_index >= self.NObject.Blob_Count():
            self.OCR_Jog_index = 0

        sample = np.empty(150, dtype=np.uint8)
        start_x_ptr = ctypes.c_int(0)
        start_y_ptr = ctypes.c_int(0)
        rect_w_ptr = ctypes.c_int(0)
        rect_h_ptr = ctypes.c_int(0)

        if self.NObject.Rect(self.OCR_Jog_index, start_x_ptr, start_y_ptr, rect_w_ptr, rect_h_ptr):
            start_y = start_y_ptr.value
            start_x = start_x_ptr.value
            rect_w = rect_w_ptr.value
            rect_h = rect_h_ptr.value
            if rect_h / rect_w > 3.0:
                start_x -= 5
                rect_w += 10

            self.NSplittedImg.ReSize(rect_w, rect_h)
            #影像分割：切割字元影像 Exercise_7
            if self.NImgProcess.Split_Image(self.NImg2.GetHandle(), self.NSplittedImg.GetHandle(), start_x, start_y, rect_w, rect_h):
                #特徵擷取：Resize image to 10*15  Exercise_7
                if self.NImgProcess.Small_Transform(self.NSplittedImg.GetHandle(), self.NOCRImg.GetHandle()):
                    self.NImgProcess.FromImageToVector(self.NOCRImg.GetHandle(), sample, 150)

                    self.hbitmap2 = QImage(self.NOCRImg.ReturnArray(), self.NOCRImg.GetWidth(), self.NOCRImg.GetHeight(), self.NOCRImg.GetWidth(), QImage.Format_Grayscale8)
                    self.scene_ocr = QGraphicsScene()
                    item = QGraphicsPixmapItem(QPixmap.fromImage(self.hbitmap2))
                    self.scene_ocr.addItem(item)
                    self.ui.Char_Text_GraphicsView.setScene(self.scene_ocr)
                    self.ui.Char_Text_GraphicsView.adjustSize()

                    self.OCR_Jog_index += 1
                    center_pen = QPen(QColor(255, 0, 255), 1)
                    self.scene = QGraphicsScene()
                    line = QGraphicsLineItem(int(start_x), int(start_y), int(start_x + rect_w), int(start_y))
                    line.setPen(center_pen)
                    line2 = QGraphicsLineItem(int(start_x + rect_w), int(start_y), int(start_x + rect_w), int(start_y+ rect_h))
                    line2.setPen(center_pen)
                    line3 = QGraphicsLineItem(int(start_x + rect_w), int(start_y + rect_h), int(start_x), int(start_y + rect_h))
                    line3.setPen(center_pen)
                    line4 = QGraphicsLineItem(int(start_x), int(start_y + rect_h), int(start_x), int(start_y))
                    line4.setPen(center_pen)

                    self.scene.addItem(line)
                    self.scene.addItem(line2)
                    self.scene.addItem(line3)
                    self.scene.addItem(line4)

                    # Classify using MLP
                    result = self.NMLP.Classify(sample)
                    draw_font = QFont("Time New Roman", 12)
                    area_item = QtWidgets.QGraphicsTextItem(str(result))
                    area_item.setFont(draw_font)
                    area_item.setDefaultTextColor(QColor(0, 0, 255))
                    area_item.setPos(start_x - 10, start_y - 10)
                    self.scene.addItem(area_item)

    def ocr_all(self):
        if not self.bmp_load:
            return
        
        blob_count = 0
        if self.NObject.Blob_Count() == 0:
            self.NImgProcess.OtsuThresholding(self.NImg.GetHandle(), self.NImg2.GetHandle())
            self.NImgProcess.Inverse(self.NImg2.GetHandle())
            array = self.NImg2.ReturnArray()
            blob_count = self.NObject.Blob_Labelling(array)
        else:
            blob_count = self.NObject.Blob_Count()
        
        sample = np.empty(150, dtype=np.uint8)
        start_x_ptr = ctypes.c_int(0)
        start_y_ptr = ctypes.c_int(0)
        rect_w_ptr = ctypes.c_int(0)
        rect_h_ptr = ctypes.c_int(0)
        
        for i in range(blob_count):
            if self.NObject.Rect(i, start_x_ptr, start_y_ptr, rect_w_ptr, rect_h_ptr):
                start_y = start_y_ptr.value
                start_x = start_x_ptr.value
                rect_w = rect_w_ptr.value
                rect_h = rect_h_ptr.value
                if rect_h / rect_w > 3.0:
                    start_x -= 5
                    rect_w += 10
                
                self.NSplittedImg.ReSize(rect_w, rect_h)
                if self.NImgProcess.Split_Image(self.NImg2.GetHandle(), self.NSplittedImg.GetHandle(), start_x, start_y, rect_w, rect_h):
                    if self.NImgProcess.Small_Transform(self.NSplittedImg.GetHandle(), self.NOCRImg.GetHandle()):
                        self.NImgProcess.FromImageToVector(self.NOCRImg.GetHandle(), sample, 150)
        
                        center_pen = QPen(QColor(0, 255, 0), 1)

                        line = QGraphicsLineItem(int(start_x), int(start_y), int(start_x + rect_w), int(start_y))
                        line.setPen(center_pen)
                        line2 = QGraphicsLineItem(int(start_x + rect_w), int(start_y), int(start_x + rect_w), int(start_y+ rect_h))
                        line2.setPen(center_pen)
                        line3 = QGraphicsLineItem(int(start_x + rect_w), int(start_y + rect_h), int(start_x), int(start_y + rect_h))
                        line3.setPen(center_pen)
                        line4 = QGraphicsLineItem(int(start_x), int(start_y + rect_h), int(start_x), int(start_y))
                        line4.setPen(center_pen)
                        self.scene.addItem(line)
                        self.scene.addItem(line2)
                        self.scene.addItem(line3)
                        self.scene.addItem(line4)

                        result = self.NMLP.Classify(sample)
                        draw_font = QFont("Time New Roman", 12)
                        area_item = QtWidgets.QGraphicsTextItem(str(result))
                        area_item.setFont(draw_font)
                        area_item.setDefaultTextColor(QColor("Blue"))
                        area_item.setPos(start_x - 10, start_y - 10)                            
                        self.scene.addItem(area_item)

    def ReloadImg(self, img):
        self.originalQImage = QImage(img.ReturnArray(), img.GetWidth(), img.GetHeight(), img.GetWidth(), QImage.Format_Grayscale8)
        self.scene = QGraphicsScene()
        item = QGraphicsPixmapItem(QPixmap.fromImage(self.originalQImage))
        self.scene.addItem(item)
        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.adjustSize()

