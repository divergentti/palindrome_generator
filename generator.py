# Form implementation generated from reading ui file 'generator.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_generate_palindromes_Dialog(object):
    def setupUi(self, generate_palindromes_Dialog):
        generate_palindromes_Dialog.setObjectName("generate_palindromes_Dialog")
        generate_palindromes_Dialog.resize(820, 711)
        self.gridLayout = QtWidgets.QGridLayout(generate_palindromes_Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.top_label = QtWidgets.QLabel(parent=generate_palindromes_Dialog)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.top_label.setFont(font)
        self.top_label.setObjectName("top_label")
        self.verticalLayout.addWidget(self.top_label, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.line = QtWidgets.QFrame(parent=generate_palindromes_Dialog)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.info_text_Label = QtWidgets.QLabel(parent=generate_palindromes_Dialog)
        self.info_text_Label.setMinimumSize(QtCore.QSize(800, 300))
        self.info_text_Label.setMaximumSize(QtCore.QSize(800, 16777215))
        self.info_text_Label.setWordWrap(True)
        self.info_text_Label.setObjectName("info_text_Label")
        self.verticalLayout_2.addWidget(self.info_text_Label, 0, QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignTop)
        self.line_2 = QtWidgets.QFrame(parent=generate_palindromes_Dialog)
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_2.addWidget(self.line_2)
        self.gridLayout.addLayout(self.verticalLayout_2, 1, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gener_info_label = QtWidgets.QLabel(parent=generate_palindromes_Dialog)
        self.gener_info_label.setMinimumSize(QtCore.QSize(400, 50))
        self.gener_info_label.setWordWrap(True)
        self.gener_info_label.setObjectName("gener_info_label")
        self.horizontalLayout.addWidget(self.gener_info_label, 0, QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignTop)
        self.line_3 = QtWidgets.QFrame(parent=generate_palindromes_Dialog)
        self.line_3.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout.addWidget(self.line_3)
        self.filenames_comboBox = QtWidgets.QComboBox(parent=generate_palindromes_Dialog)
        self.filenames_comboBox.setObjectName("filenames_comboBox")
        self.horizontalLayout.addWidget(self.filenames_comboBox)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.status_labelLeft = QtWidgets.QLabel(parent=generate_palindromes_Dialog)
        self.status_labelLeft.setMinimumSize(QtCore.QSize(400, 0))
        self.status_labelLeft.setObjectName("status_labelLeft")
        self.horizontalLayout_3.addWidget(self.status_labelLeft)
        self.line_4 = QtWidgets.QFrame(parent=generate_palindromes_Dialog)
        self.line_4.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_4.setObjectName("line_4")
        self.horizontalLayout_3.addWidget(self.line_4)
        self.status_Right_label = QtWidgets.QLabel(parent=generate_palindromes_Dialog)
        self.status_Right_label.setObjectName("status_Right_label")
        self.horizontalLayout_3.addWidget(self.status_Right_label)
        self.gridLayout.addLayout(self.horizontalLayout_3, 3, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.generate_Button = QtWidgets.QPushButton(parent=generate_palindromes_Dialog)
        self.generate_Button.setMinimumSize(QtCore.QSize(0, 0))
        self.generate_Button.setObjectName("generate_Button")
        self.horizontalLayout_2.addWidget(self.generate_Button)
        self.cancel_generation_pushButton = QtWidgets.QPushButton(parent=generate_palindromes_Dialog)
        self.cancel_generation_pushButton.setObjectName("cancel_generation_pushButton")
        self.horizontalLayout_2.addWidget(self.cancel_generation_pushButton)
        self.convertButton = QtWidgets.QPushButton(parent=generate_palindromes_Dialog)
        self.convertButton.setObjectName("convertButton")
        self.horizontalLayout_2.addWidget(self.convertButton)
        self.gridLayout.addLayout(self.horizontalLayout_2, 4, 0, 1, 1)

        self.retranslateUi(generate_palindromes_Dialog)
        QtCore.QMetaObject.connectSlotsByName(generate_palindromes_Dialog)

    def retranslateUi(self, generate_palindromes_Dialog):
        _translate = QtCore.QCoreApplication.translate
        generate_palindromes_Dialog.setWindowTitle(_translate("generate_palindromes_Dialog", "Dialog"))
        self.top_label.setText(_translate("generate_palindromes_Dialog", "Palindromien generointi ja json-muunnos"))
        self.info_text_Label.setText(_translate("generate_palindromes_Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Verbit, adjektiivit ja substantiivit (CSV-tiedostot) </span>on tuotettu Kotimaisten Kielten Keskuksen <a href=\"https://www.kotus.fi/aineistot/sana-aineistot/nykysuomen_sanalista\"><span style=\" text-decoration: underline; color:#0000ff;\">Nykysuomen sanalistan</span></a> (Creative Commons CC-BY) pohjalta. <span style=\" font-style:italic;\">Taivutusmuotoja ei ole huomioitu</span>. Palindromien generoinnissa käytetään lisäksi kirjallisuudesta poimittuja sanoja. Kirjan tekstisisältö ladataan<span style=\" font-weight:600;\"> long_sentences_file</span>-muuttujalla.</p><p>Tiedostojen nimet ladataan <span style=\" font-weight:600;\">runtimeconfig.json</span>-tiedostosta. Latauksen yhteydessä teksti puhdistetaan ylimääräisistä merkeistä ja muunnetaan pieniksi kirjaimiksi. Näitä tietoja käytetään osana peliä, jossa palindromien tulee olla sekä symmetrisiä että loogisesti järkeviä.</p><p>Koneoppimista ja peliä varten on tuotettu yli <span style=\" font-weight:600;\">21 000 </span>lyhyttä suomenkielistä palindromia hyödyntämällä edellä mainittuja tiedostoja uusien palindromien pohjaksi. </p><p>Jos haluat tuottaa lisää palindromiaineistoa tai vaihtaa esimerkiksi kieltä, voit luoda uudet CSV-tiedostot ja valita \'<span style=\" font-weight:600;\">Generoi palindromeja\'</span> -toiminnon. Tämä prosessi voi olla erittäin hidas ja riippuu sanojen määrästä – esimerkiksi pelkkien substantiivien käsittely voi kestää jopa toista vuorokautta! Löydetyistä palindromeista tuotetaan new_-alkuisia CSV-tiedostoja, jotka voidaan lopuksi konvertoida JSON-muotoon käyttämällä \'<span style=\" font-weight:600;\">Konvertoi</span>\' -toimintoa. Edellytyksenä konvertoinnille on kaikkien neljän new_*.csv- tiedoston olemassaolo. Tätä JSON-tiedostoa käytetään pelissä. Edistymisen näet alla olevasta ruudusta. Voit keskeyttää prosessin <span style=\" font-weight:600;\">Peru</span>- nappulalla.</p></body></html>"))
        self.gener_info_label.setText(_translate("generate_palindromes_Dialog", "<html><head/><body><p>Valitse palindromien <span style=\" font-weight:600;\">generointiin</span> käytettävä csv-tiedosto listasta. </p><p>Tiedosto käydään läpi sana kerrallaan ja tuotetaan palindromeja </p><p>kokeilemalla ja yhdistelemällä muihin sanalistoja sisältäviin tiedostoihin.</p></body></html>"))
        self.status_labelLeft.setText(_translate("generate_palindromes_Dialog", "Valitse tiedosto ..."))
        self.status_Right_label.setText(_translate("generate_palindromes_Dialog", "Tilannetietoa generoimisesta"))
        self.generate_Button.setText(_translate("generate_palindromes_Dialog", "Generoi palindromeja"))
        self.cancel_generation_pushButton.setText(_translate("generate_palindromes_Dialog", "Peru generointi"))
        self.convertButton.setText(_translate("generate_palindromes_Dialog", "Konvertoi csv-tiedostot json-tiedostoksi"))
