import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC

st.title("APLIKASI DATA MINING")
st.write("Dibuat Oleh Arusal Khofiqoyni | 200411100071 untuk Project UAS")
import_data, preporcessing, modeling, implementation = st.tabs(["Import Data", "Prepocessing", "Modeling", "Implementation"])

with import_data:
    st.write("""# Import Data""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        data = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(data)

with preporcessing:
    st.write("""# Preprocessing""")
    data.head()
    # create a dataframe with all training data except the target column
    X = data.drop(columns=["PATIENT_TYPE"])

    # # check that the target variable has been removed
    X.head()
    info = X.info()
    st.write(X)
    labels = data["PATIENT_TYPE"]

    st.write(" ## Normalisasi")
    
    sebelum_dinormalisasi = ['USMER', 'MEDICAL_UNIT', 'SEX', 'INTUBED', 'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']
    sesudah_dinormalisasi = ['NORM_USMER', 'NORM_MEDICAL_UNIT', 'NORM_SEX', 'NORM_INTUBED', 'NORM_PNEUMONIA', 'NORM_AGE', 'NORM_PREGNANT', 'NORM_DIABETES', 'NORM_COPD', 'NORM_ASTHMA', 'NORM_INMSUPR', 'NORM_HIPERTENSION', 'NORM_OTHER_DISEASE', 'NORM_CARDIOVASCULAR', 'NORM_OBESITY', 'NORM_RENAL_CHRONIC', 'NORM_TOBACCO', 'NORM_ICU']
        

    normalisasi_fitur = X[sebelum_dinormalisasi]
    st.dataframe(normalisasi_fitur)

    scaler = MinMaxScaler()
    scaler.fit(normalisasi_fitur)
    fitur_ternormalisasi = scaler.transform(normalisasi_fitur)
        

    fitur_ternormalisasi_df = pd.DataFrame(fitur_ternormalisasi, columns = sesudah_dinormalisasi)

    st.write("Data yang telah dinormalisasi")
    st.dataframe(fitur_ternormalisasi)

    data_sudah_normal = data.drop(columns=sebelum_dinormalisasi)
        
    data_sudah_normal = data_sudah_normal.join(fitur_ternormalisasi_df)

    st.write("data yang sudah dinormalisasi dan sudah disatukan dalam 1 sata frame")
    st.dataframe(data_sudah_normal)

with modeling:
    st.write("# MODELING")

    # Y = data_sudah_normal["PATIENT_TYPE"]
    # # st.dataframe(Y)
    # X = data_sudah_normal.iloc[:,1:18]
    # # st.dataframe(X)

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

    # ### Dictionary to store model and its accuracy

    # model_accuracy = OrderedDict()

    # ### Dictionary to store model and its precision

    # model_precision = OrderedDict()

    # ### Dictionary to store model and its recall

    # model_recall = OrderedDict()
    
    # # Naive Bayes
    # naive_bayes_classifier = GaussianNB()
    # naive_bayes_classifier.fit(X_train, y_train)
    # Y_pred_nb = naive_bayes_classifier.predict(X_test)

    # # decision tree
    # clf_dt = DecisionTreeClassifier(criterion="gini")
    # clf_dt = clf_dt.fit(X_train, y_train)
    # Y_pred_dt = clf_dt.predict(X_test)
    
    # # Bagging Decision tree
    # clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10, random_state=0).fit(X_train, y_train)
    # rsc = clf.predict(X_test)
    # c = ['Naive Bayes']
    # tree = pd.DataFrame(rsc,columns = c)

    

    # # K-Nearest Neighboor
    # k_range = range(1,26)
    # for k in k_range:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn.fit(X_train, y_train)
    #     Y_pred_knn = knn.predict(X_test)

    # naive_bayes_accuracy = round(100 * accuracy_score(y_test, Y_pred_nb), 2)
    # decision_tree_accuracy = round(100* metrics.accuracy_score(y_test, Y_pred_dt))
    # bagging_Dc = round(100 * accuracy_score(y_test, tree), 2)
    # knn_accuracy = round(100 * accuracy_score(y_test, Y_pred_knn), 2)
    

    # st.write("Pilih Metode : ")
    # naive_bayes_cb = st.checkbox("Naive Bayes")
    # decision_tree_cb = st.checkbox("Decision Tree")
    # bagging_tree_cb = st.checkbox("Bagging Decision Tree")
    # knn_cb = st.checkbox("K-Nearest Neighboor")

    # if naive_bayes_cb:
    #     st.write('Akurasi Metode Naive Bayes {} %.'.format(naive_bayes_accuracy))
    # if decision_tree_cb:
    #     st.write('Akurasi Metode Decision Tree {} %.'.format(decision_tree_accuracy))
    # if bagging_tree_cb:
    #     st.write('Akurasi Metode Bagging Decision Tree {} %.'.format(bagging_Dc))
    # if knn_cb:
    #     st.write('Akurasi Metode KNN {} %.'.format(knn_accuracy))

    # if naive_bayes_accuracy > decision_tree_accuracy and naive_bayes_accuracy > bagging_Dc and naive_bayes_accuracy > knn_accuracy:
    #     hasil_tinggi = naive_bayes_classifier
    #     metode = "Naive Bayes"
    # elif decision_tree_accuracy > naive_bayes_accuracy and decision_tree_accuracy > bagging_Dc and decision_tree_accuracy > knn_accuracy:
    #     hasil_tinggi = clf_dt
    #     metode = "Decision Tree"
    # elif bagging_Dc > naive_bayes_accuracy and bagging_Dc > decision_tree_accuracy and bagging_Dc > knn_accuracy:
    #     hasil_tinggi = clf
    #     metode = "Esamble Bagging Decision Tree"
    # else:
    #     hasil_tinggi = knn
    #     metode = "K-Nearest Neighboor"

# with implementation:
    
#     st.write("# IMPLEMENTATION")
#     nama_pasien = st.text_input("Masukkan Nama")
#     radius_mean = st.number_input("Masukkan Rata-rata Radius (6.98 - 28.1)", min_value=6.98, max_value=28.1)
#     texture_mean = st.number_input("Masukkan rata-rata Texture (9.71 - 39.3)", min_value=9.71, max_value=39.3)
#     perimeter_mean = st.number_input("Masukkan rata-rata perimeter (43.8 - 189)", min_value=43.8, max_value=189.0)
#     area_mean = st.number_input("Masukkan rata-rata area (144 - 2500)", min_value=144, max_value=2500)
#     smoothness_mean = st.number_input("Masukkan Rata-rata Smothness (0.05 - 0.16)", min_value=0.05, max_value=0.16)
#     compactness_mean = st.number_input("Masukkan rata-rata compactness (0.02 - 0.35)", min_value=0.02, max_value=0.35)
#     concative_mean = st.number_input("Masukkan rata-rata concative (0 - 0.43)", min_value=0.0, max_value=0.43)
#     concave_point_mean = st.number_input("Masukkan rata-rata concave point (0 - 0.2)", min_value=0.0, max_value=0.2)

#     st.write("Cek apakah kanker masuk kategori jinak atau ganas")
#     cek = st.button('Cek Kanker')
#     inputan = [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concative_mean, concave_point_mean]]

#     scaler_jl = joblib.load('normal')
#     scaler_jl.fit(inputan)
#     inputan_normal = scaler.transform(inputan)

#     FIRST_IDX = 0
#     if cek:
#         hasil_test = hasil_tinggi.predict(inputan_normal)[FIRST_IDX]
#         if hasil_test == 0:
#             st.write("Nama Customer ", nama_pasien , "Mengidap Kanker Payudara Banign/jinak Berdasarkan Model", metode)
#         else:
#             st.write("Nama Customer ", nama_pasien , "Mengidap Kanker Payudara Malignant/Ganas Berdasarkan Model", metode)

# with evaluation:
#     st.write("# EVALUATION")
#     bagan = pd.DataFrame({'Akurasi ' : [naive_bayes_accuracy,decision_tree_accuracy, bagging_Dc, knn_accuracy], 'Metode' : ["Naive Bayes", "Decision Tree", "Bagging Decision Tree", "K-Nearest Neighboor"]})

#     bar_chart = alt.Chart(bagan).mark_bar().encode(
#         y = 'Akurasi ',
#         x = 'Metode',
#     )

#     st.altair_chart(bar_chart, use_container_width=True)
