

"""
===============================================================
Librerias 
===============================================================
""" 

import pandas as pd
import numpy as np
import argparse as ap
from pathlib import Path
import os, time, logging
from copy import deepcopy
# Importar Decision Tree de scikit-learn
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Importar objeto para k-fold cross-validation con grid de hyperparámetros
from sklearn.model_selection import GridSearchCV
# Importar objeto para separación de datos en conjuntos de entrenamiento y evaluación
from sklearn.model_selection import train_test_split
# Importar objeto para escalamiento de datos
from sklearn.preprocessing import StandardScaler
# Importar metricas de evaluación
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
# Importar objeto para imprimir matriz de confusión
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Importar objeto para imprimir reporte de clasificación
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
# Plots
import matplotlib.pyplot as plt



"""
===============================================================
Argument parser 
===============================================================
""" 

def parser(): 
    """
    Funcion para resivir los arguimentos CLI y guardarlos en el objeto tipo ArgumentParser
    
    Returns: 
        - parser : los argumentos parseados listos para accesar a su valor dentro de este objeto
    """
    parser = ap.ArgumentParser(description="Entrenar y comparar modelo RF, DT y SVM")
    #agregamos todos los argumentos
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        help="Id para los resultados",
        default="ML_compar"
    )
    parser.add_argument(
        "-d",
        "--data_s",
        type=str,
        help="Path al data set a utilizar",
        required=True
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Directorio de salida de los resultados",
        default="ML_compar"
    )
    parser.add_argument(
        "-c",
        "--col_inter",
        type=str,
        help="columna de interes para la prediccion",
        default="cardio"
    )
    parser.add_argument(
        "-r",
        "--remove_col",
        type=str,
        help="Lista de id de columnas a eliminar para el training y test del modelo separadas por una coma",
        default=None
    )
    parser.add_argument(
        "-t",
        "--test_porpo",
        type=float,
        help="Proporcion < 1 de datos para el test",
        default=0.2
    )
    return parser.parse_args()

"""
===============================================================
Train model 
===============================================================
"""

def train_M(model_ML, X_train, y_train, hiper_P={}, k=5, name="ML_train"):
    """
    Function that trains 3 different machine learning algorthms 
    Arguments:
        -model_ML: type of model to create, 3 suported options: DT for desicion trees 
            RF for random forest and SVM for suported vector machines 
        -X_train: set of all the variables to be used for training
        -y_train: labels of the variable of interest to be predicted in the X_train 
        -hiper_P: dictionary with the hiperparameter names as keys and a list for the values for 
            the grid for each hiperparameter 
        -name: name to be used in the res df
    """
    match model_ML:
        case "DT": 
            model_t=tree.DecisionTreeClassifier()
        case "RF": 
            model_t=RandomForestClassifier()
        case "SVM":
            model_t=SVC()
        case _: 
            return 0
    #hacer el grid de hiperparametros 
    model_f= GridSearchCV(model_t, hiper_P, verbose=2, scoring="f1_macro", cv=k)
    grid_df=pd.DataFrame(ParameterGrid(model_f.param_grid))
    #entrenar el modelo 
    model_f.fit(X_train, y_train)
    res_df = pd.DataFrame({
        "Model": [name],
        "hiperparameters": [str(model_f.best_params_)],
        "F-1_train_cv": [model_f.best_score_]
    })
    print(res_df)
    return [model_f, res_df, grid_df]


"""
===============================================================
Test model 
===============================================================
"""

def test_M(model_t, X_test, y_test,type_m="macro", name="ML_test", axes=None):
    """
    function to test the model 
    Arguments: 
        -model_t: object with the model to be tested 
        -X_test: set of the variables needed to make the prediction
        -y_test: labels of the predicted data
        -type_m: tyope of the metrics to be used, macro or weighted
    """
    #make the prediction test 
    y_pred=model_t.predict(X_test)
    
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
    # Matriz absoluta
    confusion_matrix_t = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_t,
                                  display_labels=model_t.classes_)
    disp.plot(cmap=plt.cm.Blues, ax=axes[0]) 
    axes[0].set_title(f"Confusion matrix {name}")

    # Matriz normalizada 
    norm_confused_mat = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=norm_confused_mat,
                                  display_labels=model_t.classes_)
    disp.plot(cmap=plt.cm.Blues, ax=axes[1]) 
    axes[1].set_title(f"Confusion matrix normalized {name}")

    # Reportes de clasificación 
    report_class = classification_report(y_test, y_pred, output_dict=True)
    report_class_f = classification_report(y_test, y_pred)
    res_df = pd.DataFrame({
        "Model": [name],
        f"F-1_score_{type_m}": [f1_score(y_test, y_pred, average=type_m)],
        f"precision_{type_m}": [precision_score(y_test, y_pred, average=type_m)],
        f"recall_{type_m}": [recall_score(y_test, y_pred, average=type_m)],
        f"AUROC_{type_m}": [roc_auc_score(y_test, y_pred)],
        "F-1_score_weighted": [report_class["weighted avg"]["f1-score"]],
        "precision_weighted": [report_class["weighted avg"]["precision"]],
        "recall_weighted": [report_class["weighted avg"]["recall"]]
    })
    
    return [res_df, report_class_f]





"""
===============================================================
Main function
===============================================================
"""

def main():
    args= parser()
    pref=args.prefix
    out_dir=args.outdir
    data_set=args.data_s
    var_interes=args.col_inter
    remove_cols=args.remove_col.strip().split(",") if args.remove_col else []
    test_propor=args.test_porpo
    
    #logging info 
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    logging.basicConfig(
        filename=os.path.join(out_dir,f"{pref}_execution.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logging.info("Leeyendo el modelo ")
    data_set = pd.read_csv(data_set, index_col=0).rename_axis(None)
    #eliminar columnas que no son de interes 
    if remove_cols:
        data_set.drop(remove_cols, axis=1, inplace=True)

    logging.info(f"Dimensionalidad del data set:{data_set.shape}")  

    for col in data_set.columns: 
        non_defined_val=len(data_set[data_set[col]=="NA"].index) + len(data_set[data_set[col]==None].index)
        logging.info(f"NA values in {col}: {non_defined_val}") 


    logging.info(f"Categorías únicas de clasificación:\n{data_set[var_interes].unique()}")
    

    logging.info(f"Num de intancias en la categoria {var_interes}:")
    logging.info(f"{data_set.groupby(var_interes).size()}") 
    
    
    #preparación del data set 
    X = data_set.loc[:,data_set.columns[data_set.columns != var_interes]]
    logging.info("Características")
    logging.info(f"{X.head()}") 


    y = data_set.loc[:,var_interes]  # todos los renglones y sólo la columna 12
    logging.info("Categorias de clase")
    logging.info("{y.head()}")
    
    
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X) 

    data_set_T={
        "scaled":{
            "X":X_scaled,
            "X_train":None,
            "y_train":None,
            "X_test":None,
            "y_test":None
        },
        "unscaled":{
            "X":X,
            "X_train":None,
            "y_train":None,
            "X_test":None,
            "y_test":None
        }
    }
    
    #fijar un random state para reproducibilida
    random_state = 42

    for data_t in data_set_T.keys():
        print(f"Separando datos de {data_t}")
        data_set_T[data_t]["X_train"], data_set_T[data_t]["X_test"], data_set_T[data_t]["y_train"], data_set_T[data_t]["y_test"] = train_test_split(data_set_T[data_t]["X"], y, test_size=test_propor, random_state=random_state, stratify=y)
    
    #entrenar y evaluar
    res_df=pd.DataFrame()
    res_dic={
        "DT":{
            "HP":{'criterion':('gini', 'entropy'), 'class_weight':['balanced', None]}
        }, 
        "RF":{
            "HP":{'class_weight':['balanced', None], 'n_estimators':np.arange(50,550, 50), 'criterion': ['gini', 'entropy']}
        }, 
        "SVM":{
            "HP":{'kernel':('linear', 'rbf', 'poly'), 'C':np.arange(1,11,1)}
        }
    } 
    #out paths 
    Path(f"{out_dir}/Test").mkdir(parents=True, exist_ok=True)
    Path(f"{out_dir}/Train").mkdir(parents=True, exist_ok=True)

    for ML_a in res_dic.keys(): 
        conf_mat, big_axes = plt.subplots(2, 2, figsize=(15, 12))
        ax_c=0
        for data_t in data_set_T.keys():
            #make the training 
            name_m=f"{ML_a}_{data_t}"
            logging.info(f"{'='*40}\nTraining {name_m} model {time.perf_counter() - start_time:.2f}s\n{'='*40}")
            train_res=train_M(ML_a,data_set_T[data_t]["X_train"], data_set_T[data_t]["y_train"],res_dic[ML_a]["HP"], name=name_m)
            model_t=train_res[0]
            res_df_tr=train_res[1]
            grid_t=train_res[2]
            logging.info(f"Training {name_m} finished {time.perf_counter() - start_time:.2f}s\nGrid training:\n{grid_t}")
            #save the grid 
            grid_t.to_csv(f"{out_dir}/Test/{name_m}_grid_hip.csv")
            logging.info(f"Testing {name_m} {time.perf_counter() - start_time:.2f}s")
            #make the tsting
            test_res=test_M(model_t,data_set_T[data_t]["X_test"], data_set_T[data_t]["y_test"], name=name_m, axes=big_axes[ax_c])
            res_df_ts = test_res[0]
            ax_c+=1
            logging.info(f"Training finished {name_m} {time.perf_counter() - start_time:.2f}s, report:\n{test_res[1]}")
            model_full_metrics = pd.merge(res_df_tr, res_df_ts, on="Model")
            # Guardamos en el diccionario si quieres mantener ese registro
            res_dic[ML_a][data_t] = model_full_metrics
            #add to the res df
            res_df = pd.concat([res_df, model_full_metrics], ignore_index=True)
        #save the confusion matrix 
        conf_mat.savefig(f"{out_dir}/Train/{pref}_{ML_a}_confusion_matrix.png")
        
    #finally save the results 
    res_df.to_csv(f"{out_dir}/{pref}_final_compar_res.csv")    
    
if __name__ == "__main__":
    main()