import pandas as pd
import numpy as np
import gradio as gr
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split
import seaborn as sn
import matplotlib.pyplot as plt
from PIL import Image


def evaluate_model(dataset_file):
    # Load the dataset from the CSV file
    # df = pd.read_csv(dataset_file)

    # Split the dataset into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

    # TODO: Replace this with your own model
    # Train and evaluate your model on the train and test sets
    y_train = np.random.randint(0, 6, size=100)
    y_test = np.random.randint(0, 6, size=100)

    y_pred_train = np.random.randint(0, 6, size=100)
    y_pred_test = np.random.randint(0, 6, size=100)

    # Generate the classification report and confusion matrix
    class_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    report_train = classification_report(y_train, y_pred_train, target_names=class_names, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, target_names=class_names, output_dict=True)
    matrix_train = confusion_matrix(y_train, y_pred_train)
    matrix_test = confusion_matrix(y_test, y_pred_test)

    train_cm = pd.DataFrame(matrix_train, class_names, class_names)
    test_cm = pd.DataFrame(matrix_test, class_names, class_names)
    # plt.figure(figsize=(10,7))
    # sn.set(font_scale=1.4)  # for label size
    sn.heatmap(train_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.savefig('./train_cm.tiff')
    plt.clf()
    # sn.set(font_scale=1.4)  # for label size
    sn.heatmap(test_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.savefig('./test_cm.tiff')

    # Create a dictionary with the results
    # results = {
    #     'classification_report_train': report_train,
    #     'classification_report_test': report_test,
    #     'confusion_matrix_train': matrix_train,
    #     'confusion_matrix_test': matrix_test
    # }
    results = [
        pd.DataFrame(report_train).transpose(),
        # report_train,
        pd.DataFrame(report_test).transpose(),
        # report_test,
        # matrix_train,
        Image.open('./train_cm.tiff'),
        # matrix_test
        Image.open('./test_cm.tiff')
    ]

    return results


# Create a Gradio interface for the model evaluation
# inputs = gr.components.Textbox(label="Dataset file (CSV)")
inputs = gr.components.File(label="Dataset file (CSV)")
outputs = [
    gr.components.Dataframe(label="Classification report (train)"),
    gr.components.Dataframe(label="Classification report (test)"),
    gr.components.Image(type='pil', label="Confusion matrix (train)"),
    gr.components.Image(type='pil', label="Confusion matrix (test)"),
]

interface = gr.Interface(fn=evaluate_model, inputs=inputs, outputs=outputs)

# Launch the Gradio interface
interface.launch()
