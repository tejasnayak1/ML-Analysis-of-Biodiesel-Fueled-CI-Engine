
import pandas

dataset = pandas.read_csv("./dataset.csv")

#|%%--%%| <vPPNELChVL|xz1vRACtPF>

import sklearn.model_selection

train: pandas.DataFrame
test: pandas.pandas.DataFrame
train, test = sklearn.model_selection.train_test_split(dataset, train_size=.8)
y_train: pandas.DataFrame = train[["BTE_percentage", "BSFC_kg_per_kWh", "NOx_ppm", "HC_ppm", "CO_percentage", "CO2_percentage"]]
x_train = train[train.columns.difference(["BTE_percentage", "BSFC_kg_per_kWh", "NOx_ppm", "HC_ppm", "CO_percentage", "CO2_percentage"])]
y_test : pandas.DataFrame= test[["BTE_percentage", "BSFC_kg_per_kWh", "NOx_ppm", "HC_ppm", "CO_percentage", "CO2_percentage"]]
x_test : pandas.DataFrame= test[test.columns.difference(["BTE_percentage", "BSFC_kg_per_kWh", "NOx_ppm", "HC_ppm", "CO_percentage", "CO2_percentage"])]

#|%%--%%| <xz1vRACtPF|lZOLt6stwb>


import sklearn.svm
import matplotlib.pyplot as plt
import sklearn.metrics


for output_parameter in ["BTE_percentage", "BSFC_kg_per_kWh", "NOx_ppm", "HC_ppm", "CO_percentage", "CO2_percentage"]:
    current_y_train = y_train[output_parameter].to_numpy()
    scale = sum(current_y_train) / len(current_y_train)
    min_y = min(current_y_train)
    max_y = max(current_y_train)
    current_y_train = current_y_train / scale
    model = sklearn.svm.SVR()
    model = model.fit(x_train, current_y_train)
    train_pred = model.predict(x_train) * scale
    test_pred = model.predict(x_test) * scale
    plt.scatter(y_test[output_parameter],  test_pred, marker="x", color="red", label="Testing set")
    plt.scatter(y_train[output_parameter], train_pred, marker="*", color="blue", label="Training set")
    plt.legend()
    plt.text(min_y, max_y, f"MSE Train : {sklearn.metrics.mean_squared_error(train_pred, y_train[output_parameter]):.3f}, MSE Test  : {sklearn.metrics.mean_squared_error(test_pred, y_test[output_parameter]):.3f}")
    plt.text(min_y, max_y * .95, f"MAE Train  : {sklearn.metrics.mean_absolute_error(train_pred, y_train[output_parameter]):.3f} MAE Test  : {sklearn.metrics.mean_absolute_error(test_pred, y_test[output_parameter]):.3f}")
    plt.text(min_y, max_y * .9, f"R2 Train  : {sklearn.metrics.r2_score(train_pred, y_train[output_parameter]):.3f} R2 Test  : {sklearn.metrics.r2_score(test_pred, y_test[output_parameter]):.3f}")
    plt.plot(
            [min(y_train[output_parameter]), max(y_train[output_parameter])],
            [min(y_train[output_parameter]), max(y_train[output_parameter])],
            linestyle="dashed",
            color="black"
    )
    plt.ylabel(f"Predicted {output_parameter}")
    plt.xlabel(f"Actual {output_parameter}")
    plt.title("Support Vector Regression: Actual vs Predicted")
    plt.show()
