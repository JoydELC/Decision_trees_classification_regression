# Decision trees for classification and regression.

In this publication, we will perform two examples using decision trees, first we will do an example of classification to analyze the **load_breast_cancer** data set, and second we will analyze an invented dataset to see how the algorithm works for regression, to finish with an example of regression using the **ram_price.csv** as data set.

## Example for classification

First, for this example, we load our data set and perform the respective data split, build our model and obtain results.

```python
#Build model
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

```

**Decision tree obtained**


![image](https://user-images.githubusercontent.com/115313115/210190006-a12e92f7-02d5-4a72-8b2e-6955782f1283.png)


## Example for regression

For this example we create our own data set by creating two arrays, one with the characteristics and the other with the target, once the data set is created we create our model and we train.

**Dataset**

![image](https://user-images.githubusercontent.com/115313115/210190915-90f4e759-a32b-49d6-887f-70d8c3e4db06.png)


```python
#build model
dt_regressor = DecisionTreeRegressor(max_depth=2)
dt_regressor.fit(features, targets)

export_graphviz(dt_regressor, out_file="dt_regressor.dot", filled=True, rounded=True,
                    special_characters=True)

with open("dt_regressor.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

```

**Decision tree obtained**


![image](https://user-images.githubusercontent.com/115313115/210190951-ef2cc07d-530a-4297-966f-e14c2a5bbb2a.png)


**Final results **


![image](https://user-images.githubusercontent.com/115313115/210190966-a0d91ede-61fc-45ac-9be4-a2a870cfd7f7.png)

We proceed in the same way with our Ram prices dataset.

**Ram prices data**


![image](https://user-images.githubusercontent.com/115313115/210191020-6234a1d8-59d9-49b8-813a-90d1c8543981.png)

```python
#Build model
from sklearn.linear_model import LinearRegression

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)
# predict on all data
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)
```

**Final results**


![image](https://user-images.githubusercontent.com/115313115/210191078-33bbf47f-ba16-465b-84d0-dec5330fa637.png)
