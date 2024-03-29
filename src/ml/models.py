from enum import Enum
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class ModelType(Enum):
    RANDOM_FOREST_CLASSIFIER = RandomForestClassifier(random_state=42)
    RANDOM_FOREST_REGRESSOR = RandomForestRegressor(random_state=42)
    LINEAR_REGRESSION = LinearRegression()
    DECISION_TREE_CLASSIFIER = DecisionTreeClassifier()
    DECISION_TREE_REGRESSOR = DecisionTreeRegressor()
    SVR = SVR(C=1.0, epsilon=0.2)
    GRADIANT_BOOSTING_REGRESSOR = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
