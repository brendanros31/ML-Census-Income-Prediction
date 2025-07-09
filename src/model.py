from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



def build_model(model_type, params=None):
    if model_type == 'LogisticRegression':
        return LogisticRegression(**(params or {}))
    
    elif model_type == 'RandomForestClassifier':
        return RandomForestClassifier(**(params or {}))



def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)
