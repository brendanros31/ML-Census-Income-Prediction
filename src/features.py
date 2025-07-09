from sklearn.preprocessing import LabelEncoder, StandardScaler



# Encoding selected Features
def encode_features(df, categorical_cols):
    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df



# Scaling data
def scale_features(col):
    scaler = StandardScaler()
    return scaler.fit_transform(col)


