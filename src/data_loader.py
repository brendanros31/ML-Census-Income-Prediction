import pandas as pd



# Loading data
def load_data(path):
    df = pd.read_csv(path)
    return df



# Cleaning data
class clean_data:

# Replacing '?' with most common value of field
    def replace(df, to_replace, col_item={}):
        df_clean = df.copy()
        for col, relplace_with in col_item.items():
            df_clean[col] = df_clean[col].replace(to_replace,relplace_with)

        return df_clean
    

# Apply the mapping to the columns
    def category_mapping(df, col, map={}):
        df[col] = df[col].replace(map)

        return df
    

