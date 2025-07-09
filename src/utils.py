import matplotlib.pyplot as plt
import seaborn as sns



# Impact on Income
def income_impact(df, col_name, rotation=0, figsize=(12,6)):
    grouped_data = df.groupby([df[col_name], 'annual_income']).size().reset_index(name='count')

    plt.figure(figsize=figsize)
    sns.barplot(data=grouped_data, x=col_name, y='count', hue='annual_income', palette='muted')

    plt.title(f'Income Distribution by {col_name.title()}')
    plt.xlabel(f'{col_name.title()}')
    plt.ylabel('Count')

    if rotation != 0:
        plt.xticks(rotation=rotation)

    plt.show()