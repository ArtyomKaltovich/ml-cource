def extract_target_column(df, column_name):
    column = df[column_name]
    df = df.drop(column_name, axis=1)
    return df, column
