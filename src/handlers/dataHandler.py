import pandas as pd

def extractColumns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    return pd.DataFrame(data[columns])

if __name__ == '__main__':
    extractColumns('')