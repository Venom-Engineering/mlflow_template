from functools import reduce
import pandas as pd


def composite_function(*func):
    
    def compose(f, g):
        return lambda x : f(g(x))
            
    return reduce(compose, func, lambda x : x)


def drop_columns(x: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x.drop(columns, axis=1, inplace=True)
    return x

def mapper(x: pd.DataFrame, mapper: dict, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        x[column] = x[column].map(mapper)
    
    return x

def fillna_pipe(x: pd.DataFrame) -> pd.DataFrame:
    x.fillna(-999, inplace=True)
    return x