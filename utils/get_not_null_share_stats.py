import pandas as pd
import numpy as np

def get_not_null_share_df(df: pd.DataFrame) -> pd.DataFrame:
    '''Returns pd.DataFrame with isNotNullShare of each column of given df'''
    # Calculate percent of not null share each column 
    col_pct_notNull = [] 
    for col in df.columns: 
        percent_notNull = np.mean(~df[col].isnull())
        col_pct_notNull.append([col, percent_notNull]) 
        
    col_pct_notNull_df = pd.DataFrame(col_pct_notNull, columns = ['column_name','isNotNullShare']).sort_values(by = 'isNotNullShare', ascending = False) 
    #print(col_pct_notNull_df)
    return col_pct_notNull_df