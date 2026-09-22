from pathlib import Path
import sqlite3, pandas as pd
BASE=Path(__file__).resolve().parents[1]
DB=BASE/'database'/'carsure.db'
DATA=BASE/'data'
conn=sqlite3.connect(DB)
for name in ['vehicles','accidents','service_history','insurance','market_prices','vehicle_condition']:
    p=DATA/f'{name}.csv'
    if p.exists():
        df=pd.read_csv(p)
        df.to_sql(name, conn, if_exists='replace', index=False)
print('Database created:', DB)
conn.close()
