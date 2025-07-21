from sqlalchemy import create_engine
import pandas as pd

#This is for Data ingestion from postgres database.
def data_ingestion():
    try:
        

        #  Create SQLAlchemy engine
        engine = create_engine("postgresql+psycopg2://postgres:Aravind45@localhost:5432/mydb45")

        #  Upload data to PostgreSQL (replace table if exists)
        df_base=pd.read_csv("Lead Scoring.csv")
        df_base.to_sql('leadstable', con=engine, if_exists='replace', schema='public', index=False)

        # Read back from database
        query = "SELECT * FROM leadstable "
        df_from_db = pd.read_sql_query(query, con=engine)
        print(" Data read back from PostgreSQL:")

        #  Preview the data
        print(df_from_db.head())

        return df_from_db

    except Exception as e:
        print(" Error during data ingestion:", e)
        return None


#  Call the function

df = data_ingestion()