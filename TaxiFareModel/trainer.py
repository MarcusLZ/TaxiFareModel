# imports
from cgi import test
from re import X
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ("dist_trans", DistanceTransformer()),
            ("std_scaler", StandardScaler())
            ])

        time_pipe = Pipeline([
            ("time_trans", TimeFeaturesEncoder('pickup_datetime')),
            ("robust_scaler", RobustScaler())
            ])

        preproc_pipe = ColumnTransformer([
            ('distance_pipe',dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time_pipe',time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

        self.pipeline = pipe

        return self


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)

        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        return compute_rmse(self.pipeline.predict(X_test),y_test)


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate

    df = get_data()
    df = clean_data(df)

    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

    pipe = Trainer(X_train,y_train)
    pipe.run()
    print(pipe.evaluate(X_test, y_test))
