import pandas as pd
import numpy as np
from keras import Model
from keras.layers import *



# data = pd.read_csv('streaming_data.csv')
# data = data["data"]
# all_data = []
# for i in data:
#
#     d = i.strip("[]")
#     d = d.split()
#     clean = [item.replace("\n", "").replace("'", "").replace("-", "0") for item in d]
#     cd = [item.strip("'") for item in clean]
#     all_data.append(cd)
# df = pd.DataFrame(all_data)


def Mutual_information_features():
    # relationship between feature and label variable,
    # which feature gives better quality to get better performance
    # Converting to DataFrame
    df = pd.DataFrame(data)

    # Assume the last column is the target variable and the rest are features
    X = df.drop(columns=['Column22'])
    y = df['Column22']

    # Calculate mutual information
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(X, y, discrete_features='auto')
    # Create a DataFrame for better readability
    mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})
    # Sort by mutual information
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)
    # Print the mutual information of each feature
    print(mi_df)


class WatershedReservoirSampler:
    def __init__(self, k):
        self.k = k
        self.reservoirs = []

    def sample(self, item):
        if len(self.reservoirs) < self.k:
            self.reservoirs.append(item)
        else:
            idx = random.randint(0, len(self.reservoirs))
            if idx < self.k:
                self.reservoirs[idx] = item

    def get_sample(self):
        return self.reservoirs

class Proposed_Model:
    def __init__(self, xtrain, ytrain, xtest, ytest):
        self.data = None
        self.k = None

        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest

        self.reservoirs = []

    def sample(self, item):
        if len(self.reservoirs) < self.k:
            self.reservoirs.append(item)
        else:
            idx = random.randint(0, len(self.reservoirs))
            if idx < self.k:
                self.reservoirs[idx] = item

    # def get_sample(self):
    #     return self.reservoirs

    def reservoir_sampling1(self):
        sample = []
        k = self.k
        count = 0
        threshold = 100
        for i, point in enumerate(self.data):
            count += 1
            if i < k:
                sample.append(point)
            else:
                j = np.random.randint(0, i + 1)
                if j < k and count <= threshold:
                    sample[j] = point
        return sample

    def splitting(self):
        data = pd.concat([pd.DataFrame(self.xtrain), pd.DataFrame(self.ytrain)], axis=1)
        self.data = np.array(data)
        # Perform reservoir sampling on each node
        # self.k = 100  # sample_size
        # sampled_data = self.reservoir_sampling1()

        self.k = 3
        sampler = WatershedReservoirSampler(self.k)
        for z in self.data:
            sampler.sample(z)
        sampled_data = sampler.get_sample()

        # Split data into features and labels
        Xtrain = np.array([item[:-1] for item in sampled_data])
        Ytrain = np.array([item[-1] for item in sampled_data])
        return Xtrain, Ytrain

    def ARIMA_BILSTM(self, epochs):

        X1_train, Y1_train = self.splitting()
        X2_train, Y2_train = self.splitting()

        # ARIMA
        arima_params1 = []
        for item in X1_train:
            # ARIMA Model
            model = ARIMA(item, order=(1, 1, 1))
            # Train the data and append the params
            arima_params1.append(model.fit().params)

        arima_params2 = []
        for item in X2_train:
            # ARIMA Model
            model = ARIMA(item, order=(1, 1, 1))
            # Train the data and append the params
            arima_params2.append(model.fit().params)

        arima_params3 = []
        for item in self.xtest:
            # ARIMA Model
            model = ARIMA(item, order=(1, 1, 1))
            # Train the data and append the params
            arima_params3.append(model.fit().params)

        X1_train_arima = []
        for item, params in zip(X1_train, arima_params1):
            forecast = np.dot(params, item[-len(params):])  # Example forecast using dot product
            X1_train_arima.append(item * forecast)
        X2_train_arima = []
        for item, params in zip(X2_train, arima_params2):
            forecast = np.dot(params, item[-len(params):])  # Example forecast using dot product
            X2_train_arima.append(item * forecast)
        X_test_arima = []
        for item, params in zip(self.xtest, arima_params3):
            forecast = np.dot(params, item[-len(params):])  # Example forecast using dot product
            X_test_arima.append(item * forecast)

        # forecast values to array
        x1train = np.array(X1_train_arima)
        x2train = np.array(X2_train_arima)
        xtest = np.array(X_test_arima)

        x1train = x1train.reshape(x1train.shape[0], x1train.shape[1], 1)
        x2train = x2train.reshape(x2train.shape[0], x2train.shape[1], 1)
        xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)

        input_layer1 = Input(shape=(x1train.shape[1], x1train.shape[2]))
        input_layer2 = Input(shape=(x2train.shape[1], x2train.shape[2]))

        # First BiLSTM model
        model1 = Bidirectional(LSTM(16, return_sequences=True))(input_layer1)
        # Second BiLSTM model
        model2 = Bidirectional(LSTM(16, return_sequences=True))(input_layer2)
        # Concatenate both models
        merged = concatenate([model1, model2])
        merged = Flatten()(merged)
        merged = Dense(128, activation='relu')(merged)
        # Output layer
        output = Dense(1, activation='softmax')(merged)
        # Build the combined
        model = Model(inputs=[input_layer1, input_layer2], outputs=output)
        # Compile the model
        model.compile(optimizer='adam',
                      loss='mse',  # Use appropriate loss function for your task
                      metrics=['MSE'])

        # Train the model
        model.fit([x1train, x2train], [Y1_train, Y2_train], epochs=epochs, batch_size=8)
        pred = model.predict(xtest)
        preds = pred.reshape(pred.shape[0] * pred.shape[1])
        return self.ytest, preds