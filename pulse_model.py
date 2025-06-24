import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

#read the dataset
df = pd.read_csv("C:/Users/mailv/OneDrive/Desktop/sensor1.csv")

#extract Pulse data for anomaly detection
pulse_data = df[['Pulse']]

#scale the data using StandardScaler
scaler = StandardScaler()
pulse_scaled = scaler.fit_transform(pulse_data)

#create the Isolation Forest model
model = IsolationForest(n_estimators=50, contamination=0.1, random_state=2)

#fit the model to the scaled pulse data
model.fit(pulse_scaled)

#make predictions (1 for normal, -1 for anomaly)
pulse_pred = model.predict(pulse_scaled)

#add predictions (Anomaly label) to the dataframe
df['Pulse_Anomaly'] = pulse_pred

#extract the anomalies
pulse_anomalies = df[df['Pulse_Anomaly'] == -1]

#print the anomalies in the specified format
print("Detected Pulse Anomalies (Tabular Format):")
for index, row in pulse_anomalies.iterrows():
    print(f"Anomaly detected in pulse rate at datetime {row['Datetime']} with Pulse Rate = {row['Pulse']}")

#plot the anomaly detection results
plt.figure(figsize=(10, 6))

#define colors for normal and anomalous points
colors = df['Pulse_Anomaly'].map({1: 'blue', -1: 'red'})

#scatter plot for Pulse vs. Anomaly detection results
plt.scatter(df['Datetime'], df['Pulse'], c=colors, label='Normal')

#set title and labels
plt.title('Anomaly Detection in Pulse Data')
plt.xlabel('Datetime')
plt.ylabel('Pulse Rate')
plt.legend()

plt.xticks([])
plt.show()

#save the model and scaler
joblib.dump(model, "pulse_anomaly_detection_model.pkl")
joblib.dump(scaler, "pulse_scaler.pkl")

pulse_anomalies.to_csv("pulse_anomalies.csv", index=False)
