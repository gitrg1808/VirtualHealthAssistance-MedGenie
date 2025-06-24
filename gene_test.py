import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

#function to encode genotypes
def encode_genotype(genotype_str):
    genotype = genotype_str.split('(')[-1].split(')')[0]  
    alleles = genotype.split(';')  
    allele_map = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    encoded_alleles = [allele_map.get(allele, -1) for allele in alleles] 
    return encoded_alleles

#load the trained model
model = load_model('genecnn.h5') 

#load the label encoder classes (ensure this is saved previously using np.save)
label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes  

#preprocess the new data (new SNPs, Magnitude, Repute)
new_data = pd.DataFrame({
    'SNP': ['Rs661(A;A)'],  # Example SNPs
    'Magnitude': [9],
    'Repute': [1]
})

#encode the genotypes in the SNP column
new_data['EncodedGenotype'] = new_data['SNP'].apply(encode_genotype)
new_data[['Allele1', 'Allele2']] = pd.DataFrame(new_data['EncodedGenotype'].to_list(), index=new_data.index)
new_data = new_data.drop(columns=['SNP', 'EncodedGenotype'])  

#normalize Magnitude and Repute
scaler = StandardScaler() 
new_data[['Magnitude', 'Repute']] = scaler.fit_transform(new_data[['Magnitude', 'Repute']])

#reshape the input data to match the CNN input shape (samples, features, 1)
new_data_array = new_data.to_numpy()
new_data_reshaped = new_data_array.reshape(new_data_array.shape[0], new_data_array.shape[1], 1)

#predict using the trained model
predictions = model.predict(new_data_reshaped)

#decode the predicted labels (the class numbers) to actual disease names
predicted_classes = np.argmax(predictions, axis=1)  
predicted_diseases = label_encoder.inverse_transform(predicted_classes)

#output the results
print(f'Predicted disease classes: {predicted_classes}')
print(f'Predicted disease names: {predicted_diseases}')
