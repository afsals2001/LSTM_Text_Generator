
# README  

## **LSTM Text Generator**  

This project is a text generator application using Long Short-Term Memory (LSTM) neural networks, built with TensorFlow and Streamlit. The app allows users to upload a text file, train an LSTM model on the text, and generate new text based on the trained model.  

---  

## **Features**  

- **File Upload**: Upload a text file for training the LSTM model.  
- **Text Generation**: Generate new text based on the trained LSTM model.  
- **Interactive UI**: Adjust the length and randomness (temperature) of the generated text through sliders.  

---  

## **Requirements**  

### **Python Libraries**  
The following Python libraries are required to run the application:  

- **Streamlit**: `pip install streamlit`  
- **TensorFlow**: `pip install tensorflow`  
- **NumPy**: `pip install numpy`  

---  

## **How to Run the Application**  

1. **Clone or Download the Repository**  
   ```  
   git clone https://github.com/your-repo/lstm-text-generator.git  
   cd generatepoetictext  
   ```  

2. **Install Dependencies**  
   ```  
   pip install -r requirements.txt  
   ```  

3. **Run the Application**  
   ```  
   streamlit run app.py  
   ```  

4. **Access the Application**  
   Open your browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).  

---  

## **How to Use**  

### **1. Upload Text File**  
- Click on the **"Upload Text File for Training"** button.  
- Upload a `.txt` file containing the text you want the model to train on.  

### **2. Train the Model**  
- Click the **"Train Model"** button.  
- The model will train on the uploaded text file. Training might take a few minutes, depending on the text length.  

### **3. Generate Text**  
- Adjust the **Temperature** and **Text Length** using the sliders:  
  - **Temperature**: Controls randomness in text generation (higher values create more diverse text).  
  - **Text Length**: Controls the number of characters in the generated text.  
- Click the **"Generate Text"** button to produce the text.  

### **4. View Generated Text**  
- The generated text will be displayed in a text area below the sliders.  

---  

## **Model Architecture**  

The application uses an LSTM model with the following architecture:  

- **Input Layer**: 40-character sequences.  
- **LSTM Layer**: 128 units.  
- **Dense Layer**: Fully connected to the output layer.  
- **Activation Layer**: Softmax for predicting the probability of the next character.  

---  

## **Customization**  

- **Sequence Length**: Modify the `SEQUENCE_LENGTH` variable in the code to change the size of the input sequences.  
- **Step Size**: Adjust the `STEP_SIZE` variable to change the interval at which sequences are generated.  

---  

## **Limitations**  

- Requires sufficient text data for meaningful training (minimum length: `SEQUENCE_LENGTH + 1` characters).  
- Training may take time depending on the file size and computational power.  

---  

## **Troubleshooting**  

- **File Too Short**: Ensure the uploaded file has more characters than the sequence length (`40` by default).  
- **Training Errors**: Ensure TensorFlow and NumPy are properly installed.  
- **Text Generation Issues**: Verify the model has been trained and saved before generating text.  

---  

## **License**  

This project is licensed under the MIT License. You are free to use, modify, and distribute it with attribution.  

---  

## **Contact**  

For any questions or issues, feel free to contact:  

- **Name**: Afsal S  
- **Email**: afsals2001@gmail.com  
---  
