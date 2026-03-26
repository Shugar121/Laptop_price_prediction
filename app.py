from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


try:
    metrics = pickle.load(open("metrics.pkl", "rb"))
    r2, mae = metrics
except:
    r2, mae = None, None

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))    
scaler = pickle.load(open('scaler.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))


df = pd.read_csv('laptop_data.csv')
df = df.drop(columns=["Unnamed: 0"], errors='ignore')


def prepare_input(form_data):
    input_dict = dict.fromkeys(columns, 0)

    
    input_dict['Inches'] = float(form_data['Inches'])
    input_dict['Weight'] = float(form_data['Weight'])

  
    input_dict[f"Company_{form_data['Company']}"] = 1
    input_dict[f"TypeName_{form_data['Type']}"] = 1
    input_dict[f"Ram_{form_data['Ram']}"] = 1
    input_dict[f"Memory_{form_data['Memory']}"] = 1
    input_dict[f"OpSys_{form_data['OpSys']}"] = 1
    input_dict[f"Gpu_{form_data['Gpu']}"] = 1
    input_dict[f"Cpu_{form_data['Cpu']}"] = 1
    input_dict[f"ScreenResolution_{form_data['Resolution']}"] = 1

    return pd.DataFrame([input_dict])


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
       
            data = prepare_input(request.form)
            data_scaled = scaler.transform(data)
            prediction = model.predict(data_scaled)[0]

        except ValueError:
            
            error = "❌ Ошибка: введены неправильные данные! (число/текст перепутаны)"
        except Exception as e:
          
            error = f"❌ Ошибка: {e}"

    return render_template(
        'index.html',
        prediction=prediction,
        error=error,
        tables=[df.head(10).to_html(classes='data')],
        r2=r2,          
        mae=mae         
    )


if __name__ == '__main__':
    app.run(debug=True)