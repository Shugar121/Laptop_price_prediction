import pickle #save model
import pandas as pd


with open("model.pkl", "rb") as f: #import
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)


print("Введите данные для предсказания цены ноутбука:")

try:
  
    ram = float(input("RAM (GB): "))
    weight = float(input("Weight (kg): "))
    cpu_freq = float(input("CPU Frequency (GHz): "))
    company = input("Company (например Dell, HP): ")

    
    data = {
        "Ram": [ram],
        "Weight": [weight],
        "Cpu_freq": [cpu_freq],
        "Company": [company]
    }
    df = pd.DataFrame(data)

   

    df = pd.get_dummies(df)

   
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0


    df = df[model_columns]

   
    df_scaled = scaler.transform(df)

    
    prediction = model.predict(df_scaled)
    print("💻 Предсказанная цена:", (prediction[0], 2))

except ValueError:
    print("❌ Ошибка: введены неправильные данные! (число/текст перепутаны)")
except Exception as e:
    print("❌ Ошибка:", e)