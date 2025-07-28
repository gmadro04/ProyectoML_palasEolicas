from data_set import WindTurbineDataset

# 1. Cargar y procesar datos
dataset = WindTurbineDataset(r'C:\Users\GMADRO04\Documents\PROYECTOML')
dataset.load_and_process_data()
df = dataset.to_dataframe()

print("Cabecera del frame: \n", df.head())
# 2. Preparar características y etiquetas
# X = np.array([list(item.values()) for item in df['features']])
# y = df['is_defective'].astype(int).values

# 3. Dividir datos
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 4. Construir y evaluar modelos
# models = build_models(X_train, y_train)
# results = evaluate_models(models, X_test, y_test)

# 5. (Opcional) Implementación en PyTorch
# ... código para entrenar modelo personalizado ...