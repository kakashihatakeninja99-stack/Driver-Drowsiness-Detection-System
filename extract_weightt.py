import zipfile

with zipfile.ZipFile('drowsy.keras', 'r') as z:
    print("Files inside:", z.namelist())
    z.extract('model.weights.h5', '.')

print("Extraction done.")
