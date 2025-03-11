import subprocess
import os
from dotenv import load_dotenv


# Charger les variables d'environnement
load_dotenv()

# Utilisation des variables d'environnement
folder_path =  os.getenv("INPUT_PATH")
output_folder_path = os.getenv("DATA_PDF_PATH")


def convert_all_docx_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            input_path = os.path.join(folder_path, filename)
            
            try:
                subprocess.run(["soffice", "--headless", "--convert-to", "pdf", input_path,"--outdir", output_folder_path])
                print(f"✅ Le fichier {filename} a été converti en PDF.")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Erreur lors de la conversion de {filename} : {e}")

convert_all_docx_in_folder(folder_path)
