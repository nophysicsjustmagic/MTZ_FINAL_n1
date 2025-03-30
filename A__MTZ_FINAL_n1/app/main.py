import subprocess
import sys
from app.api.endpoints import app

def install_packages():
    packages = [
        "python-multipart", "multipart", "numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "tensorflow", "keras", "fastapi", "uvicorn", "sqlalchemy"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


#Загрузка необходимых библиотек
#install_packages()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
    print("End program\n")