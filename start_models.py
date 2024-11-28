import subprocess
import os

# Ścieżki do skryptów
script1 = "./models/lenet-5/main.py"
script2 = "./models/gnb/main.py"
script3 = "./models/svm/main.py"

def run_script(script_path):
    """Uruchamia skrypt Python."""
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Skrypt {os.path.basename(script_path)} zakończony pomyślnie.\n")
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Skrypt {os.path.basename(script_path)} zakończony z błędem.\n")
        print("Output:\n", e.stdout)
        print("Error:\n", e.stderr)

if __name__ == "__main__":
    print("Uruchamianie pierwszego skryptu...")
    run_script(script1)

    print("Uruchamianie drugiego skryptu...")
    run_script(script2)

    print("Uruchamianie trzeciego skryptu...")
    run_script(script3)
