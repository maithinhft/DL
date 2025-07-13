import subprocess
from training_model import training_model
from test_model import test_model
from model import demo_model
if __name__ == "__main__":
    # demo_model()  # Test model creation

    subprocess.run(["bash", "./download_dataset.sh"]);
    
    print("Training model started...")  # Placeholder for actual training process
    training_model()
    
    accuracy = test_model()
    print(f"\nFinal Accuracy: {accuracy:.4f}")