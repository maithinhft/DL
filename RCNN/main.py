import subprocess
from training_model import training_model
from test_model import test_model
from model import demo_model
from download_dataset import download_dataset
if __name__ == "__main__":
    # demo_model()  # Test model creation

    # subprocess.run(["bash", "./download_dataset.sh"]);
    dataset_path = download_dataset() 
    
    print("Training model started...")  # Placeholder for actual training process
    training_model(dataset_path)
    
    accuracy = test_model(dataset_path)
    print(f"\nFinal Accuracy: {accuracy:.4f}")