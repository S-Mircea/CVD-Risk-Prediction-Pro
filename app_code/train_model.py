from ml_model import CVDRiskModel
import sys

def main():
    print("Training CVD Risk Assessment Model...")
    print("=" * 50)
    
    # Try both model types and select the best one
    models = ['random_forest', 'logistic_regression']
    best_model = None
    best_accuracy = 0
    
    for model_type in models:
        print(f"\nTraining {model_type.replace('_', ' ').title()} model...")
        model = CVDRiskModel(model_type=model_type)
        
        try:
            accuracy, cv_scores = model.train_model()
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_type = model_type
        
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            continue
    
    if best_model:
        print(f"\n{'='*50}")
        print(f"Best model: {best_model_type.replace('_', ' ').title()}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        
        # Save the best model
        best_model.save_model('cvd_risk_model.pkl')
        print("\nModel training completed successfully!")
        print("You can now run the GUI application.")
    else:
        print("\nError: No model could be trained successfully.")
        sys.exit(1)

if __name__ == "__main__":
    main()