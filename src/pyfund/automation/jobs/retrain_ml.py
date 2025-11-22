def retrain_job():
    print("[Automation] Retraining ML models...")
    # In real use, this would call your ML training pipeline
    try:
        from pyfund.ml.predictor import MLPredictor

        predictor = MLPredictor()
        predictor.train_all()
        print("[Automation] ML models retrained successfully")
    except Exception as e:
        print(f"[Automation] ML retraining failed: {e}")
