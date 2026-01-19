import logging
import json
import time
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from pathlib import Path

# --- MLOPS: DYNAMIC PATH INJECTION ---
# This ensures Python looks inside the 'src' folder for our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# --- IMPORT MODULES ---
try:
    # We use aliases (as ...) to keep the variable names consistent
    import optimizer as aadhaar_optimizer
    import forensics as aadhaar_forensics
    import prognosis as aadhaar_prognosis
    import causal as aadhaar_causal
    import gnn as aadhaar_supreme_gnn
except ImportError as e:
    print(f"CRITICAL: Missing Module. Ensure files are in 'src/' folder. Error: {e}")
    sys.exit(1)

# --- MLOPS CONFIGURATION ---
LOG_FILE = "logs/system_execution.log"
AUDIT_FILE = "artifacts/production_audit.json"

# Ensure directories exist (Auto-fix for missing folders)
os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Setup Professional Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

class DataContract:
    """
    Implements 'Data Quality as Code'. 
    Validates schema and statistical distributions before training.
    """
    @staticmethod
    def validate_schema(df: pd.DataFrame, required_cols: list, name: str):
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"[{name}] Schema Violation. Missing columns: {missing}")
        logging.info(f"[{name}] Schema Validation Passed.")

    @staticmethod
    def validate_distribution(df: pd.DataFrame, col: str, min_val: float, max_val: float):
        actual_min = df[col].min()
        actual_max = df[col].max()
        if actual_min < min_val or actual_max > max_val:
            logging.warning(f"Data Drift Detected in '{col}'. Range [{actual_min}, {actual_max}] outside expected [{min_val}, {max_val}].")
        else:
            logging.info(f"Distribution Check Passed for '{col}'.")

class MLOpsOrchestrator:
    """
    The Pipeline Controller.
    Manages the DAG (Directed Acyclic Graph) of the ML workflow.
    """
    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.audit_log = {
            "run_id": self.run_id,
            "status": "INIT",
            "steps": {},
            "performance_metrics": {}
        }
        logging.info(f"--- STARTING PIPELINE RUN ID: {self.run_id} ---")

    def run_stage(self, stage_name, execution_func):
        """
        Executes a pipeline stage with Error Handling and Timing.
        """
        start_time = time.time()
        logging.info(f"🚀 STARTING STAGE: {stage_name}")
        
        try:
            # Execute the module's main logic
            execution_func()
            
            duration = time.time() - start_time
            logging.info(f"✅ STAGE COMPLETED: {stage_name} ({duration:.2f}s)")
            self.audit_log["steps"][stage_name] = {"status": "SUCCESS", "duration": duration}
            
        except Exception as e:
            logging.error(f"❌ STAGE FAILED: {stage_name} | Error: {str(e)}")
            self.audit_log["steps"][stage_name] = {"status": "FAILED", "error": str(e)}
            self.audit_log["status"] = "FAILED"
            self._save_audit()
            raise e

    def _save_audit(self):
        with open(AUDIT_FILE, 'w') as f:
            json.dump(self.audit_log, f, indent=4)

    def validate_input_data(self):
        logging.info("🔍 STAGE: Data Validation (DataOps)")
        try:
            import glob
            files = glob.glob('data/api_data_aadhar_*.csv')
            if not files:
                raise FileNotFoundError("No input data found in /data directory. Please check file paths.")
            
            sample = pd.read_csv(files[0])
            DataContract.validate_schema(sample, ['date', 'district', 'state'], "Raw_Shard")
            
            self.audit_log["steps"]["validation"] = "SUCCESS"
        except Exception as e:
            logging.error(f"Data Validation Failed: {e}")
            raise e

    def execute_pipeline(self):
        # 1. DataOps: Validate Inputs
        self.validate_input_data()
        
        # 2. OpsResearch: Queueing Optimization
        # FIX: Updated to use 'aadhaar_optimizer' (the alias), not 'optimizer'
        self.run_stage("Optimization Engine (M/G/k)", aadhaar_optimizer.optimize_network)
        
        # 3. Deep Learning: Prognosis
        # FIX: Updated to use 'aadhaar_prognosis' (the alias)
        self.run_stage("Deep Temporal Forecasting", aadhaar_prognosis.run_prognosis)
        
        # 4. Spatial AI: Graph Neural Network (GNN)
        # FIX: Updated to use 'aadhaar_supreme_gnn' (the alias)
        self.run_stage("Spatial Graph Learning (GNN)", aadhaar_supreme_gnn.run_supreme_training)

        # 5. Causal AI: ROI Analysis
        # FIX: Updated to use 'aadhaar_causal' (the alias)
        self.run_stage("Double Machine Learning (DML)", aadhaar_causal.run_causal_analysis)
        
        # 6. Final Wrap-up
        self.audit_log["status"] = "SUCCESS"
        self._save_audit()
        logging.info(f"--- PIPELINE COMPLETED SUCCESSFULLY ---")
        logging.info(f"Audit Trail saved to {AUDIT_FILE}")

if __name__ == "__main__":
    orchestrator = MLOpsOrchestrator()
    orchestrator.execute_pipeline()